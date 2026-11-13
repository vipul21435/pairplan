import numpy as np
import pytest
from scipy.stats import kendalltau

from pairplan.engine import PlantedSource, PoolSource, VotePool, run_schedule
from pairplan.metrics import kendall_tau
from pairplan.replay import ReplayCurve, replay
from pairplan.schedulers import SCHEDULERS
from pairplan.votes import Counts, Vote

NAMES = ["a", "b", "c", "d"]
THETA = [1.0, 0.4, -0.3, -1.1]


def make_votes(n=240, seed=0):
    """Uneven pools: pair (a, b) gets few votes, so it runs out early under adaptive rules."""
    rng = np.random.default_rng(seed)
    source = PlantedSource(THETA, nu=0.3)
    votes = []
    while len(votes) < n:
        i, j = (int(x) for x in rng.choice(4, size=2, replace=False))
        if {i, j} == {0, 1} and rng.random() < 0.7:
            continue
        outcome = source.draw(i, j, rng)
        votes.append(Vote(NAMES[i], NAMES[j], outcome))
    return votes


def test_kendall_tau_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = [int(x) for x in rng.permutation(7)]
        b = [int(x) for x in rng.permutation(7)]
        rank_a = np.argsort(a)
        rank_b = np.argsort(b)
        assert kendall_tau(a, b) == pytest.approx(kendalltau(rank_a, rank_b).statistic)
    assert kendall_tau([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert kendall_tau([0, 1, 2], [2, 1, 0]) == pytest.approx(-1.0)
    assert kendall_tau([5], [5]) == 1.0
    with pytest.raises(ValueError):
        kendall_tau([0, 1], [0, 2])


def test_pool_groups_votes_by_pair_and_flips_outcomes():
    votes = [Vote("b", "a", "a"), Vote("a", "b", "a"), Vote("a", "b", "tie"), Vote("c", "a", "b")]
    pool = VotePool(votes, ["a", "b", "c"])
    assert pool.by_pair[(0, 1)] == [(0, "b"), (1, "a"), (2, "tie")]
    assert pool.by_pair[(0, 2)] == [(3, "a")]
    assert pool.by_pair[(1, 2)] == []
    assert pool.sizes().tolist() == [[0, 3, 1], [3, 0, 0], [1, 0, 0]]


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_replay_never_uses_a_vote_twice(scheduler):
    votes = make_votes()
    pool = VotePool(votes, NAMES)
    full = Counts.from_votes(votes, NAMES)
    for budget in (100, len(votes)):
        rng = np.random.default_rng(3)
        source = PoolSource(pool, rng)
        run = run_schedule(source, NAMES, scheduler, budget=budget, batch=7, rng=rng)
        assert run.votes == budget
        assert len(source.used) == budget
        assert len(set(source.used)) == budget
        # What the scheduler saw is exactly the votes that were handed out.
        assert np.array_equal(
            run.counts.wins, Counts.from_votes([votes[k] for k in source.used], NAMES).wins
        )
        assert np.all(run.counts.wins <= full.wins) and np.all(run.counts.ties <= full.ties)
        assert np.all(source.capacity() == full.pair_totals - run.counts.pair_totals)
    # With the whole pool as budget every vote is used exactly once.
    assert sorted(source.used) == list(range(len(votes)))
    assert np.array_equal(run.counts.wins, full.wins)
    assert np.array_equal(run.counts.ties, full.ties)


def test_run_ends_when_the_pool_is_empty():
    votes = make_votes(60)
    rng = np.random.default_rng(0)
    source = PoolSource(VotePool(votes, NAMES), rng)
    run = run_schedule(source, NAMES, "uncertainty", budget=500, batch=8, rng=rng)
    assert run.votes == 60 and run.exhausted
    assert run.first_exhausted_at is not None and run.first_exhausted_at <= 60
    with pytest.raises(RuntimeError, match="no votes left"):
        source.draw(0, 1, rng)


def test_replay_curves_end_on_the_full_data_ranking():
    votes = make_votes()
    result = replay(votes, NAMES, runs=6, batch=8, every=40, seed=1)
    assert result.total_votes == 240 and result.budget == 240 and result.every == 40
    assert result.pool_min < result.pool_max
    assert [c.scheduler for c in result.curves] == list(SCHEDULERS)
    for curve in result.curves:
        assert curve.checkpoints == (40, 80, 120, 160, 200, 240)
        assert curve.tau_mean[-1] == pytest.approx(1.0)
        assert curve.exact_share[-1] == pytest.approx(1.0)
        assert all(-1.0 <= t <= 1.0 for t in curve.tau_mean)
        assert all(p10 <= mean + 1e-12 for p10, mean in zip(curve.tau_p10, curve.tau_mean,
                                                           strict=True))
        assert len(curve.settle) == 6 and all(s is not None for s in curve.settle)
        assert all(x is not None for x in curve.first_exhausted)


def test_replay_is_seeded_and_budget_and_every_are_respected():
    votes = make_votes()
    one = replay(votes, NAMES, runs=3, batch=10, budget=100, every=25, seed=2,
                 schedulers=("uniform", "adjacent"))
    two = replay(votes, NAMES, runs=3, batch=10, budget=100, every=25, seed=2,
                 schedulers=("uniform", "adjacent"))
    assert one.curves == two.curves
    assert one.every == 30  # rounded up to a multiple of the batch
    assert one.curves[0].checkpoints == (30, 60, 90, 100)
    with pytest.raises(ValueError):
        replay([], ["a", "b"])


def test_curve_thresholds_need_the_curve_to_stay_above():
    curve = ReplayCurve(
        scheduler="x",
        checkpoints=(10, 20, 30, 40),
        tau_mean=(0.95, 0.80, 0.92, 0.97),
        tau_p10=(0.5, 0.5, 0.5, 0.5),
        exact_share=(0.6, 0.4, 0.4, 0.9),
        settle=(10, None),
        first_exhausted=(None, 30),
    )
    assert curve.votes_to_tau(0.9) == 30
    assert curve.votes_to_tau(0.96) == 40
    assert curve.votes_to_tau(0.99) is None
    assert curve.votes_to_exact(0.5) == 40
