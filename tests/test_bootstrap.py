import numpy as np
import pytest

from pairplan.bootstrap import bootstrap
from pairplan.engine import PlantedSource
from pairplan.model import fit
from pairplan.votes import Counts, Vote, VoteError

NAMES = ["a", "b", "c", "d"]
THETA = [0.9, 0.3, -0.2, -1.0]


def make_votes(n=600, seed=0):
    rng = np.random.default_rng(seed)
    source = PlantedSource(THETA, nu=0.4)
    votes = []
    for k in range(n):
        i, j = (int(x) for x in rng.choice(4, size=2, replace=False))
        votes.append(Vote(NAMES[i], NAMES[j], source.draw(i, j, rng), f"judge-{k % 12}",
                          f"prompt-{k % 30}"))
    return votes


def test_bootstrap_spread_matches_model_standard_errors():
    votes = make_votes()
    counts = Counts.from_votes(votes, NAMES)
    result = fit(counts)
    boot = bootstrap(counts, n_boot=400, seed=1)
    assert boot.thetas.shape == (400, 4)
    assert boot.thetas.sum(axis=1) == pytest.approx(np.zeros(400), abs=1e-9)
    assert boot.thetas.std(axis=0) == pytest.approx(result.se, rel=0.2)
    intervals = boot.intervals
    assert np.all(intervals[:, 0] < result.theta) and np.all(result.theta < intervals[:, 1])
    ranks = boot.rank_intervals
    assert np.all(ranks[:, 0] <= ranks[:, 1]) and ranks.min() >= 1 and ranks.max() <= 4
    assert boot.order_share(0, 3) == pytest.approx(1.0)
    assert boot.order_share(0, 1) + boot.order_share(1, 0) == pytest.approx(1.0)
    assert boot.order_share(0, 1) == pytest.approx(result.prob_order(0, 1), abs=0.08)


def test_bootstrap_is_seeded():
    counts = Counts.from_votes(make_votes(200), NAMES)
    one = bootstrap(counts, n_boot=30, seed=5)
    two = bootstrap(counts, n_boot=30, seed=5)
    other = bootstrap(counts, n_boot=30, seed=6)
    assert np.array_equal(one.thetas, two.thetas)
    assert not np.array_equal(one.thetas, other.thetas)


def test_resampling_keeps_the_number_of_votes(monkeypatch):
    import pairplan.bootstrap as module

    counts = Counts.from_votes(make_votes(150), NAMES)
    seen = []
    real = module.fit_model

    def spy(sample, **kwargs):
        seen.append(sample.total)
        assert np.array_equal(sample.ties, sample.ties.T)
        return real(sample, **kwargs)

    monkeypatch.setattr(module, "fit_model", spy)
    bootstrap(counts, n_boot=10, seed=0)
    assert seen == [150] * 10


def test_cluster_bootstrap_by_judge_and_prompt():
    votes = make_votes()
    counts = Counts.from_votes(votes, NAMES)
    by_judge = bootstrap(counts, n_boot=100, seed=2, votes=votes, cluster="judge")
    by_prompt = bootstrap(counts, n_boot=100, seed=2, votes=votes, cluster="prompt")
    assert by_judge.unit == "judge" and by_prompt.unit == "prompt"
    result = fit(counts)
    for boot in (by_judge, by_prompt):
        assert boot.thetas.mean(axis=0) == pytest.approx(result.theta, abs=0.1)
        assert np.all(boot.thetas.std(axis=0) > 0.02)


def test_bootstrap_errors():
    votes = make_votes(50)
    counts = Counts.from_votes(votes, NAMES)
    with pytest.raises(ValueError):
        bootstrap(counts, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap(counts, level=1.5)
    with pytest.raises(ValueError):
        bootstrap(counts, cluster="system")
    with pytest.raises(ValueError, match="needs the votes"):
        bootstrap(counts, cluster="judge")
    with pytest.raises(VoteError, match="no judge"):
        bootstrap(counts, votes=[Vote("a", "b", "a")], cluster="judge")
    with pytest.raises(VoteError, match="no votes"):
        bootstrap(Counts.empty(NAMES))
