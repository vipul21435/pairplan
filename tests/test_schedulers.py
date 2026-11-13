import math

import numpy as np
import pytest

from pairplan.engine import PlantedSource, run_schedule
from pairplan.model import fit, information
from pairplan.schedulers import SCHEDULERS, next_pairs, unit_information
from pairplan.votes import Counts, all_pairs

THETA = [1.2, 0.9, 0.1, -0.4, -1.8]
NAMES = [f"s{k}" for k in range(5)]


def some_counts(seed=0, n=120):
    rng = np.random.default_rng(seed)
    source = PlantedSource(THETA, nu=0.3)
    counts = Counts.empty(NAMES)
    for _ in range(n):
        i, j = sorted(rng.choice(5, size=2, replace=False))
        counts.add(int(i), int(j), source.draw(int(i), int(j), rng))
    return counts


def test_unknown_scheduler_and_bad_arguments():
    counts = some_counts()
    with pytest.raises(ValueError, match="unknown scheduler"):
        next_pairs("best", counts)
    with pytest.raises(ValueError):
        next_pairs("uniform", counts, -1)
    with pytest.raises(ValueError):
        next_pairs("adjacent", counts, 1, floor=-1)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_batch_size_and_valid_pairs(scheduler):
    counts = some_counts()
    batch = next_pairs(scheduler, counts, 7, rng=np.random.default_rng(1))
    assert len(batch) == 7
    assert all(0 <= i < j < 5 for i, j in batch)
    assert next_pairs(scheduler, counts, 0, rng=np.random.default_rng(1)) == []


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_deterministic_under_a_seed(scheduler):
    counts = some_counts(seed=3)
    first = next_pairs(scheduler, counts, 12, rng=np.random.default_rng(42))
    second = next_pairs(scheduler, counts, 12, rng=np.random.default_rng(42))
    assert first == second
    runs = [
        run_schedule(
            PlantedSource(THETA, nu=0.3), NAMES, scheduler, budget=150, batch=5,
            rng=np.random.default_rng(7),
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].counts.wins, runs[1].counts.wins)
    assert np.array_equal(runs[0].counts.ties, runs[1].counts.ties)


def test_round_robin_cycles_through_all_pairs():
    counts = Counts.empty(NAMES)
    batch = next_pairs("round-robin", counts, 20)
    assert batch[:10] == all_pairs(5)
    assert batch[10:] == all_pairs(5)
    counts.add(0, 1, "a", 3)
    assert (0, 1) not in next_pairs("round-robin", counts, 9)


def test_uniform_covers_pairs_roughly_evenly():
    counts = Counts.empty(NAMES)
    batch = next_pairs("uniform", counts, 5000, rng=np.random.default_rng(0))
    share = np.array([batch.count(p) for p in all_pairs(5)]) / 5000
    assert share == pytest.approx(np.full(10, 0.1), abs=0.02)


def test_adjacent_asks_only_neighbours_once_the_floor_is_met():
    counts = some_counts(n=400)
    result = fit(counts)
    order = result.order
    neighbours = {tuple(sorted(p)) for p in zip(order[:-1], order[1:], strict=True)}
    batch = next_pairs("adjacent", counts, 8, floor=1, floor_share=0.0)
    assert set(batch) <= neighbours
    # Slots go to the neighbouring pair with the fewest votes, so a long batch evens them out.
    batch = next_pairs("adjacent", counts, 200, floor=1, floor_share=0.0)
    assert set(batch) == neighbours
    after = [counts.pair_totals[i, j] + batch.count((i, j)) for i, j in sorted(neighbours)]
    assert max(after) - min(after) <= 1


def test_adjacent_falls_back_to_nearest_rank_when_neighbours_run_out():
    counts = some_counts(n=400)
    order = fit(counts).order
    capacity = np.full((5, 5), 100)
    for hi, lo in zip(order[:-1], order[1:], strict=True):
        capacity[hi, lo] = capacity[lo, hi] = 0
    batch = next_pairs("adjacent", counts, 3, floor_share=0.0, capacity=capacity)
    position = {k: r for r, k in enumerate(order)}
    assert all(abs(position[i] - position[j]) == 2 for i, j in batch)


def test_unit_information_matches_information():
    theta = np.array([0.5, -0.1, 0.9, -1.3])
    for log_nu in (math.log(0.4), None):
        unit = unit_information(theta, log_nu, all_pairs(4))
        size = 5 if log_nu is not None else 4
        for k, (i, j) in enumerate(all_pairs(4)):
            weights = np.zeros((4, 4))
            weights[i, j] = weights[j, i] = 1.0
            expected = information(theta, log_nu, weights)[:size, :size]
            assert unit[k] == pytest.approx(expected, abs=1e-12)


def test_uncertainty_prefers_the_close_pair():
    # s0 and s1 are nearly tied and everything else is far apart: with equal votes on every
    # pair, the vote that most reduces the risk of a wrong order is on (s0, s1).
    theta = [0.55, 0.45, -3.0, -6.0]
    rng = np.random.default_rng(5)
    source = PlantedSource(theta)
    counts = Counts.empty(["s0", "s1", "s2", "s3"])
    for i, j in all_pairs(4):
        for _ in range(60):
            counts.add(i, j, source.draw(i, j, rng))
    result = fit(counts)
    top_two = tuple(sorted(result.order[:2]))
    assert top_two == (0, 1)
    assert next_pairs("uncertainty", counts, 1, floor_share=0.0) == [(0, 1)]
    batch = next_pairs("uncertainty", counts, 30, floor_share=0.0)
    assert batch.count((0, 1)) >= 20


def test_uncertainty_gain_is_never_negative():
    from pairplan.schedulers import _Uncertainty

    counts = some_counts(seed=9, n=200)
    state = _Uncertainty(fit(counts), counts, all_pairs(5), "full")
    assert np.all(state.gains() >= -1e-12)
    top = _Uncertainty(fit(counts), counts, all_pairs(5), "top-2")
    assert len(top.deciding) == 6 and np.all(top.gains() >= -1e-12)


@pytest.mark.parametrize("scheduler", ["adjacent", "uncertainty"])
def test_floor_serves_unseen_pairs_first(scheduler):
    counts = Counts.empty(NAMES)
    counts.add(0, 1, "a", 50)
    batch = next_pairs(scheduler, counts, 9, floor=1, floor_share=0.0)
    assert sorted(batch) == [p for p in all_pairs(5) if p != (0, 1)]


@pytest.mark.parametrize("scheduler", SCHEDULERS)
@pytest.mark.parametrize("batch", [1, 6])
def test_no_pair_is_starved(scheduler, batch):
    # After 600 votes every pair has at least floor_share times the average votes per pair
    # (minus one batch of slack) under the adaptive rules, and far more under the others.
    run = run_schedule(
        PlantedSource(THETA, nu=0.3), NAMES, scheduler, budget=600, batch=batch,
        rng=np.random.default_rng(11), floor=2, floor_share=0.2,
    )
    totals = run.counts.pair_totals[np.triu_indices(5, 1)]
    assert totals.sum() == 600
    assert totals.min() >= 2
    assert totals.min() >= 0.2 * 60 - batch
    if scheduler == "round-robin":
        assert totals.max() - totals.min() <= 1


def test_capacity_is_respected_and_batch_shrinks():
    counts = Counts.empty(["x", "y", "z"])
    capacity = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    for scheduler in SCHEDULERS:
        batch = next_pairs(scheduler, counts, 10, rng=np.random.default_rng(0), capacity=capacity)
        assert sorted(batch) == [(0, 1), (0, 1), (1, 2)]
    empty = np.zeros((3, 3), dtype=int)
    assert next_pairs("uncertainty", counts, 4, capacity=empty) == []
