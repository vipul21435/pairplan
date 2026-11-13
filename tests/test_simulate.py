import math

import numpy as np
import pytest

from pairplan.simulate import fixed_design_votes, plan, simulate, true_order
from pairplan.stopping import StopRule
from pairplan.votes import Counts


def test_true_order_breaks_ties_by_position():
    assert true_order([0.1, 0.9, 0.1, -1.0]) == [1, 0, 2, 3]


def test_sequential_rule_keeps_the_wrong_ranking_rate_within_the_stated_level():
    # Stated level: at confidence 0.9 with the sequential correction, at most 10 % of the runs
    # that stop may stop on a wrong order. The correction is a union bound, so the measured
    # rate should sit well below that; the test allows up to 10 %.
    summaries = simulate(
        [0.8, 0.0, -0.8], runs=40, batch=5, budget=1500, seed=0,
        rule=StopRule(0.9, correction="sequential"), schedulers=("round-robin", "uncertainty"),
    )
    for summary in summaries:
        assert summary.stopped >= 36
        assert summary.wrong_rate <= 0.10
        assert summary.votes_median <= summary.votes_p90 <= 1500
        assert summary.tau_mean > 0.95


def test_uncorrected_rule_stops_wrong_more_often_than_its_nominal_level():
    # Three nearly tied systems, nominal 90 % per pair, a look after every batch: the share
    # of stopped runs with a wrong order is far above 10 % (about a third with this seed).
    (summary,) = simulate(
        [0.1, 0.0, -0.1], runs=60, batch=5, budget=300, seed=0, rule=StopRule(0.9),
        schedulers=("uncertainty",),
    )
    assert summary.stopped >= 10
    assert summary.wrong_rate > 0.15
    assert summary.wrong_total >= summary.wrong_stopped


def test_simulation_repeats_under_a_seed_and_changes_with_it():
    kwargs = dict(runs=8, batch=4, budget=200, rule=StopRule(0.9), schedulers=("adjacent",))
    one = simulate([0.6, 0.0, -0.6], seed=3, **kwargs)
    two = simulate([0.6, 0.0, -0.6], seed=3, **kwargs)
    other = simulate([0.6, 0.0, -0.6], seed=4, **kwargs)
    assert one == two
    assert one != other


def test_top_k_target_needs_fewer_votes_than_the_full_order():
    theta = [2.0, 0.1, 0.0, -0.1]
    kwargs = dict(runs=12, batch=4, budget=400, seed=0, schedulers=("uncertainty",))
    (full,) = simulate(theta, rule=StopRule(0.9), **kwargs)
    (top,) = simulate(theta, rule=StopRule(0.9, target="top-1"), **kwargs)
    assert top.stopped == 12 and top.wrong_stopped == 0
    assert top.votes_mean < 0.5 * full.votes_mean


def test_fixed_design_two_systems_closed_form():
    # Two systems, no ties: var(d) = 1 / (n p (1 - p)), so the rule d / se >= z is first met
    # at n = ceil(z^2 / (d^2 p (1 - p))).
    d = 0.8
    p = 1.0 / (1.0 + math.exp(-d))
    z = 1.6448536269514722
    expected = math.ceil(z * z / (d * d * p * (1 - p)))
    for scheduler in ("uniform", "round-robin", "adjacent", "uncertainty"):
        got = fixed_design_votes(
            [0.4, -0.4], scheduler, rule=StopRule(0.95), batch=1, ridge=0.0
        )
        assert got == expected


def test_fixed_design_orders_the_schedulers_and_respects_budget_and_start():
    theta = [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
    votes = {
        s: fixed_design_votes(theta, s, nu=0.4, rule=StopRule(0.95), budget=20000)
        for s in ("uniform", "round-robin", "adjacent", "uncertainty")
    }
    assert votes["uniform"] == votes["round-robin"]
    assert votes["uncertainty"] <= votes["adjacent"] <= votes["uniform"]
    assert fixed_design_votes(theta, "uniform", nu=0.4, rule=StopRule(0.95), budget=100) is None
    start = Counts.empty([f"s{k}" for k in range(6)])
    start.wins[:] = 400
    np.fill_diagonal(start.wins, 0)
    assert fixed_design_votes(theta, "uniform", rule=StopRule(0.95), start=start) == 0
    stricter = fixed_design_votes(
        theta, "uniform", nu=0.4, rule=StopRule(0.95, correction="sequential"), budget=20000
    )
    assert stricter > votes["uniform"]
    with pytest.raises(ValueError):
        fixed_design_votes(theta, "fastest")


def test_plan_adds_the_fixed_design_figure_and_starts_from_votes():
    theta = [0.9, 0.0, -0.9]
    summaries = plan(theta, runs=10, batch=3, budget=600, rule=StopRule(0.9), seed=0)
    assert [s.scheduler for s in summaries] == ["uniform", "round-robin", "adjacent",
                                                "uncertainty"]
    assert all(s.fixed_design is not None and s.fixed_design > 0 for s in summaries)
    start = Counts.empty(["a", "b", "c"])
    start.wins[:] = np.array([[0, 30, 30], [8, 0, 30], [8, 8, 0]])
    after = plan(theta, runs=5, batch=3, budget=600, rule=StopRule(0.9), start=start)
    assert all(s.votes_mean == 0 and s.fixed_design == 0 for s in after)


def test_simulate_rejects_mismatched_names():
    with pytest.raises(ValueError):
        simulate([0.5, -0.5], systems=["only"], runs=1)
