import numpy as np
import pytest

from pairplan.engine import PlantedSource, run_schedule
from pairplan.model import fit
from pairplan.stopping import StopRule, check, is_correct, parse_target, target_pairs
from pairplan.votes import Counts


def counts_with(wins):
    names = [f"s{k}" for k in range(len(wins))]
    counts = Counts.empty(names)
    counts.wins[:] = np.array(wins)
    return counts


def test_parse_target():
    assert parse_target("full", 5) is None
    assert parse_target("top-2", 5) == 2
    for bad in ("top-0", "top-5", "top-x", "best"):
        with pytest.raises(ValueError):
            parse_target(bad, 5)


def test_target_pairs_full_and_top_k():
    order = [3, 0, 2, 1]
    assert target_pairs(order, "full") == [(3, 0), (0, 2), (2, 1)]
    assert target_pairs(order, "top-1") == [(3, 0), (3, 2), (3, 1)]
    assert target_pairs(order, "top-2") == [(3, 2), (3, 1), (0, 2), (0, 1)]


def test_is_correct():
    assert is_correct([0, 1, 2, 3], [0, 1, 2, 3], "full")
    assert not is_correct([1, 0, 2, 3], [0, 1, 2, 3], "full")
    assert is_correct([1, 0, 2, 3], [0, 1, 2, 3], "top-2")
    assert not is_correct([0, 2, 1, 3], [0, 1, 2, 3], "top-2")


def test_alpha_per_pair_under_each_correction():
    assert StopRule(0.9).alpha_pair(4, look=7) == pytest.approx(0.1)
    assert StopRule(0.9, correction="pairs").alpha_pair(4, look=7) == pytest.approx(0.025)
    rule = StopRule(0.9, correction="sequential")
    assert rule.alpha_pair(4, look=1) == pytest.approx(0.1 / 4 / 2)
    assert rule.alpha_pair(4, look=3) == pytest.approx(0.1 / 4 / 12)
    # The per-look shares 1 / (t (t + 1)) sum to one, so the total error budget is alpha.
    spent = sum(rule.alpha_pair(1, look=t) for t in range(1, 200001))
    assert spent == pytest.approx(0.1, rel=1e-4)
    assert spent < 0.1


def test_rule_validation():
    with pytest.raises(ValueError):
        StopRule(confidence=1.0)
    with pytest.raises(ValueError):
        StopRule(confidence=0.4)
    with pytest.raises(ValueError):
        StopRule(correction="bonferroni")
    with pytest.raises(ValueError):
        StopRule(target="bottom-2")
    with pytest.raises(ValueError):
        check(fit(counts_with([[0, 5], [3, 0]])), StopRule(), look=0)


def test_check_reports_unsettled_pairs():
    # s0 clearly beats s1 and s2; s1 against s2 is close.
    counts = counts_with([[0, 40, 40], [5, 0, 21], [5, 19, 0]])
    result = fit(counts)
    status = check(result, StopRule(0.95))
    assert not status.settled
    assert [(p.upper, p.lower) for p in status.unsettled] == [("s1", "s2")]
    assert status.needed == pytest.approx(0.95)
    assert len(status.pairs) == 2
    top = check(result, StopRule(0.95, target="top-1"))
    assert top.settled and top.unsettled == []
    strict = check(result, StopRule(0.95, target="top-1", correction="sequential"), look=10**9)
    assert not strict.settled


def test_min_votes_holds_the_rule_back():
    counts = counts_with([[0, 40], [2, 0]])
    result = fit(counts)
    assert check(result, StopRule(0.95)).settled
    assert not check(result, StopRule(0.95, min_votes=100)).settled


def test_run_stops_when_rule_is_met_and_reports_status():
    run = run_schedule(
        PlantedSource([1.5, 0.0, -1.5]), ["a", "b", "c"], "round-robin", budget=2000, batch=3,
        rng=np.random.default_rng(0), rule=StopRule(0.95),
    )
    assert run.stopped and run.votes < 2000
    assert run.status is not None and run.status.settled
    assert run.looks == run.votes // 3
    assert run.counts.total == run.votes


def test_run_from_settled_start_asks_for_nothing():
    start = counts_with([[0, 60], [5, 0]])
    run = run_schedule(
        PlantedSource([1.0, -1.0]), ["s0", "s1"], "uniform", budget=50,
        rng=np.random.default_rng(0), rule=StopRule(0.95), start=start,
    )
    assert run.stopped and run.votes == 0 and run.looks == 1
    assert start.total == 65
