import math

import numpy as np
import pytest
from scipy.optimize import minimize

from pairplan.model import FitError, fit, information, mle_obstacle, outcome_probs
from pairplan.votes import Counts


def make_counts(m, seed, n_per_pair=40, nu=0.5, spread=0.8):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, spread, size=m)
    counts = Counts.empty([f"s{k}" for k in range(m)])
    win, tie = outcome_probs(theta, math.log(nu) if nu > 0 else None)
    for i in range(m):
        for j in range(i + 1, m):
            draws = rng.multinomial(n_per_pair, [win[i, j], win[j, i], tie[i, j]])
            counts.add(i, j, "a", int(draws[0]))
            counts.add(i, j, "b", int(draws[1]))
            counts.add(i, j, "tie", int(draws[2]))
    return counts, theta


def reference_negloglik(params, counts, with_ties):
    """Direct, loop-based negative log-likelihood, written independently of the package."""
    m = counts.m
    theta = np.append(params[: m - 1], -np.sum(params[: m - 1]))
    nu = math.exp(params[m - 1]) if with_ties else 0.0
    total = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pi, pj = math.exp(theta[i]), math.exp(theta[j])
            pt = nu * math.sqrt(pi * pj)
            denom = pi + pj + pt
            total -= counts.wins[i, j] * math.log(pi / denom)
            total -= counts.wins[j, i] * math.log(pj / denom)
            if counts.ties[i, j]:
                total -= counts.ties[i, j] * math.log(pt / denom)
    return total


def test_two_systems_closed_form():
    # With two systems the model is saturated: fitted probabilities equal observed
    # frequencies, so theta_1 - theta_2 = log(w1 / w2) and nu = t / sqrt(w1 * w2).
    counts = Counts.empty(["x", "y"])
    counts.add(0, 1, "a", 30)
    counts.add(0, 1, "b", 12)
    counts.add(0, 1, "tie", 9)
    result = fit(counts, ridge=0.0)
    assert result.converged
    assert result.theta[0] - result.theta[1] == pytest.approx(math.log(30 / 12), abs=1e-8)
    assert result.nu == pytest.approx(9 / math.sqrt(30 * 12), abs=1e-8)
    assert result.theta.sum() == pytest.approx(0.0, abs=1e-12)
    p_x, p_y, p_tie = result.win_prob(0, 1)
    assert (p_x, p_y, p_tie) == pytest.approx((30 / 51, 12 / 51, 9 / 51), abs=1e-8)
    assert result.ranking == ["x", "y"]


def test_two_systems_no_ties_standard_error():
    # Plain Bradley-Terry with two systems: d = log(w1 / w2), var(d) = 1 / w1 + 1 / w2.
    counts = Counts.empty(["x", "y"])
    counts.add(0, 1, "a", 40)
    counts.add(0, 1, "b", 10)
    result = fit(counts, ridge=0.0)
    assert result.log_nu is None and result.nu == 0.0
    diff, se = result.difference(0, 1)
    assert diff == pytest.approx(math.log(4.0), abs=1e-8)
    assert se == pytest.approx(math.sqrt(1 / 40 + 1 / 10), rel=1e-6)
    assert result.prob_order(0, 1) == pytest.approx(0.5 * math.erfc(-diff / se / math.sqrt(2)))
    assert result.prob_order(1, 0) == pytest.approx(1 - result.prob_order(0, 1))


@pytest.mark.parametrize("seed,nu", [(0, 0.5), (1, 0.2), (2, 0.0), (3, 1.5)])
def test_fit_matches_direct_optimiser(seed, nu):
    counts, _ = make_counts(5, seed, nu=nu)
    with_ties = counts.ties.sum() > 0
    result = fit(counts, ridge=0.0)
    m = counts.m
    x0 = np.zeros(m if with_ties else m - 1)
    best = minimize(
        reference_negloglik, x0, args=(counts, with_ties), method="BFGS", options={"gtol": 1e-8}
    )
    theta_ref = np.append(best.x[: m - 1], -np.sum(best.x[: m - 1]))
    assert result.theta == pytest.approx(theta_ref, abs=2e-5)
    if with_ties:
        assert result.log_nu == pytest.approx(best.x[m - 1], abs=2e-5)
    assert -result.loglik == pytest.approx(best.fun, abs=1e-6)


def test_covariance_matches_numeric_hessian():
    counts, _ = make_counts(4, 11, n_per_pair=60)
    result = fit(counts, ridge=0.0)
    m = counts.m
    x = np.append(result.theta[: m - 1], result.log_nu)
    h = 1e-4
    size = x.size
    hess = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            def f(dr, dc, r=r, c=c):
                y = x.copy()
                y[r] += dr
                y[c] += dc
                return reference_negloglik(y, counts, True)
            hess[r, c] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    cov_reduced = np.linalg.inv(hess)
    # Map the (m - 1 free strengths, log nu) parametrisation back to centred theta.
    jac = np.zeros((m + 1, size))
    jac[: m - 1, : m - 1] = np.eye(m - 1)
    jac[m - 1, : m - 1] = -1.0
    jac[m, m - 1] = 1.0
    expected = jac @ cov_reduced @ jac.T
    assert result.cov_full == pytest.approx(expected, rel=1e-3, abs=1e-6)
    assert result.cov.sum(axis=1) == pytest.approx(np.zeros(m), abs=1e-10)


def test_information_rows_sum_to_zero_over_strengths():
    theta = np.array([0.3, -0.2, 1.0, -1.1])
    weights = np.full((4, 4), 5.0)
    np.fill_diagonal(weights, 0.0)
    info = information(theta, math.log(0.4), weights)
    assert info[:, :4].sum(axis=1) == pytest.approx(np.zeros(5), abs=1e-12)
    assert np.all(np.linalg.eigvalsh(info) > -1e-10)


def test_recovers_planted_strengths():
    counts, theta = make_counts(6, 5, n_per_pair=4000, nu=0.6)
    result = fit(counts)
    assert result.theta == pytest.approx(theta - theta.mean(), abs=0.08)
    assert result.nu == pytest.approx(0.6, abs=0.05)


def test_ridge_keeps_unbeaten_system_finite_and_mle_raises():
    counts = Counts.empty(["x", "y", "z"])
    counts.add(0, 1, "a", 5)
    counts.add(0, 2, "a", 5)
    counts.add(1, 2, "a", 3)
    counts.add(1, 2, "b", 2)
    with pytest.raises(FitError):
        fit(counts, ridge=0.0)
    result = fit(counts, ridge=0.1)
    assert result.converged
    assert np.all(np.isfinite(result.theta))
    assert result.ranking[0] == "x"


def test_no_votes_gives_flat_fit_with_ridge():
    counts = Counts.empty(["x", "y", "z"])
    result = fit(counts)
    assert result.theta == pytest.approx(np.zeros(3))
    assert result.ranking == ["x", "y", "z"]
    assert [p.prob_correct for p in result.adjacent()] == pytest.approx([0.5, 0.5])


def test_adjacent_pairs_follow_ranking():
    counts, _ = make_counts(5, 8, n_per_pair=200)
    result = fit(counts)
    pairs = result.adjacent()
    assert [p.upper for p in pairs] == result.ranking[:-1]
    assert [p.lower for p in pairs] == result.ranking[1:]
    assert all(p.difference >= 0 and 0.5 <= p.prob_correct <= 1 for p in pairs)


def test_warm_start_gives_same_answer():
    counts, _ = make_counts(5, 9)
    cold = fit(counts)
    counts.add(0, 1, "a", 3)
    again = fit(counts, start=cold)
    fresh = fit(counts)
    assert again.theta == pytest.approx(fresh.theta, abs=1e-7)
    assert again.iterations <= fresh.iterations


def test_negative_ridge_rejected():
    with pytest.raises(ValueError):
        fit(Counts.empty(["x", "y"]), ridge=-1.0)


def counts_from(names, wins=(), ties=()):
    counts = Counts.empty(names)
    index = {name: k for k, name in enumerate(names)}
    for winner, loser, k in wins:
        counts.add(index[winner], index[loser], "a", k)
    for one, other, k in ties:
        counts.add(index[one], index[other], "tie", k)
    return counts


def test_unbeaten_system_has_no_mle_and_is_named():
    counts = counts_from("xyz", wins=[("x", "y", 5), ("x", "z", 5), ("y", "z", 3), ("z", "y", 2)])
    assert "'x' never lost or tied" in mle_obstacle(counts)
    with pytest.raises(FitError, match="'x' never lost or tied a vote"):
        fit(counts, ridge=0.0)


def test_winless_system_has_no_mle_and_is_named():
    counts = counts_from("xyz", wins=[("x", "z", 4), ("y", "z", 4), ("x", "y", 3), ("y", "x", 2)])
    with pytest.raises(FitError, match="'z' never won or tied a vote"):
        fit(counts, ridge=0.0)


def test_group_that_never_lost_to_the_rest_has_no_mle_and_is_named():
    # a and b beat each other, c and d beat each other, and the first group always beats the
    # second: every system has wins and losses, but the groups cannot be placed.
    wins = [("a", "b", 3), ("b", "a", 2), ("c", "d", 2), ("d", "c", 4), ("a", "c", 5),
            ("b", "d", 1)]
    counts = counts_from("abcd", wins=wins)
    with pytest.raises(FitError, match="the group 'a', 'b' never lost to or tied with"):
        fit(counts, ridge=0.0)
    # One win back from the lower group connects the graph and the fit exists.
    counts.add(3, 0, "a")
    assert mle_obstacle(counts) is None
    result = fit(counts, ridge=0.0)
    assert result.converged and np.all(np.isfinite(result.theta))


def test_unseen_and_unconnected_systems_have_no_mle():
    counts = counts_from("xyz", wins=[("x", "y", 2), ("y", "x", 2)])
    with pytest.raises(FitError, match="'z' has no votes"):
        fit(counts, ridge=0.0)
    wins = [("a", "b", 2), ("b", "a", 2), ("c", "d", 2), ("d", "c", 2)]
    with pytest.raises(FitError, match="never lost to or tied with a system outside it"):
        fit(counts_from("abcd", wins=wins), ridge=0.0)


def test_connected_case_fits_at_ridge_zero():
    wins = [("x", "y", 6), ("y", "x", 2), ("y", "z", 5), ("z", "y", 3), ("z", "x", 1),
            ("x", "z", 7)]
    counts = counts_from("xyz", wins=wins, ties=[("x", "y", 2), ("y", "z", 1)])
    assert mle_obstacle(counts) is None
    result = fit(counts, ridge=0.0)
    assert result.converged
    assert result.ranking == ["x", "y", "z"]
    assert result.theta.sum() == pytest.approx(0.0, abs=1e-10)
    assert 0.0 < result.nu < 5.0


def test_a_tie_connects_the_graph_when_a_cycle_has_more_wins_than_ties():
    # x beat y, y beat z, z tied x: the cycle x -> y -> z -> x has two wins and one tie.
    counts = counts_from("xyz", wins=[("x", "y", 1), ("y", "z", 1)], ties=[("z", "x", 1)])
    assert mle_obstacle(counts) is None
    result = fit(counts, ridge=0.0)
    assert result.converged and np.all(np.abs(result.theta) < 20)
    # A system that only ever tied is placed by its ties once wins go both ways elsewhere.
    counts = counts_from("xyz", wins=[("x", "y", 3), ("y", "x", 2)], ties=[("x", "z", 4)])
    assert mle_obstacle(counts) is None
    assert fit(counts, ridge=0.0).converged


def test_tie_parameter_unbounded_cases_have_no_mle():
    # One system beat and tied the other but never lost: fitted frequencies would need a
    # zero win probability for the loser and an infinite tie parameter.
    counts = counts_from("xy", wins=[("x", "y", 3)], ties=[("x", "y", 2)])
    with pytest.raises(FitError, match="tie parameter is unbounded"):
        fit(counts, ridge=0.0)
    only_ties = counts_from("xyz", ties=[("x", "y", 2), ("y", "z", 2), ("x", "z", 1)])
    with pytest.raises(FitError, match="tie parameter is unbounded"):
        fit(only_ties, ridge=0.0)
    assert np.all(np.isfinite(fit(only_ties, ridge=0.01).theta))


@pytest.mark.parametrize("seed", range(12))
def test_structural_check_agrees_with_what_the_likelihood_does(seed):
    # Sparse random results on four systems. Where the check says the estimate exists the
    # fit converges to moderate values; where it says it does not, a weakly penalised fit
    # runs off: some strength or the tie parameter becomes large.
    rng = np.random.default_rng(seed)
    counts = Counts.empty(["a", "b", "c", "d"])
    for _ in range(7):
        i, j = (int(x) for x in rng.choice(4, size=2, replace=False))
        counts.add(i, j, str(rng.choice(["a", "b", "tie"], p=[0.4, 0.3, 0.3])))
    obstacle = mle_obstacle(counts)
    if obstacle is None:
        result = fit(counts, ridge=0.0)
        assert result.converged
        assert np.abs(result.theta).max() < 15 and abs(result.log_nu or 0.0) < 15
    else:
        with pytest.raises(FitError):
            fit(counts, ridge=0.0)
        if "has no votes" in obstacle:
            return  # that strength is undetermined rather than infinite
        loose = fit(counts, ridge=1e-7)
        spread = np.abs(loose.theta).max()
        assert spread > 4 or abs(loose.log_nu or 0.0) > 4
