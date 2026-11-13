import math

import numpy as np
import pytest
from scipy.optimize import minimize

from pairplan.model import FitError, fit, information, outcome_probs
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
