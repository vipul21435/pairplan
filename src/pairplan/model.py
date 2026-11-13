"""Bradley-Terry strengths with ties (Davidson's model), fitted by Newton's method.

For systems ``i`` and ``j`` with log-strengths ``theta_i`` and ``theta_j`` and a tie parameter
``nu >= 0``, one vote has three outcomes::

    P(i wins) = exp(theta_i) / D
    P(j wins) = exp(theta_j) / D
    P(tie)    = nu * exp((theta_i + theta_j) / 2) / D
    D         = exp(theta_i) + exp(theta_j) + nu * exp((theta_i + theta_j) / 2)

With ``nu = 0`` this is the Bradley-Terry model. In ``(theta, log nu)`` the model is a
three-category multinomial logit with linear predictors, so the log-likelihood is concave and
the observed information equals the expected information. The strengths are identified up to a
shift; they are reported centred (``sum(theta) == 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .votes import Counts


class FitError(RuntimeError):
    """The maximum-likelihood estimate does not exist or was not reached."""


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


_erfc = np.frompyfunc(math.erfc, 1, 1)


def normal_sf(z: np.ndarray) -> np.ndarray:
    """Upper tail of the standard normal, elementwise (numpy has no erfc of its own)."""
    return 0.5 * _erfc(np.asarray(z, dtype=float) / math.sqrt(2.0)).astype(float)


def outcome_probs(theta: np.ndarray, log_nu: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Matrices ``win[i, j] = P(i beats j)`` and ``tie[i, j] = P(tie between i and j)``.

    ``log_nu=None`` means no ties (``nu = 0``). Diagonals are meaningless.
    """
    a = theta[:, None]
    b = theta[None, :]
    if log_nu is None:
        top = np.maximum(a, b)
        wa = np.exp(a - top)
        wb = np.exp(b - top)
        return wa / (wa + wb), np.zeros_like(wa)
    t = log_nu + 0.5 * (a + b)
    top = np.maximum(np.maximum(a, b), t)
    wa = np.exp(a - top)
    wb = np.exp(b - top)
    wt = np.exp(t - top)
    denom = wa + wb + wt
    return wa / denom, wt / denom


def information(
    theta: np.ndarray, log_nu: float | None, weights: np.ndarray
) -> np.ndarray:
    """Fisher information of ``(theta, log nu)`` for ``weights[i, j]`` votes on each pair.

    ``weights`` is a symmetric matrix of vote counts per pair (zero diagonal). The result is
    ``(m + 1) x (m + 1)``; the last row and column belong to ``log nu`` and are zero when
    ``log_nu`` is None.
    """
    m = theta.shape[0]
    win, tie = outcome_probs(theta, log_nu)
    score = win + 0.5 * tie
    info = np.zeros((m + 1, m + 1))
    off = weights * (0.25 * tie - score * score.T)
    np.fill_diagonal(off, 0.0)
    diag = weights * (win + 0.25 * tie - score * score)
    np.fill_diagonal(diag, 0.0)
    info[:m, :m] = off
    info[np.arange(m), np.arange(m)] = diag.sum(axis=1)
    if log_nu is not None:
        cross = weights * (0.5 * tie - score * tie)
        np.fill_diagonal(cross, 0.0)
        info[:m, m] = cross.sum(axis=1)
        info[m, :m] = info[:m, m]
        tt = weights * (tie - tie * tie)
        np.fill_diagonal(tt, 0.0)
        info[m, m] = 0.5 * tt.sum()
    return info


def _loglik(counts: Counts, theta: np.ndarray, log_nu: float | None, ridge: float) -> float:
    win, tie = outcome_probs(theta, log_nu)
    mask = counts.wins > 0
    value = float((counts.wins[mask] * np.log(win[mask])).sum())
    if log_nu is not None:
        tmask = counts.ties > 0
        value += 0.5 * float((counts.ties[tmask] * np.log(tie[tmask])).sum())
        value -= 0.5 * ridge * log_nu * log_nu
    return value - 0.5 * ridge * float(theta @ theta)


@dataclass(frozen=True)
class AdjacentPair:
    """Two neighbours in the fitted ranking and how sure their order is."""

    upper: str
    lower: str
    difference: float
    se: float
    prob_correct: float


@dataclass(frozen=True)
class Fit:
    """A fitted model. ``theta`` is centred; ``cov`` is its covariance (rows sum to zero)."""

    systems: tuple[str, ...]
    theta: np.ndarray
    log_nu: float | None
    cov: np.ndarray
    cov_full: np.ndarray
    loglik: float
    n_votes: int
    ridge: float
    iterations: int
    converged: bool

    @property
    def nu(self) -> float:
        return 0.0 if self.log_nu is None else math.exp(self.log_nu)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    @property
    def order(self) -> list[int]:
        """System indices from strongest to weakest; equal strengths keep input order."""
        return sorted(range(len(self.systems)), key=lambda k: (-self.theta[k], k))

    @property
    def ranking(self) -> list[str]:
        return [self.systems[k] for k in self.order]

    def difference(self, i: int, j: int) -> tuple[float, float]:
        """``theta_i - theta_j`` and its standard error."""
        var = self.cov[i, i] + self.cov[j, j] - 2.0 * self.cov[i, j]
        return float(self.theta[i] - self.theta[j]), math.sqrt(max(float(var), 0.0))

    def prob_order(self, i: int, j: int) -> float:
        """Normal-approximation probability that system ``i`` is truly stronger than ``j``."""
        diff, se = self.difference(i, j)
        if se == 0.0:
            return 0.5 if diff == 0.0 else float(diff > 0.0)
        return normal_cdf(diff / se)

    def adjacent(self) -> list[AdjacentPair]:
        """Each neighbouring pair of the ranking with the probability its order is right."""
        order = self.order
        pairs = []
        for hi, lo in zip(order[:-1], order[1:], strict=True):
            diff, se = self.difference(hi, lo)
            pairs.append(
                AdjacentPair(self.systems[hi], self.systems[lo], diff, se, self.prob_order(hi, lo))
            )
        return pairs

    def win_prob(self, i: int, j: int) -> tuple[float, float, float]:
        """Model probabilities ``(i wins, j wins, tie)`` for one vote on the pair."""
        win, tie = outcome_probs(self.theta, self.log_nu)
        return float(win[i, j]), float(win[j, i]), float(tie[i, j])


def covariance(info: np.ndarray, m: int, ridge: float, has_ties: bool) -> np.ndarray:
    """Covariance of ``(theta, log nu)`` from the information, with ``theta`` centred.

    The information is singular along the shift direction ``(1, ..., 1, 0)``; adding the
    projector onto that direction makes it invertible, and centring the result removes the
    added term again, which gives the pseudo-inverse when ``ridge == 0``.
    """
    size = m + 1 if has_ties else m
    a = info[:size, :size].copy()
    a[np.arange(size), np.arange(size)] += ridge
    a[:m, :m] += 1.0 / m
    inv = np.linalg.inv(a)
    centre = np.eye(size)
    centre[:m, :m] -= 1.0 / m
    full = np.zeros((m + 1, m + 1))
    full[:size, :size] = centre @ inv @ centre.T
    return full


def fit(
    counts: Counts,
    *,
    ridge: float = 0.01,
    tol: float = 1e-9,
    max_iter: int = 200,
    start: Fit | None = None,
) -> Fit:
    """Fit the model to ``counts`` by penalised maximum likelihood.

    ``ridge`` is the weight of a quadratic penalty ``ridge / 2 * (|theta|^2 + (log nu)^2)``,
    i.e. a weak Gaussian prior with standard deviation ``ridge ** -0.5`` that keeps the
    estimate finite when a system has won or lost every vote or the comparison graph is not
    connected. ``ridge=0`` gives the plain maximum-likelihood estimate and raises
    :class:`FitError` when it does not exist. When the data contain no ties, ``nu`` is fixed at
    zero (the Bradley-Terry model) and not estimated.
    """
    if ridge < 0:
        raise ValueError("ridge must be >= 0")
    m = counts.m
    weights = counts.pair_totals.astype(float)
    wins = counts.wins.astype(float)
    ties = counts.ties.astype(float)
    n_ties = float(ties.sum()) / 2.0
    has_ties = n_ties > 0
    size = m + 1 if has_ties else m

    theta = np.zeros(m)
    log_nu: float | None = 0.0 if has_ties else None
    if start is not None and start.theta.shape == (m,):
        theta = start.theta - start.theta.mean()
        if has_ties and start.log_nu is not None:
            log_nu = start.log_nu

    row_score = (wins + 0.5 * ties).sum(axis=1)
    value = _loglik(counts, theta, log_nu, ridge)
    converged = False
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        win, tie = outcome_probs(theta, log_nu)
        grad = np.zeros(size)
        grad[:m] = row_score - (weights * (win + 0.5 * tie)).sum(axis=1) - ridge * theta
        if has_ties:
            assert log_nu is not None
            grad[m] = n_ties - 0.5 * float((weights * tie).sum()) - ridge * log_nu
        hess = information(theta, log_nu, weights)[:size, :size]
        hess[np.arange(size), np.arange(size)] += ridge
        hess[:m, :m] += 1.0 / m
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        scale = 1.0
        while True:
            new_theta = theta + scale * step[:m]
            new_theta = new_theta - new_theta.mean()
            new_nu = log_nu + scale * float(step[m]) if has_ties and log_nu is not None else None
            new_value = _loglik(counts, new_theta, new_nu, ridge)
            if new_value >= value - 1e-12 or scale < 1e-8:
                break
            scale *= 0.5
        moved = float(np.max(np.abs(scale * step)))
        theta, log_nu, value = new_theta, new_nu, new_value
        if moved < tol:
            converged = True
            break
        if float(np.max(np.abs(theta))) > 30.0 or (log_nu is not None and abs(log_nu) > 30.0):
            break

    if not converged and ridge == 0:
        raise FitError(
            "no finite maximum-likelihood estimate: some system has won or lost every vote, "
            "or the comparison graph is not connected; use ridge > 0"
        )
    info = information(theta, log_nu, weights)
    try:
        cov_full = covariance(info, m, ridge, has_ties)
    except np.linalg.LinAlgError:
        raise FitError("the information matrix is singular; use ridge > 0") from None
    return Fit(
        systems=counts.systems,
        theta=theta,
        log_nu=log_nu,
        cov=cov_full[:m, :m],
        cov_full=cov_full,
        loglik=value,
        n_votes=counts.total,
        ridge=ridge,
        iterations=iterations,
        converged=converged,
    )
