"""Schedulers: given the votes so far, which pairs to ask about next.

Every scheduler is reached through :func:`next_pairs`, which returns a batch of ``(i, j)``
index pairs (``i < j``; a pair can appear more than once in a batch).

- ``uniform``: each slot is a pair drawn uniformly at random.
- ``round-robin``: each slot goes to the pair with the fewest votes so far (counting the
  slots already filled in this batch); equal counts are broken by pair order.
- ``adjacent``: fit the model, rank the systems, and give each slot to a pair of neighbours
  in that ranking, the one with the fewest votes first. If no neighbouring pair can be asked
  any more, the pairs closest in rank that still can are used.
- ``uncertainty``: fit the model and pick the pair whose next vote is expected to reduce the
  risk of a wrong ranking the most. With ``d_k`` the fitted difference of the k-th pair that
  decides the target (adjacent pairs for the full order, see ``stopping.target_pairs``) and
  ``s_k`` its standard error, the risk is ``R = sum_k Phi(-d_k / s_k)``: the sum of the
  normal-approximation probabilities that each deciding pair is misordered, an upper bound on
  the probability that the target is wrong. One more vote on pair ``(i, j)`` adds its Fisher
  information ``J_ij`` (at the fitted strengths) to the information matrix, which shrinks the
  ``s_k`` to ``s_k(i, j)``. The pair chosen maximises
  ``R - sum_k Phi(-d_k / s_k(i, j))``; equal gains go to the pair with fewer votes. In a
  batch the choice is greedy: the information of each chosen pair is added before the next
  slot is filled, so a batch spreads over several pairs.

The two adaptive schedulers honour a floor so that no pair is starved: while any pair has
fewer votes than ``max(floor, floor_share * average votes per pair)``, the pair with the
fewest votes is served first.
"""

from __future__ import annotations

import math

import numpy as np

from .model import Fit, information, normal_sf, outcome_probs
from .model import fit as fit_model
from .stopping import target_pairs
from .votes import Counts, all_pairs

Pair = tuple[int, int]

SCHEDULERS = ("uniform", "round-robin", "adjacent", "uncertainty")
ADAPTIVE = ("adjacent", "uncertainty")


def unit_information(theta: np.ndarray, log_nu: float | None, pairs: list[Pair]) -> np.ndarray:
    """Fisher information of one vote on each pair: an array ``(n_pairs, size, size)``.

    ``size`` is ``m + 1`` with a tie parameter and ``m`` without.
    """
    m = theta.shape[0]
    size = m + 1 if log_nu is not None else m
    win, tie = outcome_probs(theta, log_nu)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    pi, pj, pt = win[ii, jj], win[jj, ii], tie[ii, jj]
    si, sj = pi + 0.5 * pt, pj + 0.5 * pt
    unit = np.zeros((len(pairs), size, size))
    rows = np.arange(len(pairs))
    unit[rows, ii, ii] = pi + 0.25 * pt - si * si
    unit[rows, jj, jj] = pj + 0.25 * pt - sj * sj
    unit[rows, ii, jj] = unit[rows, jj, ii] = 0.25 * pt - si * sj
    if log_nu is not None:
        unit[rows, ii, m] = unit[rows, m, ii] = 0.5 * pt - si * pt
        unit[rows, jj, m] = unit[rows, m, jj] = 0.5 * pt - sj * pt
        unit[rows, m, m] = pt - pt * pt
    return unit


def ranking_risk(inverse: np.ndarray, theta: np.ndarray, deciding: list[Pair]) -> np.ndarray:
    """``sum_k Phi(-d_k / s_k)`` for a stack of inverse information matrices."""
    hi = np.array([p[0] for p in deciding])
    lo = np.array([p[1] for p in deciding])
    diff = theta[hi] - theta[lo]
    var = inverse[..., hi, hi] + inverse[..., lo, lo] - 2.0 * inverse[..., hi, lo]
    z = diff / np.sqrt(np.clip(var, 1e-300, None))
    return normal_sf(z).sum(axis=-1)


class _Uncertainty:
    """State of the uncertainty rule while one batch is being filled."""

    def __init__(self, result: Fit, counts: Counts, pairs: list[Pair], target: str) -> None:
        m = counts.m
        self.theta = result.theta
        self.size = m + 1 if result.log_nu is not None else m
        self.deciding = target_pairs(result.order, target)
        weights = counts.pair_totals.astype(float)
        info = information(result.theta, result.log_nu, weights)[: self.size, : self.size]
        info[np.arange(self.size), np.arange(self.size)] += result.ridge
        info[:m, :m] += 1.0 / m
        self.info = info
        self.unit = unit_information(result.theta, result.log_nu, pairs)

    def gains(self) -> np.ndarray:
        base = ranking_risk(np.linalg.inv(self.info), self.theta, self.deciding)
        after = ranking_risk(np.linalg.inv(self.info[None] + self.unit), self.theta, self.deciding)
        return base - after

    def take(self, k: int) -> None:
        self.info = self.info + self.unit[k]


def _first_min(values: np.ndarray, mask: np.ndarray) -> int:
    """Index of the smallest value among ``mask``; the first one on ties."""
    idx = np.flatnonzero(mask)
    return int(idx[np.argmin(values[idx])])


def next_pairs(
    scheduler: str,
    counts: Counts,
    batch: int = 1,
    *,
    rng: np.random.Generator | None = None,
    fit: Fit | None = None,
    ridge: float = 0.01,
    floor: int = 1,
    floor_share: float = 0.1,
    capacity: np.ndarray | None = None,
    target: str = "full",
) -> list[Pair]:
    """The next ``batch`` pairs to ask about.

    ``rng`` is used by ``uniform`` only (the other schedulers are deterministic functions of
    the votes). ``fit`` may pass a fit of ``counts`` to avoid refitting. ``capacity`` is an
    optional symmetric matrix with the number of votes still obtainable per pair; pairs at
    zero are never returned, and fewer than ``batch`` pairs come back when the total capacity
    is smaller.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {scheduler!r}; choose from {', '.join(SCHEDULERS)}")
    if batch < 0:
        raise ValueError("batch must be >= 0")
    if floor < 0 or floor_share < 0:
        raise ValueError("floor and floor_share must be >= 0")
    m = counts.m
    pairs = all_pairs(m)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    seen = counts.pair_totals[ii, jj].astype(float)
    cap = np.full(len(pairs), math.inf) if capacity is None else capacity[ii, jj].astype(float)

    adaptive = scheduler in ADAPTIVE
    result = fit
    if adaptive and result is None:
        result = fit_model(counts, ridge=ridge)
    if scheduler == "uniform" and rng is None:
        rng = np.random.default_rng()
    position = None
    state = None
    if scheduler == "adjacent":
        assert result is not None
        position = np.empty(m, dtype=np.int64)
        position[result.order] = np.arange(m)
        distance = np.abs(position[ii] - position[jj])
        upper = np.minimum(position[ii], position[jj])

    chosen: list[Pair] = []
    for _ in range(batch):
        is_open = cap > 0
        if not is_open.any():
            break
        k = None
        if adaptive:
            level = max(float(floor), floor_share * float(seen.sum()) / len(pairs))
            starved = is_open & (seen < level)
            if starved.any():
                k = _first_min(seen, starved)
        if k is None:
            if scheduler == "uniform":
                assert rng is not None
                k = int(rng.choice(np.flatnonzero(is_open)))
            elif scheduler == "round-robin":
                k = _first_min(seen, is_open)
            elif scheduler == "adjacent":
                idx = np.flatnonzero(is_open)
                best = np.lexsort((upper[idx], seen[idx], distance[idx]))[0]
                k = int(idx[best])
            else:
                assert result is not None
                if state is None:
                    state = _Uncertainty(result, counts, pairs, target)
                gain = state.gains()
                top = float(gain[is_open].max())
                close = is_open & (gain >= top - 1e-9 * abs(top) - 1e-15)
                k = _first_min(seen, close)
        if state is not None:
            state.take(k)
        elif scheduler == "uncertainty" and result is not None:
            state = _Uncertainty(result, counts, pairs, target)
            state.take(k)
        chosen.append(pairs[k])
        seen[k] += 1
        cap[k] -= 1
    return chosen
