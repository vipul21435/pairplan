"""Stopping rules: is the ranking (or its top-k set) settled at a chosen confidence?

A pair ``(upper, lower)`` of the fitted ranking is *settled* when the normal-approximation
probability that ``upper`` is truly the stronger one is at least ``1 - alpha_pair``. Which
pairs are checked depends on the target:

- ``full``: every adjacent pair of the ranking (``m - 1`` pairs). If all adjacent pairs are
  ordered correctly, the whole order is.
- ``top-k``: every pair with one system inside the current top ``k`` and one outside
  (``k * (m - k)`` pairs). The order inside the top set does not matter.

``alpha_pair`` depends on the correction:

- ``none``: ``alpha_pair = 1 - confidence``. Each pair is tested on its own, at every look.
  The stated confidence is then per pair and per look, not for the ranking: checking several
  pairs, and checking again after every batch until the rule fires, both raise the chance of
  stopping on a wrong ranking above ``1 - confidence``.
- ``pairs``: ``alpha_pair = (1 - confidence) / n_pairs`` (a union bound over the pairs
  checked at one look).
- ``sequential``: ``alpha_pair = (1 - confidence) / (n_pairs * t * (t + 1))`` at look
  ``t = 1, 2, ...``. The terms ``1 / (t * (t + 1))`` sum to one over all looks, so this is a
  union bound over pairs and looks. It holds to the extent the normal approximation does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Fit

CORRECTIONS = ("none", "pairs", "sequential")


def parse_target(target: str, m: int) -> int | None:
    """Return ``None`` for the full order or ``k`` for a ``top-k`` target."""
    if target == "full":
        return None
    if target.startswith("top-"):
        try:
            k = int(target[4:])
        except ValueError:
            k = 0
        if 1 <= k < m:
            return k
    raise ValueError(f"target must be 'full' or 'top-k' with 1 <= k < {m}, got {target!r}")


def target_pairs(order: list[int], target: str) -> list[tuple[int, int]]:
    """The ``(upper, lower)`` index pairs whose order decides the target."""
    k = parse_target(target, len(order))
    if k is None:
        return list(zip(order[:-1], order[1:], strict=True))
    return [(hi, lo) for hi in order[:k] for lo in order[k:]]


def is_correct(order: list[int], truth: list[int], target: str) -> bool:
    """Whether ``order`` matches ``truth`` on the target (full order, or the top-k set)."""
    k = parse_target(target, len(order))
    if k is None:
        return list(order) == list(truth)
    return set(order[:k]) == set(truth[:k])


@dataclass(frozen=True)
class StopRule:
    """When to stop collecting votes."""

    confidence: float = 0.95
    target: str = "full"
    correction: str = "none"
    min_votes: int = 0

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0.5 and 1")
        if self.correction not in CORRECTIONS:
            raise ValueError(f"correction must be one of {', '.join(CORRECTIONS)}")
        if self.target != "full" and not self.target.startswith("top-"):
            raise ValueError("target must be 'full' or 'top-k'")

    def alpha_pair(self, n_pairs: int, look: int = 1) -> float:
        alpha = 1.0 - self.confidence
        if self.correction == "none":
            return alpha
        alpha /= max(n_pairs, 1)
        if self.correction == "sequential":
            alpha /= look * (look + 1)
        return alpha


@dataclass(frozen=True)
class PairStatus:
    upper: str
    lower: str
    prob_correct: float
    settled: bool


@dataclass(frozen=True)
class StopStatus:
    """Outcome of one check. ``needed`` is the per-pair probability required at this look."""

    settled: bool
    needed: float
    look: int
    pairs: tuple[PairStatus, ...]

    @property
    def unsettled(self) -> list[PairStatus]:
        return [pair for pair in self.pairs if not pair.settled]


def check(result: Fit, rule: StopRule, look: int = 1) -> StopStatus:
    """Check ``rule`` against a fit. ``look`` counts the checks made so far, starting at 1."""
    if look < 1:
        raise ValueError("look starts at 1")
    pairs = target_pairs(result.order, rule.target)
    needed = 1.0 - rule.alpha_pair(len(pairs), look)
    statuses = []
    for hi, lo in pairs:
        prob = result.prob_order(hi, lo)
        statuses.append(PairStatus(result.systems[hi], result.systems[lo], prob, prob >= needed))
    settled = all(s.settled for s in statuses) and result.n_votes >= rule.min_votes
    return StopStatus(settled, needed, look, tuple(statuses))
