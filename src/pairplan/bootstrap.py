"""Bootstrap intervals for the strengths, resampling votes (or whole judges or prompts)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Fit
from .model import fit as fit_model
from .votes import Counts, Vote, VoteError


@dataclass(frozen=True)
class Bootstrap:
    """Bootstrap replicates of the centred strengths, one row per replicate."""

    systems: tuple[str, ...]
    thetas: np.ndarray
    level: float
    unit: str

    @property
    def intervals(self) -> np.ndarray:
        """Percentile interval of each strength: an ``(m, 2)`` array."""
        tail = (1.0 - self.level) / 2.0
        return np.quantile(self.thetas, [tail, 1.0 - tail], axis=0).T

    @property
    def rank_intervals(self) -> np.ndarray:
        """Percentile interval of each system's rank (1 = strongest): an ``(m, 2)`` array."""
        ranks = (-self.thetas).argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1
        tail = (1.0 - self.level) / 2.0
        lo = np.floor(np.quantile(ranks, tail, axis=0)).astype(int)
        hi = np.ceil(np.quantile(ranks, 1.0 - tail, axis=0)).astype(int)
        return np.stack([lo, hi], axis=1)

    def order_share(self, i: int, j: int) -> float:
        """Share of replicates in which system ``i`` is stronger than system ``j``."""
        return float(np.mean(self.thetas[:, i] > self.thetas[:, j]))


def _cluster_arrays(
    votes: Sequence[Vote], systems: Sequence[str], cluster: str
) -> tuple[np.ndarray, np.ndarray]:
    keys: dict[str, int] = {}
    index = {name: k for k, name in enumerate(systems)}
    rows = []
    for vote in votes:
        key = getattr(vote, cluster)
        if key is None:
            raise VoteError(f"a vote has no {cluster}; cannot resample by {cluster}")
        rows.append((keys.setdefault(key, len(keys)), index[vote.a], index[vote.b], vote.outcome))
    m = len(systems)
    wins = np.zeros((len(keys), m, m))
    ties = np.zeros((len(keys), m, m))
    for c, i, j, outcome in rows:
        if outcome == "a":
            wins[c, i, j] += 1
        elif outcome == "b":
            wins[c, j, i] += 1
        else:
            ties[c, i, j] += 1
            ties[c, j, i] += 1
    return wins, ties


def bootstrap(
    counts: Counts,
    *,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 0,
    ridge: float = 0.01,
    votes: Sequence[Vote] | None = None,
    cluster: str | None = None,
    start: Fit | None = None,
) -> Bootstrap:
    """Refit the model on ``n_boot`` resamples of the votes.

    By default single votes are resampled with replacement (a multinomial draw over the
    pair-and-outcome cells). With ``cluster="judge"`` or ``cluster="prompt"`` whole judges or
    prompts are resampled instead, which needs the original ``votes``. ``ridge`` must be
    positive enough for every resample to have a finite fit; the default is.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if cluster not in (None, "judge", "prompt"):
        raise ValueError("cluster must be 'judge' or 'prompt'")
    rng = np.random.default_rng(seed)
    m = counts.m
    thetas = np.empty((n_boot, m))
    if cluster is None:
        iu = np.triu_indices(m, 1)
        cells = np.concatenate([counts.wins[iu], counts.wins.T[iu], counts.ties[iu]]).astype(float)
        total = int(cells.sum())
        if total == 0:
            raise VoteError("no votes to resample")
        prob = cells / total
        n_pairs = len(iu[0])
        for b in range(n_boot):
            draw = rng.multinomial(total, prob)
            sample = Counts.empty(counts.systems)
            sample.wins[iu] = draw[:n_pairs]
            sample.wins.T[iu] = draw[n_pairs : 2 * n_pairs]
            sample.ties[iu] = draw[2 * n_pairs :]
            sample.ties.T[iu] = draw[2 * n_pairs :]
            thetas[b] = fit_model(sample, ridge=ridge, start=start).theta
    else:
        if votes is None:
            raise ValueError("resampling by judge or prompt needs the votes")
        wins, ties = _cluster_arrays(votes, counts.systems, cluster)
        n_clusters = wins.shape[0]
        for b in range(n_boot):
            weight = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
            sample = Counts(
                counts.systems,
                np.tensordot(weight, wins, axes=1).astype(np.int64),
                np.tensordot(weight, ties, axes=1).astype(np.int64),
            )
            thetas[b] = fit_model(sample, ridge=ridge, start=start).theta
    return Bootstrap(counts.systems, thetas, level, cluster or "vote")
