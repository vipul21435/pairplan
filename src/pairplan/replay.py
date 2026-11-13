"""Replay schedulers on real votes: each one only sees the votes it asks for."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .engine import PoolSource, VotePool, run_schedule
from .metrics import kendall_tau
from .model import Fit
from .model import fit as fit_model
from .schedulers import SCHEDULERS
from .votes import Counts, Vote, systems_of


@dataclass(frozen=True)
class ReplayCurve:
    """One scheduler over many replay runs.

    ``tau_mean[c]`` and ``exact_share[c]`` are the mean Kendall tau to the full-data ranking
    and the share of runs whose ranking equals it after ``checkpoints[c]`` votes.
    ``settle[r]`` is, for run ``r``, the first checkpoint from which the ranking equals the
    full-data ranking at every later checkpoint (``None`` if it differs at the last one).
    ``first_exhausted[r]`` is the number of votes at which some pair first ran out.
    """

    scheduler: str
    checkpoints: tuple[int, ...]
    tau_mean: tuple[float, ...]
    tau_p10: tuple[float, ...]
    exact_share: tuple[float, ...]
    settle: tuple[int | None, ...]
    first_exhausted: tuple[int | None, ...]

    def votes_to_tau(self, level: float) -> int | None:
        """First checkpoint from which the mean tau stays at or above ``level``."""
        answer = None
        for votes, tau in zip(self.checkpoints, self.tau_mean, strict=True):
            if tau >= level:
                if answer is None:
                    answer = votes
            else:
                answer = None
        return answer

    def votes_to_exact(self, share: float) -> int | None:
        """First checkpoint from which at least ``share`` of runs have the full-data ranking."""
        answer = None
        for votes, value in zip(self.checkpoints, self.exact_share, strict=True):
            if value >= share:
                if answer is None:
                    answer = votes
            else:
                answer = None
        return answer


@dataclass(frozen=True)
class ReplayResult:
    reference: Fit
    total_votes: int
    pool_min: int
    pool_max: int
    runs: int
    batch: int
    budget: int
    every: int
    curves: tuple[ReplayCurve, ...]


def replay(
    votes: Sequence[Vote],
    systems: Sequence[str] | None = None,
    *,
    schedulers: Sequence[str] = SCHEDULERS,
    runs: int = 50,
    batch: int = 10,
    budget: int | None = None,
    every: int | None = None,
    seed: int = 0,
    ridge: float = 0.01,
    floor: int = 1,
    floor_share: float = 0.1,
) -> ReplayResult:
    """Replay each scheduler ``runs`` times against the pool of real votes.

    A scheduler asks for a pair and receives one real vote on that pair, drawn without
    replacement; no vote is handed out twice within a run. A pair whose votes are used up is
    no longer offered to the scheduler, which then picks among the pairs that still have
    votes. ``budget`` (default: all votes) caps the votes per run and ``every`` (rounded up to
    a multiple of ``batch``) sets how often the ranking is compared with the ranking fitted
    on all votes. With the default budget every run ends having used every vote, so all
    schedulers meet at the full-data ranking; the curve before that point is what differs.
    """
    votes = list(votes)
    names = list(systems) if systems is not None else systems_of(votes)
    pool = VotePool(votes, names)
    reference = fit_model(Counts.from_votes(votes, names), ridge=ridge)
    ref_order = reference.order
    budget = pool.total if budget is None else min(budget, pool.total)
    if budget < 1:
        raise ValueError("no votes to replay")
    if every is None:
        every = max(budget // 20, batch)
    every = max(batch, -(-every // batch) * batch)
    sizes = pool.sizes()[np.triu_indices(len(names), 1)]

    curves = []
    for s_index, scheduler in enumerate(schedulers):
        traces = []
        first_exhausted = []
        for r in range(runs):
            rng = np.random.default_rng([seed, s_index, r])
            source = PoolSource(pool, rng)
            run = run_schedule(
                source, names, scheduler, budget=budget, rng=rng, batch=batch, ridge=ridge,
                floor=floor, floor_share=floor_share, record_every=every,
            )
            trace = run.trace
            if not trace or trace[-1][0] != run.votes:
                trace = [*trace, (run.votes, tuple(run.order))]
            traces.append(trace)
            first_exhausted.append(run.first_exhausted_at)
        checkpoints = tuple(n for n, _ in traces[0])
        taus = np.array([[kendall_tau(order, ref_order) for _, order in t] for t in traces])
        exact = np.array([[list(order) == ref_order for _, order in t] for t in traces])
        settle: list[int | None] = []
        for row in exact:
            wrong = np.flatnonzero(~row)
            if wrong.size == 0:
                settle.append(checkpoints[0])
            elif wrong[-1] == len(row) - 1:
                settle.append(None)
            else:
                settle.append(checkpoints[wrong[-1] + 1])
        curves.append(
            ReplayCurve(
                scheduler=scheduler,
                checkpoints=checkpoints,
                tau_mean=tuple(float(x) for x in taus.mean(axis=0)),
                tau_p10=tuple(float(x) for x in np.quantile(taus, 0.1, axis=0)),
                exact_share=tuple(float(x) for x in exact.mean(axis=0)),
                settle=tuple(settle),
                first_exhausted=tuple(first_exhausted),
            )
        )
    return ReplayResult(
        reference=reference,
        total_votes=pool.total,
        pool_min=int(sizes.min()),
        pool_max=int(sizes.max()),
        runs=runs,
        batch=batch,
        budget=budget,
        every=every,
        curves=tuple(curves),
    )
