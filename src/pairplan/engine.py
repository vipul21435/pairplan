"""Run a scheduler against a source of votes, with an optional stopping rule."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .model import Fit, outcome_probs
from .model import fit as fit_model
from .schedulers import ADAPTIVE, next_pairs
from .stopping import StopRule, StopStatus, check
from .votes import Counts, Vote, all_pairs


class PlantedSource:
    """Votes drawn from the model with known strengths; never runs out."""

    def __init__(self, theta: Sequence[float], nu: float = 0.0) -> None:
        if nu < 0:
            raise ValueError("nu must be >= 0")
        self.theta = np.asarray(theta, dtype=float)
        self.nu = float(nu)
        self.win, self.tie = outcome_probs(self.theta, math.log(nu) if nu > 0 else None)

    def capacity(self) -> np.ndarray | None:
        return None

    def draw(self, i: int, j: int, rng: np.random.Generator) -> str:
        u = rng.random()
        if u < self.win[i, j]:
            return "a"
        if u < self.win[i, j] + self.win[j, i]:
            return "b"
        return "tie"


class VotePool:
    """Real votes grouped by pair, for replay."""

    def __init__(self, votes: Sequence[Vote], systems: Sequence[str]) -> None:
        self.systems = tuple(systems)
        index = {name: k for k, name in enumerate(self.systems)}
        m = len(self.systems)
        self.by_pair: dict[tuple[int, int], list[tuple[int, str]]] = {p: [] for p in all_pairs(m)}
        flip = {"a": "b", "b": "a", "tie": "tie"}
        for number, vote in enumerate(votes):
            i, j = index[vote.a], index[vote.b]
            if i < j:
                self.by_pair[(i, j)].append((number, vote.outcome))
            else:
                self.by_pair[(j, i)].append((number, flip[vote.outcome]))
        self.total = len(votes)

    def sizes(self) -> np.ndarray:
        m = len(self.systems)
        out = np.zeros((m, m), dtype=np.int64)
        for (i, j), items in self.by_pair.items():
            out[i, j] = out[j, i] = len(items)
        return out


class PoolSource:
    """Draws from a :class:`VotePool` without replacement.

    Each pair's votes are shuffled once; a draw hands out the next unused vote of that pair.
    ``used`` lists the position (in the original vote list) of every vote handed out.
    """

    def __init__(self, pool: VotePool, rng: np.random.Generator) -> None:
        self.pool = pool
        self._queues: dict[tuple[int, int], list[tuple[int, str]]] = {}
        for pair, items in pool.by_pair.items():
            order = rng.permutation(len(items))
            self._queues[pair] = [items[k] for k in order]
        self._remaining = pool.sizes()
        self.used: list[int] = []

    def capacity(self) -> np.ndarray | None:
        return self._remaining

    def draw(self, i: int, j: int, rng: np.random.Generator) -> str:
        queue = self._queues[(i, j)]
        if not queue:
            raise RuntimeError(f"no votes left for pair {(i, j)}")
        number, outcome = queue.pop()
        self._remaining[i, j] -= 1
        self._remaining[j, i] -= 1
        self.used.append(number)
        return outcome


@dataclass
class RunResult:
    """What one run of a scheduler did."""

    scheduler: str
    votes: int
    stopped: bool
    exhausted: bool
    looks: int
    order: list[int]
    counts: Counts
    status: StopStatus | None = None
    trace: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    first_exhausted_at: int | None = None


def run_schedule(
    source: PlantedSource | PoolSource,
    systems: Sequence[str],
    scheduler: str,
    *,
    budget: int,
    rng: np.random.Generator,
    batch: int = 1,
    rule: StopRule | None = None,
    ridge: float = 0.01,
    floor: int = 1,
    floor_share: float = 0.1,
    record_every: int = 0,
    start: Counts | None = None,
) -> RunResult:
    """Collect up to ``budget`` new votes, ``batch`` at a time, as ``scheduler`` asks for them.

    After each batch the model is refitted; with a ``rule`` the run stops at the first look
    where the rule is met (a run that starts from ``start`` votes is checked once before
    asking for anything). ``record_every`` > 0 records the fitted order each time that many
    new votes have been collected. ``votes`` in the result counts new votes only.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    counts = start.copy() if start is not None else Counts.empty(systems)
    target = rule.target if rule is not None else "full"
    need_fit = scheduler in ADAPTIVE or rule is not None
    current: Fit | None = None
    looks = 0
    status: StopStatus | None = None
    stopped = False
    exhausted = False
    first_exhausted_at: int | None = None
    trace: list[tuple[int, tuple[int, ...]]] = []
    new_votes = 0
    next_record = record_every if record_every > 0 else None

    if start is not None and rule is not None:
        current = fit_model(counts, ridge=ridge)
        looks = 1
        status = check(current, rule, looks)
        stopped = status.settled

    while not stopped and new_votes < budget:
        if need_fit and current is None:
            current = fit_model(counts, ridge=ridge)
        capacity = source.capacity()
        pairs = next_pairs(
            scheduler,
            counts,
            min(batch, budget - new_votes),
            rng=rng,
            fit=current,
            ridge=ridge,
            floor=floor,
            floor_share=floor_share,
            capacity=capacity,
            target=target,
        )
        if not pairs:
            exhausted = True
            break
        for i, j in pairs:
            counts.add(i, j, source.draw(i, j, rng))
        new_votes += len(pairs)
        if first_exhausted_at is None and capacity is not None:
            left = source.capacity()
            assert left is not None
            if (left[np.triu_indices(counts.m, 1)] == 0).any():
                first_exhausted_at = new_votes
        record_now = next_record is not None and new_votes >= next_record
        if need_fit or record_now:
            current = fit_model(counts, ridge=ridge, start=current)
        if record_now:
            assert current is not None and next_record is not None
            trace.append((new_votes, tuple(current.order)))
            while next_record <= new_votes:
                next_record += record_every
        if rule is not None:
            assert current is not None
            looks += 1
            status = check(current, rule, looks)
            stopped = status.settled

    if current is None or current.n_votes != counts.total:
        current = fit_model(counts, ridge=ridge, start=current)
    return RunResult(
        scheduler=scheduler,
        votes=new_votes,
        stopped=stopped,
        exhausted=exhausted,
        looks=looks,
        order=current.order,
        counts=counts,
        status=status,
        trace=trace,
        first_exhausted_at=first_exhausted_at,
    )
