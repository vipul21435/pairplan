"""Reading pairwise votes and tabulating them into count matrices."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUTCOMES = ("a", "b", "tie")

_A_NAMES = ("a", "model_a", "system_a", "left")
_B_NAMES = ("b", "model_b", "system_b", "right")
_WINNER_NAMES = ("winner", "outcome", "result")
_JUDGE_NAMES = ("judge", "rater", "annotator")
_PROMPT_NAMES = ("prompt", "question_id", "prompt_id", "item")


class VoteError(ValueError):
    """A vote file or record could not be understood."""


@dataclass(frozen=True)
class Vote:
    """One comparison: systems ``a`` and ``b`` were shown and ``outcome`` is "a", "b" or "tie"."""

    a: str
    b: str
    outcome: str
    judge: str | None = None
    prompt: str | None = None


def _pick(record: dict, wanted: str | None, fallbacks: Sequence[str], required: bool) -> str | None:
    names = (wanted,) if wanted else tuple(fallbacks)
    for name in names:
        if name in record and record[name] not in (None, ""):
            return str(record[name])
    if required:
        raise VoteError(f"missing column {' or '.join(names)!r}")
    return None


def parse_outcome(value: str, a: str, b: str) -> str:
    """Map a winner field to "a", "b" or "tie".

    Accepted: the literal side ("a", "b", "model_a", "model_b", "left", "right"), the name of
    one of the two systems, or anything starting with "tie" or "draw" (so "tie (bothbad)" is a
    tie). A system that is literally named "a" or "b" is resolved by name first.
    """
    raw = value.strip()
    if raw == a:
        return "a"
    if raw == b:
        return "b"
    low = raw.lower()
    if low in _A_NAMES:
        return "a"
    if low in _B_NAMES:
        return "b"
    if low.startswith(("tie", "draw")):
        return "tie"
    raise VoteError(f"winner {value!r} is neither side, neither system name, nor a tie")


def vote_from_record(
    record: dict,
    *,
    a_col: str | None = None,
    b_col: str | None = None,
    winner_col: str | None = None,
    judge_col: str | None = None,
    prompt_col: str | None = None,
) -> Vote:
    """Build a :class:`Vote` from one parsed row."""
    a = _pick(record, a_col, _A_NAMES, True)
    b = _pick(record, b_col, _B_NAMES, True)
    winner = _pick(record, winner_col, _WINNER_NAMES, True)
    assert a is not None and b is not None and winner is not None
    if a == b:
        raise VoteError(f"a vote compares {a!r} with itself")
    return Vote(
        a=a,
        b=b,
        outcome=parse_outcome(winner, a, b),
        judge=_pick(record, judge_col, _JUDGE_NAMES, False),
        prompt=_pick(record, prompt_col, _PROMPT_NAMES, False),
    )


def read_votes(path: str | Path, **columns: str | None) -> list[Vote]:
    """Read votes from a ``.jsonl`` or ``.csv`` file (chosen by suffix; ``.tsv`` is tab-separated).

    Column names default to ``a``/``b``/``winner`` with common aliases (``model_a``,
    ``model_b``, ...); pass ``a_col=...`` etc. to override. Blank lines are skipped. Errors
    name the file and the line.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    votes: list[Vote] = []
    with path.open(newline="", encoding="utf-8") as handle:
        if suffix in (".csv", ".tsv"):
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            rows: Iterable[tuple[int, dict]] = ((n, row) for n, row in enumerate(reader, start=2))
            for n, row in rows:
                try:
                    votes.append(vote_from_record(row, **columns))
                except VoteError as exc:
                    raise VoteError(f"{path}:{n}: {exc}") from None
        else:
            for n, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VoteError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
                if not isinstance(record, dict):
                    raise VoteError(f"{path}:{n}: expected a JSON object per line")
                try:
                    votes.append(vote_from_record(record, **columns))
                except VoteError as exc:
                    raise VoteError(f"{path}:{n}: {exc}") from None
    return votes


def systems_of(votes: Iterable[Vote]) -> list[str]:
    """Sorted names of every system that appears in the votes."""
    names: set[str] = set()
    for vote in votes:
        names.add(vote.a)
        names.add(vote.b)
    return sorted(names)


def all_pairs(m: int) -> list[tuple[int, int]]:
    """Every unordered pair ``(i, j)`` with ``i < j`` in a fixed lexicographic order."""
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


@dataclass
class Counts:
    """Votes aggregated per pair.

    ``wins[i, j]`` is the number of votes in which system ``i`` beat system ``j``;
    ``ties[i, j] == ties[j, i]`` is the number of ties between them. Both have a zero diagonal.
    """

    systems: tuple[str, ...]
    wins: np.ndarray
    ties: np.ndarray

    @classmethod
    def empty(cls, systems: Sequence[str]) -> Counts:
        names = tuple(systems)
        if len(set(names)) != len(names):
            raise VoteError("system names must be unique")
        if len(names) < 2:
            raise VoteError("need at least two systems")
        m = len(names)
        return cls(names, np.zeros((m, m), dtype=np.int64), np.zeros((m, m), dtype=np.int64))

    @classmethod
    def from_votes(cls, votes: Iterable[Vote], systems: Sequence[str] | None = None) -> Counts:
        votes = list(votes)
        counts = cls.empty(systems if systems is not None else systems_of(votes))
        index = {name: k for k, name in enumerate(counts.systems)}
        for vote in votes:
            try:
                i, j = index[vote.a], index[vote.b]
            except KeyError as exc:
                raise VoteError(f"vote names system {exc.args[0]!r}, not in the list") from None
            counts.add(i, j, vote.outcome)
        return counts

    @property
    def m(self) -> int:
        return len(self.systems)

    @property
    def pair_totals(self) -> np.ndarray:
        """Symmetric matrix of votes per pair (wins either way plus ties)."""
        return self.wins + self.wins.T + self.ties

    @property
    def total(self) -> int:
        return int(self.wins.sum() + self.ties.sum() // 2)

    def add(self, i: int, j: int, outcome: str, k: int = 1) -> None:
        """Record ``k`` votes on pair ``(i, j)``; ``outcome`` "a" means ``i`` won."""
        if i == j:
            raise VoteError("a vote needs two different systems")
        if outcome == "a":
            self.wins[i, j] += k
        elif outcome == "b":
            self.wins[j, i] += k
        elif outcome == "tie":
            self.ties[i, j] += k
            self.ties[j, i] += k
        else:
            raise VoteError(f"unknown outcome {outcome!r}")

    def copy(self) -> Counts:
        return Counts(self.systems, self.wins.copy(), self.ties.copy())
