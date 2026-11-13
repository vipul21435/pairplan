"""Simulation with planted strengths: votes each scheduler needs, and how often it is wrong."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .engine import PlantedSource, run_schedule
from .metrics import kendall_tau
from .model import information, normal_cdf
from .schedulers import SCHEDULERS, ranking_risk, unit_information
from .stopping import StopRule, is_correct, target_pairs
from .votes import Counts, all_pairs


@dataclass(frozen=True)
class Summary:
    """One scheduler over many simulated runs.

    ``votes_*`` count new votes per run; a run that never met the rule counts as ``budget``.
    ``wrong_stopped`` counts runs that stopped by the rule on a wrong answer (wrong full order,
    or wrong top-k set); ``wrong_total`` also counts wrong answers of runs cut off by the budget.
    ``fixed_design`` is filled by :func:`plan` only.
    """

    scheduler: str
    runs: int
    stopped: int
    wrong_stopped: int
    wrong_total: int
    votes_mean: float
    votes_median: float
    votes_p90: float
    tau_mean: float
    fixed_design: int | None = None

    @property
    def wrong_rate(self) -> float:
        """Share of the runs that stopped by the rule whose answer was wrong."""
        return self.wrong_stopped / self.stopped if self.stopped else float("nan")


def true_order(theta: Sequence[float]) -> list[int]:
    return sorted(range(len(theta)), key=lambda k: (-theta[k], k))


def simulate(
    theta: Sequence[float],
    *,
    nu: float = 0.0,
    systems: Sequence[str] | None = None,
    schedulers: Sequence[str] = SCHEDULERS,
    runs: int = 200,
    batch: int = 5,
    budget: int = 5000,
    rule: StopRule | None = None,
    seed: int = 0,
    ridge: float = 0.01,
    floor: int = 1,
    floor_share: float = 0.1,
    start: Counts | None = None,
) -> list[Summary]:
    """Run every scheduler ``runs`` times on votes drawn from planted strengths.

    Each run asks for ``batch`` votes at a time, checks ``rule`` after each batch, and ends
    when the rule is met or ``budget`` new votes are spent. Runs are seeded by
    ``(seed, scheduler position, run number)``, so results repeat exactly.
    """
    theta = [float(x) for x in theta]
    names = list(systems) if systems is not None else [f"s{k + 1}" for k in range(len(theta))]
    if len(names) != len(theta):
        raise ValueError("systems and strengths differ in length")
    rule = rule or StopRule()
    truth = true_order(theta)
    out = []
    for s_index, scheduler in enumerate(schedulers):
        votes = np.empty(runs)
        taus = np.empty(runs)
        stopped = wrong_stopped = wrong_total = 0
        for r in range(runs):
            rng = np.random.default_rng([seed, s_index, r])
            run = run_schedule(
                PlantedSource(theta, nu), names, scheduler, budget=budget, rng=rng, batch=batch,
                rule=rule, ridge=ridge, floor=floor, floor_share=floor_share, start=start,
            )
            wrong = not is_correct(run.order, truth, rule.target)
            stopped += run.stopped
            wrong_stopped += run.stopped and wrong
            wrong_total += wrong
            votes[r] = run.votes if run.stopped else budget
            taus[r] = kendall_tau(run.order, truth)
        out.append(
            Summary(
                scheduler=scheduler,
                runs=runs,
                stopped=int(stopped),
                wrong_stopped=int(wrong_stopped),
                wrong_total=int(wrong_total),
                votes_mean=float(votes.mean()),
                votes_median=float(np.median(votes)),
                votes_p90=float(np.quantile(votes, 0.9)),
                tau_mean=float(taus.mean()),
            )
        )
    return out


def fixed_design_votes(
    theta: Sequence[float],
    scheduler: str,
    *,
    nu: float = 0.0,
    rule: StopRule | None = None,
    batch: int = 5,
    budget: int = 100000,
    ridge: float = 0.01,
    start: Counts | None = None,
) -> int | None:
    """Votes after which ``rule`` would be met if the estimates sat exactly on the truth.

    No votes are drawn. Each batch adds the Fisher information (at the true strengths) of
    the scheduler's allocation: equal shares of every pair for ``uniform`` and
    ``round-robin``, equal shares of the truly adjacent pairs for ``adjacent``, and the
    greedy risk-reducing pair, slot by slot, for ``uncertainty``. The rule is then checked on
    the true differences with the resulting standard errors. This ignores the noise in the
    estimates and the floor (except that ``uncertainty`` first gives every pair one vote), so
    it is a planning figure, not a guarantee. Returns ``None`` when the rule is not met within
    ``budget``.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {scheduler!r}")
    rule = rule or StopRule()
    theta_arr = np.asarray(theta, dtype=float)
    theta_arr = theta_arr - theta_arr.mean()
    m = theta_arr.shape[0]
    log_nu = math.log(nu) if nu > 0 else None
    size = m + 1 if log_nu is not None else m
    pairs = all_pairs(m)
    unit = unit_information(theta_arr, log_nu, pairs)
    weights = start.pair_totals.astype(float) if start is not None else np.zeros((m, m))
    info = information(theta_arr, log_nu, weights)[:size, :size]
    info[np.arange(size), np.arange(size)] += ridge
    info[:m, :m] += 1.0 / m
    order = true_order(list(theta_arr))
    deciding = target_pairs(order, rule.target)
    hi = np.array([p[0] for p in deciding])
    lo = np.array([p[1] for p in deciding])
    diff = theta_arr[hi] - theta_arr[lo]

    def met(look: int) -> bool:
        try:
            inverse = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return False
        var = inverse[hi, hi] + inverse[lo, lo] - 2.0 * inverse[hi, lo]
        needed = 1.0 - rule.alpha_pair(len(deciding), look)
        return all(
            normal_cdf(d / math.sqrt(v)) >= needed for d, v in zip(diff, var, strict=True)
        )

    if scheduler == "adjacent":
        neighbours = {tuple(sorted(p)) for p in zip(order[:-1], order[1:], strict=True)}
        step = unit[[k for k, p in enumerate(pairs) if p in neighbours]].mean(axis=0)
    else:
        step = unit.mean(axis=0)

    given = np.array([weights[i, j] for i, j in pairs])
    look = 0
    if start is not None:
        look = 1
        if met(look) and start.total >= rule.min_votes:
            return 0
    base_total = start.total if start is not None else 0
    spent = 0
    while spent < budget:
        size_now = min(batch, budget - spent)
        if scheduler == "uncertainty":
            for _ in range(size_now):
                if given.min() < 1:
                    k = int(np.argmin(given))  # every pair once first, as the floor of 1 does
                else:
                    base = ranking_risk(np.linalg.inv(info), theta_arr, deciding)
                    after = ranking_risk(np.linalg.inv(info[None] + unit), theta_arr, deciding)
                    k = int(np.argmax(base - after))
                info = info + unit[k]
                given[k] += 1
        else:
            info = info + size_now * step
        spent += size_now
        look += 1
        if met(look) and base_total + spent >= rule.min_votes:
            return spent
    return None


def plan(
    theta: Sequence[float],
    *,
    nu: float = 0.0,
    systems: Sequence[str] | None = None,
    schedulers: Sequence[str] = SCHEDULERS,
    runs: int = 100,
    batch: int = 5,
    budget: int = 5000,
    rule: StopRule | None = None,
    seed: int = 0,
    ridge: float = 0.01,
    floor: int = 1,
    floor_share: float = 0.1,
    start: Counts | None = None,
) -> list[Summary]:
    """Expected further votes each scheduler needs to meet ``rule``, taking ``theta`` as true.

    Combines :func:`simulate` (which draws votes, starting from the ``start`` counts when
    given) with :func:`fixed_design_votes` (which does not).
    """
    rule = rule or StopRule()
    summaries = simulate(
        theta, nu=nu, systems=systems, schedulers=schedulers, runs=runs, batch=batch,
        budget=budget, rule=rule, seed=seed, ridge=ridge, floor=floor, floor_share=floor_share,
        start=start,
    )
    out = []
    for summary in summaries:
        fixed = fixed_design_votes(
            theta, summary.scheduler, nu=nu, rule=rule, batch=batch, budget=budget, ridge=ridge,
            start=start,
        )
        out.append(replace(summary, fixed_design=fixed))
    return out
