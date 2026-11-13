"""Markdown and JSON rendering of results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .model import Fit


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=False)


def fit_payload(result: Fit) -> dict:
    """A JSON-ready summary of a fit."""
    se = result.se
    return {
        "n_votes": result.n_votes,
        "model": "davidson" if result.log_nu is not None else "bradley-terry",
        "nu": result.nu,
        "ridge": result.ridge,
        "converged": result.converged,
        "log_likelihood": result.loglik,
        "ranking": [
            {
                "rank": rank,
                "system": result.systems[k],
                "theta": float(result.theta[k]),
                "se": float(se[k]),
            }
            for rank, k in enumerate(result.order, start=1)
        ],
        "adjacent": [
            {
                "upper": pair.upper,
                "lower": pair.lower,
                "difference": pair.difference,
                "se": pair.se,
                "prob_correct": pair.prob_correct,
            }
            for pair in result.adjacent()
        ],
    }


def fit_markdown(payload: dict) -> str:
    """Markdown for the payload of :func:`fit_payload` (plus optional bootstrap fields)."""
    has_boot = bool(payload["ranking"]) and "interval" in payload["ranking"][0]
    headers = ["rank", "system", "theta", "se"]
    if has_boot:
        level = payload["bootstrap"]["level"]
        headers += [f"{level:.0%} bootstrap interval", "rank interval"]
    rows = []
    for row in payload["ranking"]:
        cells = [row["rank"], row["system"], f"{row['theta']:+.3f}", f"{row['se']:.3f}"]
        if has_boot:
            lo, hi = row["interval"]
            rlo, rhi = row["rank_interval"]
            cells += [f"{lo:+.3f} to {hi:+.3f}", f"{rlo} to {rhi}"]
        rows.append(cells)
    adj_headers = ["upper", "lower", "difference", "se", "P(order correct)"]
    if has_boot:
        adj_headers.append("bootstrap share")
    adj_rows = []
    for pair in payload["adjacent"]:
        cells = [
            pair["upper"],
            pair["lower"],
            f"{pair['difference']:.3f}",
            f"{pair['se']:.3f}",
            f"{pair['prob_correct']:.3f}",
        ]
        if has_boot:
            cells.append(f"{pair['bootstrap_share']:.3f}")
        adj_rows.append(cells)
    head = (
        f"{payload['n_votes']} votes, {len(payload['ranking'])} systems, model {payload['model']}"
        f" (tie parameter nu = {payload['nu']:.3f}), ridge {payload['ridge']:g}"
    )
    if not payload["converged"]:
        head += "\n\nWarning: the fit did not converge."
    parts = [head, "", table(headers, rows), "", "Adjacent pairs of the ranking:", ""]
    parts.append(table(adj_headers, adj_rows))
    return "\n".join(parts)


def _num(value: float) -> str:
    return f"{value:.0f}"


def summaries_payload(summaries: Sequence[Any], settings: dict) -> dict:
    """JSON-ready form of the output of ``simulate`` or ``plan``."""
    rows = []
    for s in summaries:
        row = {
            "scheduler": s.scheduler,
            "runs": s.runs,
            "stopped": s.stopped,
            "votes_mean": s.votes_mean,
            "votes_median": s.votes_median,
            "votes_p90": s.votes_p90,
            "wrong_when_stopped": s.wrong_stopped,
            "wrong_rate_when_stopped": None if s.stopped == 0 else s.wrong_stopped / s.stopped,
            "wrong_total": s.wrong_total,
            "tau_mean": s.tau_mean,
        }
        if "fixed_design" in settings.get("columns", ()):
            row["fixed_design_votes"] = s.fixed_design
        rows.append(row)
    return {"settings": {k: v for k, v in settings.items() if k != "columns"}, "schedulers": rows}


def summaries_markdown(payload: dict) -> str:
    settings = payload["settings"]
    with_fixed = bool(payload["schedulers"]) and "fixed_design_votes" in payload["schedulers"][0]
    headers = ["scheduler", "stopped", "votes mean", "median", "p90"]
    if with_fixed:
        headers.append("fixed design")
    headers += ["wrong when stopped", "wrong overall", "mean tau"]
    rows = []
    for row in payload["schedulers"]:
        cells = [
            row["scheduler"],
            f"{row['stopped']}/{row['runs']}",
            _num(row["votes_mean"]),
            _num(row["votes_median"]),
            _num(row["votes_p90"]),
        ]
        if with_fixed:
            fixed = row["fixed_design_votes"]
            cells.append("over budget" if fixed is None else str(fixed))
        rate = row["wrong_rate_when_stopped"]
        cells.append(
            "n/a" if rate is None else f"{row['wrong_when_stopped']}/{row['stopped']} ({rate:.1%})"
        )
        cells.append(f"{row['wrong_total']}/{row['runs']} ({row['wrong_total'] / row['runs']:.1%})")
        cells.append(f"{row['tau_mean']:.3f}")
        rows.append(cells)
    strengths = ", ".join(f"{n}={t:+.3f}" for n, t in zip(settings["systems"], settings["theta"],
                                                          strict=True))
    lines = [
        f"Planted strengths: {strengths}; nu = {settings['nu']:.3f}",
        f"Rule: confidence {settings['confidence']:g} per pair, target {settings['target']}, "
        f"correction {settings['correction']}; batch {settings['batch']}, budget "
        f"{settings['budget']} new votes, {settings['runs']} runs per scheduler, seed "
        f"{settings['seed']}",
    ]
    if settings.get("start_votes"):
        lines.append(f"Starting from the {settings['start_votes']} votes already collected.")
    lines += ["", table(headers, rows), ""]
    lines.append(
        "Votes are new votes per run; a run that did not meet the rule counts as the budget. "
        "\"wrong\" means the fitted "
        + ("order" if settings["target"] == "full" else "top set")
        + " differs from the planted one."
    )
    if with_fixed:
        lines.append(
            "\"fixed design\" is the number of votes at which the rule would be met if the "
            "estimates sat exactly on the planted strengths (no noise, no floor)."
        )
    return "\n".join(lines)


def next_payload(result: Fit, scheduler: str, pairs: Sequence[tuple[int, int]], counts: Any,
                 status: Any, rule: Any) -> dict:
    totals = counts.pair_totals
    return {
        "scheduler": scheduler,
        "n_votes": result.n_votes,
        "ranking": result.ranking,
        "next": [
            {"a": result.systems[i], "b": result.systems[j], "votes_so_far": int(totals[i, j])}
            for i, j in pairs
        ],
        "stopping": {
            "confidence": rule.confidence,
            "target": rule.target,
            "correction": rule.correction,
            "look": status.look,
            "needed": status.needed,
            "settled": status.settled,
            "unsettled": [
                {"upper": p.upper, "lower": p.lower, "prob_correct": p.prob_correct}
                for p in status.unsettled
            ],
        },
    }


def next_markdown(payload: dict) -> str:
    stop = payload["stopping"]
    lines = [
        f"{payload['n_votes']} votes so far. Current ranking: " + " > ".join(payload["ranking"]),
        "",
        f"Next {len(payload['next'])} pairs (scheduler {payload['scheduler']}):",
        "",
        table(
            ["slot", "a", "b", "votes so far"],
            [[n, row["a"], row["b"], row["votes_so_far"]]
             for n, row in enumerate(payload["next"], start=1)],
        ),
        "",
        f"Stopping rule: confidence {stop['confidence']:g} per pair, target {stop['target']}, "
        f"correction {stop['correction']}, look {stop['look']} "
        f"(each pair needs P >= {stop['needed']:.4f}).",
    ]
    if stop["settled"]:
        lines.append("Settled: every deciding pair meets the rule.")
    else:
        lines += [
            f"Not settled: {len(stop['unsettled'])} pair(s) remain.",
            "",
            table(
                ["upper", "lower", "P(order correct)"],
                [[p["upper"], p["lower"], f"{p['prob_correct']:.3f}"] for p in stop["unsettled"]],
            ),
        ]
    return "\n".join(lines)


def replay_payload(result: Any) -> dict:
    ref = result.reference
    return {
        "total_votes": result.total_votes,
        "votes_per_pair": {"min": result.pool_min, "max": result.pool_max},
        "runs": result.runs,
        "batch": result.batch,
        "budget": result.budget,
        "every": result.every,
        "reference": fit_payload(ref),
        "schedulers": [
            {
                "scheduler": c.scheduler,
                "checkpoints": list(c.checkpoints),
                "tau_mean": list(c.tau_mean),
                "tau_p10": list(c.tau_p10),
                "exact_share": list(c.exact_share),
                "votes_to_mean_tau_0.9": c.votes_to_tau(0.9),
                "votes_to_mean_tau_0.95": c.votes_to_tau(0.95),
                "votes_to_exact_share_0.5": c.votes_to_exact(0.5),
                "votes_to_exact_share_0.9": c.votes_to_exact(0.9),
                "settle_votes": list(c.settle),
                "first_exhausted_at": list(c.first_exhausted),
            }
            for c in result.curves
        ],
    }


def _median(values: Sequence[int | None], missing: float) -> str:
    filled = sorted(missing if v is None else v for v in values)
    if not filled:
        return "n/a"
    mid = filled[len(filled) // 2] if len(filled) % 2 else (
        0.5 * (filled[len(filled) // 2 - 1] + filled[len(filled) // 2])
    )
    return "not reached" if mid == missing else f"{mid:.0f}"


def _mean_or_dash(values: Sequence[int | None]) -> str:
    seen = [v for v in values if v is not None]
    return "never" if not seen else f"{sum(seen) / len(seen):.0f}"


def replay_markdown(payload: dict) -> str:
    ref = payload["reference"]
    curves = payload["schedulers"]
    weakest = min(ref["adjacent"], key=lambda p: p["prob_correct"])
    lines = [
        f"{payload['total_votes']} votes, {len(ref['ranking'])} systems, "
        f"{payload['votes_per_pair']['min']} to {payload['votes_per_pair']['max']} votes per pair.",
        "Ranking fitted on all votes: " + " > ".join(r["system"] for r in ref["ranking"]),
        f"Least certain adjacent pair in that ranking: {weakest['upper']} over {weakest['lower']}"
        f" with P(order correct) = {weakest['prob_correct']:.3f}.",
        f"{payload['runs']} runs per scheduler, batch {payload['batch']}, budget "
        f"{payload['budget']} votes, ranking compared every {payload['every']} votes.",
        "",
        "Mean Kendall tau to the full-data ranking (share of runs with exactly that ranking):",
        "",
    ]
    headers = ["votes"] + [c["scheduler"] for c in curves]
    rows = []
    for k, votes in enumerate(curves[0]["checkpoints"]):
        rows.append(
            [votes] + [f"{c['tau_mean'][k]:.3f} ({c['exact_share'][k]:.2f})" for c in curves]
        )
    lines += [table(headers, rows), ""]
    never = float("inf")
    summary_rows = []
    for c in curves:
        unsettled = sum(1 for v in c["settle_votes"] if v is None)
        summary_rows.append(
            [
                c["scheduler"],
                c["votes_to_mean_tau_0.9"] or "not reached",
                c["votes_to_mean_tau_0.95"] or "not reached",
                c["votes_to_exact_share_0.5"] or "not reached",
                _median(c["settle_votes"], never),
                f"{unsettled}/{len(c['settle_votes'])}",
                _mean_or_dash(c["first_exhausted_at"]),
            ]
        )
    lines.append(
        table(
            [
                "scheduler",
                "votes to mean tau >= 0.9",
                "to mean tau >= 0.95",
                "to exact share >= 0.5",
                "median votes to settle",
                "runs not settled",
                "mean votes when a pair first ran out",
            ],
            summary_rows,
        )
    )
    lines += [
        "",
        "\"to settle\": the first checkpoint from which a run's ranking equals the full-data "
        "ranking at every later checkpoint. Thresholds on the mean count only if the curve "
        "stays above them afterwards.",
    ]
    return "\n".join(lines)
