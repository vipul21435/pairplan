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
