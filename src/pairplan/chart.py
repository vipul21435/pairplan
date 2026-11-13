"""Optional chart of a replay (needs matplotlib: ``pip install pairplan[plot]``)."""

from __future__ import annotations

from pathlib import Path

from .replay import ReplayResult


def replay_chart(result: ReplayResult, path: str | Path, title: str | None = None) -> Path:
    """Draw mean Kendall tau and exact-ranking share against votes used; save as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise RuntimeError("the chart needs matplotlib: pip install 'pairplan[plot]'") from exc

    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    styles = {"uniform": ":", "round-robin": "--", "adjacent": "-.", "uncertainty": "-"}
    for curve in result.curves:
        style = styles.get(curve.scheduler, "-")
        axes[0].plot(curve.checkpoints, curve.tau_mean, style, label=curve.scheduler)
        axes[1].plot(curve.checkpoints, curve.exact_share, style, label=curve.scheduler)
    axes[0].set_ylabel("mean Kendall tau to full-data ranking")
    axes[1].set_ylabel("share of runs with the full-data ranking")
    for ax in axes:
        ax.set_xlabel("votes used")
        ax.grid(True, alpha=0.3)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower right")
    fig.suptitle(title or f"Replay on {result.total_votes} votes, {result.runs} runs per scheduler")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
