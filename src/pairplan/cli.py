"""Command line interface: ``pairplan fit`` (more commands are added as they land)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__, report
from .model import FitError, fit
from .votes import Counts, Vote, VoteError, read_votes


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("votes", help="votes file: .jsonl, .csv or .tsv")
    parser.add_argument(
        "--systems",
        help="comma-separated list of systems (default: every system named in the votes)",
    )
    parser.add_argument("--a-col", help="column with the first system (default: a or model_a)")
    parser.add_argument("--b-col", help="column with the second system (default: b or model_b)")
    parser.add_argument("--winner-col", help="column with the winner (default: winner)")
    parser.add_argument("--judge-col", help="optional column naming the judge")
    parser.add_argument("--prompt-col", help="optional column naming the prompt")
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.01,
        help="weight of the quadratic penalty on log-strengths; 0 for plain maximum likelihood",
    )
    parser.add_argument("--format", choices=("md", "json"), default="md", help="output format")


def load(args: argparse.Namespace) -> tuple[list[Vote], Counts]:
    votes = read_votes(
        args.votes,
        a_col=args.a_col,
        b_col=args.b_col,
        winner_col=args.winner_col,
        judge_col=args.judge_col,
        prompt_col=args.prompt_col,
    )
    systems = None
    if args.systems:
        systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    return votes, Counts.from_votes(votes, systems)


def _emit(args: argparse.Namespace, payload: dict, markdown: str) -> None:
    print(report.to_json(payload) if args.format == "json" else markdown)


def cmd_fit(args: argparse.Namespace) -> int:
    _, counts = load(args)
    result = fit(counts, ridge=args.ridge)
    payload = report.fit_payload(result)
    _emit(args, payload, report.fit_markdown(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairplan",
        description="Plan pairwise comparisons for ranking systems from human votes.",
    )
    parser.add_argument("--version", action="version", version=f"pairplan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit strengths, standard errors and adjacent-pair odds")
    _add_input(p_fit)
    p_fit.set_defaults(func=cmd_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (VoteError, FitError, OSError) as exc:
        print(f"pairplan: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
