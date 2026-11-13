"""Command line interface: ``pairplan fit | next | simulate | replay | plan``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__, report
from .bootstrap import bootstrap
from .model import FitError, fit
from .replay import replay
from .schedulers import SCHEDULERS, next_pairs
from .simulate import plan, simulate
from .stopping import CORRECTIONS, StopRule, check, parse_target
from .votes import Counts, Vote, VoteError, read_votes


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("md", "json"), default="md", help="output format")
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.01,
        help="weight of the quadratic penalty on log-strengths; 0 for plain maximum likelihood",
    )


def _add_columns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--systems",
        help="comma-separated list of systems (default: every system named in the votes)",
    )
    parser.add_argument("--a-col", help="column with the first system (default: a or model_a)")
    parser.add_argument("--b-col", help="column with the second system (default: b or model_b)")
    parser.add_argument("--winner-col", help="column with the winner (default: winner)")
    parser.add_argument("--judge-col", help="optional column naming the judge")
    parser.add_argument("--prompt-col", help="optional column naming the prompt")


def _add_scheduling(parser: argparse.ArgumentParser, batch: int) -> None:
    parser.add_argument("--batch", type=int, default=batch, help="pairs asked per round")
    parser.add_argument(
        "--floor", type=int, default=1, help="votes every pair gets before the adaptive rules"
    )
    parser.add_argument(
        "--floor-share",
        type=float,
        default=0.1,
        help="adaptive rules keep every pair above this share of the average votes per pair",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")


def _add_rule(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="confidence required of each deciding pair"
    )
    parser.add_argument("--target", default="full", help="'full' order or 'top-k', e.g. top-3")
    parser.add_argument(
        "--correction",
        choices=CORRECTIONS,
        default="none",
        help="none: per pair and per look; pairs: union bound over pairs; "
        "sequential: union bound over pairs and looks",
    )


def _add_schedulers(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--schedulers",
        default=",".join(SCHEDULERS),
        help="comma-separated schedulers to compare (default: all four)",
    )


def _schedulers(args: argparse.Namespace) -> list[str]:
    names = [name.strip() for name in args.schedulers.split(",") if name.strip()]
    for name in names:
        if name not in SCHEDULERS:
            raise VoteError(f"unknown scheduler {name!r}; choose from {', '.join(SCHEDULERS)}")
    return names


def _rule(args: argparse.Namespace, m: int) -> StopRule:
    try:
        parse_target(args.target, m)
        return StopRule(args.confidence, args.target, args.correction)
    except ValueError as exc:
        raise VoteError(str(exc)) from None


def load(args: argparse.Namespace, path: str | None = None) -> tuple[list[Vote], Counts]:
    votes = read_votes(
        path or args.votes,
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


def parse_strengths(text: str) -> tuple[list[str], list[float]]:
    """Parse ``"1.0,0.5,-0.2"`` or ``"alpha=1.0,beta=0.5"`` into names and log-strengths."""
    names: list[str] = []
    values: list[float] = []
    for k, part in enumerate(p for p in text.split(",") if p.strip()):
        name, sep, value = part.partition("=")
        try:
            values.append(float(value if sep else name))
        except ValueError:
            raise VoteError(f"cannot read strength {part.strip()!r}") from None
        names.append(name.strip() if sep else f"s{k + 1}")
    if len(values) < 2:
        raise VoteError("need at least two strengths")
    if len(set(names)) != len(names):
        raise VoteError("system names in --strengths must be unique")
    return names, values


def _emit(args: argparse.Namespace, payload: dict, markdown: str) -> None:
    print(report.to_json(payload) if args.format == "json" else markdown)


def cmd_fit(args: argparse.Namespace) -> int:
    votes, counts = load(args)
    result = fit(counts, ridge=args.ridge)
    payload = report.fit_payload(result)
    if args.bootstrap:
        boot = bootstrap(
            counts, n_boot=args.bootstrap, level=args.level, seed=args.seed, ridge=args.ridge,
            votes=votes, cluster=args.cluster, start=result,
        )
        index = {name: k for k, name in enumerate(result.systems)}
        intervals = boot.intervals
        ranks = boot.rank_intervals
        payload["bootstrap"] = {
            "replicates": args.bootstrap, "level": args.level, "unit": boot.unit,
            "seed": args.seed,
        }
        for row in payload["ranking"]:
            k = index[row["system"]]
            row["interval"] = [float(intervals[k, 0]), float(intervals[k, 1])]
            row["rank_interval"] = [int(ranks[k, 0]), int(ranks[k, 1])]
        for pair in payload["adjacent"]:
            pair["bootstrap_share"] = boot.order_share(index[pair["upper"]], index[pair["lower"]])
    _emit(args, payload, report.fit_markdown(payload))
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    _, counts = load(args)
    rule = _rule(args, counts.m)
    result = fit(counts, ridge=args.ridge)
    if args.scheduler not in SCHEDULERS:
        raise VoteError(f"unknown scheduler {args.scheduler!r}")
    pairs = next_pairs(
        args.scheduler, counts, args.batch, rng=np.random.default_rng(args.seed), fit=result,
        ridge=args.ridge, floor=args.floor, floor_share=args.floor_share, target=args.target,
    )
    status = check(result, rule, args.look)
    payload = report.next_payload(result, args.scheduler, pairs, counts, status, rule)
    _emit(args, payload, report.next_markdown(payload))
    return 0


def _planted(args: argparse.Namespace) -> tuple[list[str], list[float], float, Counts | None]:
    """Strengths to plant: from --strengths, or fitted from a votes file."""
    votes_path = getattr(args, "votes", None) or getattr(args, "from_votes", None)
    if args.strengths and votes_path:
        raise VoteError("give either --strengths or a votes file, not both")
    if args.strengths:
        names, theta = parse_strengths(args.strengths)
        return names, theta, args.nu if args.nu is not None else 0.0, None
    if not votes_path:
        raise VoteError("give --strengths or a votes file")
    _, counts = load(args, votes_path)
    result = fit(counts, ridge=args.ridge)
    nu = args.nu if args.nu is not None else result.nu
    return list(result.systems), [float(x) for x in result.theta], nu, counts


def _settings(args: argparse.Namespace, names: list[str], theta: list[float], nu: float) -> dict:
    return {
        "systems": names, "theta": theta, "nu": nu, "confidence": args.confidence,
        "target": args.target, "correction": args.correction, "batch": args.batch,
        "budget": args.budget, "runs": args.runs, "seed": args.seed, "floor": args.floor,
        "floor_share": args.floor_share,
    }


def cmd_simulate(args: argparse.Namespace) -> int:
    names, theta, nu, _ = _planted(args)
    rule = _rule(args, len(names))
    summaries = simulate(
        theta, nu=nu, systems=names, schedulers=_schedulers(args), runs=args.runs,
        batch=args.batch, budget=args.budget, rule=rule, seed=args.seed, ridge=args.ridge,
        floor=args.floor, floor_share=args.floor_share,
    )
    payload = report.summaries_payload(summaries, _settings(args, names, theta, nu))
    _emit(args, payload, report.summaries_markdown(payload))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    names, theta, nu, counts = _planted(args)
    rule = _rule(args, len(names))
    summaries = plan(
        theta, nu=nu, systems=names, schedulers=_schedulers(args), runs=args.runs,
        batch=args.batch, budget=args.budget, rule=rule, seed=args.seed, ridge=args.ridge,
        floor=args.floor, floor_share=args.floor_share, start=counts,
    )
    settings = _settings(args, names, theta, nu)
    settings["columns"] = ("fixed_design",)
    settings["start_votes"] = counts.total if counts is not None else 0
    payload = report.summaries_payload(summaries, settings)
    _emit(args, payload, report.summaries_markdown(payload))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    votes, counts = load(args)
    result = replay(
        votes, counts.systems, schedulers=_schedulers(args), runs=args.runs, batch=args.batch,
        budget=args.budget, every=args.every, seed=args.seed, ridge=args.ridge, floor=args.floor,
        floor_share=args.floor_share,
    )
    if args.chart:
        from .chart import replay_chart

        try:
            replay_chart(result, args.chart)
        except RuntimeError as exc:
            raise VoteError(str(exc)) from None
    payload = report.replay_payload(result)
    _emit(args, payload, report.replay_markdown(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairplan",
        description="Plan pairwise comparisons for ranking systems from human votes.",
    )
    parser.add_argument("--version", action="version", version=f"pairplan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit strengths, standard errors and adjacent-pair odds")
    p_fit.add_argument("votes", help="votes file: .jsonl, .csv or .tsv")
    _add_columns(p_fit)
    _add_format(p_fit)
    p_fit.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates (0: none)")
    p_fit.add_argument("--level", type=float, default=0.95, help="bootstrap interval level")
    p_fit.add_argument(
        "--cluster", choices=("judge", "prompt"), help="resample whole judges or prompts"
    )
    p_fit.add_argument("--seed", type=int, default=0, help="random seed for the bootstrap")
    p_fit.set_defaults(func=cmd_fit)

    p_next = sub.add_parser("next", help="the next batch of pairs to ask, and what is unsettled")
    p_next.add_argument("votes", help="votes so far: .jsonl, .csv or .tsv")
    _add_columns(p_next)
    _add_format(p_next)
    p_next.add_argument("--scheduler", default="uncertainty", choices=SCHEDULERS)
    _add_scheduling(p_next, batch=10)
    _add_rule(p_next)
    p_next.add_argument(
        "--look", type=int, default=1, help="how many times the rule has been checked, this one "
        "included (matters for --correction sequential)",
    )
    p_next.set_defaults(func=cmd_next)

    p_sim = sub.add_parser("simulate", help="planted strengths: votes needed and wrong rankings")
    p_sim.add_argument("--strengths", help="log-strengths, e.g. 1.0,0.4,-0.2 or a=1.0,b=0.4")
    p_sim.add_argument("--from-votes", help="plant the strengths fitted from this votes file")
    p_sim.add_argument("--nu", type=float, help="tie parameter (default 0, or the fitted value)")
    _add_columns(p_sim)
    _add_format(p_sim)
    _add_schedulers(p_sim)
    _add_scheduling(p_sim, batch=5)
    _add_rule(p_sim)
    p_sim.add_argument("--runs", type=int, default=200, help="runs per scheduler")
    p_sim.add_argument("--budget", type=int, default=5000, help="most votes per run")
    p_sim.set_defaults(func=cmd_simulate)

    p_plan = sub.add_parser("plan", help="expected further votes each scheduler needs")
    p_plan.add_argument(
        "votes", nargs="?", help="votes so far; their fitted strengths are taken as the truth"
    )
    p_plan.add_argument("--strengths", help="plan from scratch for these log-strengths instead")
    p_plan.add_argument("--nu", type=float, help="tie parameter (default 0, or the fitted value)")
    _add_columns(p_plan)
    _add_format(p_plan)
    _add_schedulers(p_plan)
    _add_scheduling(p_plan, batch=5)
    _add_rule(p_plan)
    p_plan.add_argument("--runs", type=int, default=100, help="simulated runs per scheduler")
    p_plan.add_argument("--budget", type=int, default=5000, help="most new votes per run")
    p_plan.set_defaults(func=cmd_plan)

    p_rep = sub.add_parser("replay", help="replay the schedulers on real votes")
    p_rep.add_argument("votes", help="votes file: .jsonl, .csv or .tsv")
    _add_columns(p_rep)
    _add_format(p_rep)
    _add_schedulers(p_rep)
    _add_scheduling(p_rep, batch=10)
    p_rep.add_argument("--runs", type=int, default=50, help="runs per scheduler")
    p_rep.add_argument("--budget", type=int, help="most votes per run (default: all votes)")
    p_rep.add_argument("--every", type=int, help="compare rankings every this many votes")
    p_rep.add_argument("--chart", help="write a PNG chart here (needs matplotlib)")
    p_rep.set_defaults(func=cmd_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (VoteError, FitError, OSError, ValueError) as exc:
        print(f"pairplan: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
