import json
from pathlib import Path

import pytest

from pairplan.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "votes.jsonl"


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_fit_markdown_on_example(capsys):
    code, out, _ = run(capsys, "fit", str(EXAMPLE))
    assert code == 0
    assert "400 votes, 6 systems" in out
    assert "| rank | system | theta | se |" in out
    assert "P(order correct)" in out


def test_fit_json_on_example(capsys):
    code, out, _ = run(capsys, "fit", str(EXAMPLE), "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_votes"] == 400
    assert [row["rank"] for row in payload["ranking"]] == [1, 2, 3, 4, 5, 6]
    thetas = [row["theta"] for row in payload["ranking"]]
    assert thetas == sorted(thetas, reverse=True)
    assert sum(thetas) == pytest.approx(0.0, abs=1e-9)
    assert len(payload["adjacent"]) == 5
    assert payload["model"] == "davidson"


def test_fit_with_systems_list_adds_unseen_system(tmp_path, capsys):
    path = tmp_path / "v.csv"
    path.write_text("a,b,winner\nx,y,a\ny,x,a\nx,y,tie\n")
    code, out, _ = run(capsys, "fit", str(path), "--systems", "x,y,z", "--format", "json")
    assert code == 0
    assert {row["system"] for row in json.loads(out)["ranking"]} == {"x", "y", "z"}


def test_errors_exit_2(tmp_path, capsys):
    path = tmp_path / "v.jsonl"
    path.write_text('{"a": "x", "b": "y", "winner": "nobody"}\n')
    code, _, err = run(capsys, "fit", str(path))
    assert code == 2 and "v.jsonl:1" in err
    code, _, err = run(capsys, "fit", str(tmp_path / "missing.jsonl"))
    assert code == 2
    path.write_text('{"a": "x", "b": "y", "winner": "a"}\n')
    code, _, err = run(capsys, "fit", str(path), "--ridge", "0")
    assert code == 2 and "ridge" in err


def test_fit_with_bootstrap_adds_intervals(capsys):
    code, out, _ = run(capsys, "fit", str(EXAMPLE), "--bootstrap", "40", "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["bootstrap"] == {"replicates": 40, "level": 0.95, "unit": "vote", "seed": 0}
    for row in payload["ranking"]:
        lo, hi = row["interval"]
        assert lo < row["theta"] < hi
        assert row["rank_interval"][0] <= row["rank"] <= row["rank_interval"][1]
    assert all(0.0 <= pair["bootstrap_share"] <= 1.0 for pair in payload["adjacent"])
    code, out, _ = run(capsys, "fit", str(EXAMPLE), "--bootstrap", "20", "--cluster", "judge")
    assert code == 0 and "95% bootstrap interval" in out and "bootstrap share" in out


def test_next_lists_pairs_and_unsettled(capsys):
    code, out, _ = run(capsys, "next", str(EXAMPLE), "--batch", "6", "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["scheduler"] == "uncertainty" and len(payload["next"]) == 6
    assert payload["stopping"]["settled"] is False
    unsettled = {(p["upper"], p["lower"]) for p in payload["stopping"]["unsettled"]}
    asked = {tuple(sorted((row["a"], row["b"]))) for row in payload["next"]}
    assert asked <= {tuple(sorted(p)) for p in unsettled}
    code, out, _ = run(capsys, "next", str(EXAMPLE), "--scheduler", "round-robin", "--batch", "3")
    assert code == 0 and "Not settled: 2 pair(s) remain." in out
    code, out, _ = run(capsys, "next", str(EXAMPLE), "--target", "top-2", "--confidence", "0.9")
    assert code == 0 and "Settled" in out
    code, _, err = run(capsys, "next", str(EXAMPLE), "--target", "top-9")
    assert code == 2 and "top-k" in err


def test_next_is_seeded_for_uniform(capsys):
    args = ("next", str(EXAMPLE), "--scheduler", "uniform", "--seed", "4", "--format", "json")
    _, first, _ = run(capsys, *args)
    _, second, _ = run(capsys, *args)
    assert first == second


def test_simulate_command(capsys):
    code, out, _ = run(
        capsys, "simulate", "--strengths", "x=0.9,y=0.0,z=-0.9", "--nu", "0.3", "--runs", "6",
        "--budget", "400", "--schedulers", "uniform,uncertainty", "--format", "json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["settings"]["systems"] == ["x", "y", "z"]
    assert [row["scheduler"] for row in payload["schedulers"]] == ["uniform", "uncertainty"]
    assert all(row["runs"] == 6 for row in payload["schedulers"])
    code, out, _ = run(capsys, "simulate", "--strengths", "1,0,-1", "--runs", "3")
    assert code == 0 and "wrong when stopped" in out and "s1=+1.000" in out
    code, out, _ = run(
        capsys, "simulate", "--from-votes", str(EXAMPLE), "--runs", "2", "--budget", "60",
        "--target", "top-1",
    )
    assert code == 0 and "amber=" in out and "top set" in out


def test_simulate_argument_errors(capsys):
    code, _, err = run(capsys, "simulate")
    assert code == 2 and "--strengths" in err
    code, _, err = run(capsys, "simulate", "--strengths", "1,zero")
    assert code == 2 and "cannot read" in err
    code, _, err = run(capsys, "simulate", "--strengths", "1")
    assert code == 2
    code, _, err = run(capsys, "simulate", "--strengths", "1,0", "--schedulers", "magic")
    assert code == 2 and "unknown scheduler" in err
    code, _, err = run(capsys, "simulate", "--strengths", "1,0", "--from-votes", str(EXAMPLE))
    assert code == 2 and "not both" in err


def test_plan_command_from_votes_and_from_strengths(capsys):
    code, out, _ = run(
        capsys, "plan", str(EXAMPLE), "--runs", "3", "--budget", "300", "--target", "top-2",
        "--format", "json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["settings"]["start_votes"] == 400
    assert all("fixed_design_votes" in row for row in payload["schedulers"])
    code, out, _ = run(capsys, "plan", "--strengths", "1,0,-1", "--runs", "3", "--budget", "500")
    assert code == 0 and "fixed design" in out and "Starting from" not in out


def test_replay_command_and_chart(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    chart = tmp_path / "replay.png"
    code, out, _ = run(
        capsys, "replay", str(EXAMPLE), "--runs", "3", "--every", "100", "--chart", str(chart)
    )
    assert code == 0
    assert "Ranking fitted on all votes: amber > birch > cedar > dune > ember > flint" in out
    assert "| 400 | 1.000 (1.00) | 1.000 (1.00) | 1.000 (1.00) | 1.000 (1.00) |" in out
    assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    code, out, _ = run(
        capsys, "replay", str(EXAMPLE), "--runs", "2", "--budget", "120", "--format", "json",
        "--schedulers", "adjacent",
    )
    payload = json.loads(out)
    assert payload["budget"] == 120 and payload["schedulers"][0]["checkpoints"][-1] == 120
