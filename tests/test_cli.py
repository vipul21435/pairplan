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
