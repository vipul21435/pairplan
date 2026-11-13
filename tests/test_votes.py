import json

import numpy as np
import pytest

from pairplan.votes import Counts, Vote, VoteError, all_pairs, parse_outcome, read_votes, systems_of


def test_parse_outcome_sides_names_and_ties():
    assert parse_outcome("a", "x", "y") == "a"
    assert parse_outcome("model_b", "x", "y") == "b"
    assert parse_outcome("y", "x", "y") == "b"
    assert parse_outcome("Tie (bothbad)", "x", "y") == "tie"
    assert parse_outcome("draw", "x", "y") == "tie"
    with pytest.raises(VoteError):
        parse_outcome("z", "x", "y")


def test_system_literally_named_b_is_resolved_by_name():
    assert parse_outcome("b", "b", "c") == "a"


def test_read_jsonl_and_csv_agree(tmp_path):
    rows = [
        {"model_a": "x", "model_b": "y", "winner": "model_a", "judge": "j1", "question_id": 7},
        {"model_a": "y", "model_b": "z", "winner": "tie"},
        {"model_a": "z", "model_b": "x", "winner": "x"},
    ]
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    csv_path = tmp_path / "v.csv"
    csv_path.write_text(
        "model_a,model_b,winner,judge,question_id\nx,y,model_a,j1,7\ny,z,tie,,\nz,x,x,,\n"
    )
    from_jsonl = read_votes(jsonl)
    assert from_jsonl == read_votes(csv_path)
    assert from_jsonl[0] == Vote("x", "y", "a", "j1", "7")
    assert from_jsonl[2].outcome == "b"
    assert systems_of(from_jsonl) == ["x", "y", "z"]


def test_read_custom_columns(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text(json.dumps({"first": "x", "second": "y", "pick": "second"}) + "\n")
    with pytest.raises(VoteError, match="v.jsonl:1"):
        read_votes(path)
    with pytest.raises(VoteError, match="neither side"):
        read_votes(path, a_col="first", b_col="second", winner_col="pick")
    path.write_text(json.dumps({"first": "x", "second": "y", "pick": "y"}) + "\n")
    votes = read_votes(path, a_col="first", b_col="second", winner_col="pick")
    assert votes == [Vote("x", "y", "b")]


def test_bad_lines_are_reported_with_line_numbers(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"a": "x", "b": "y", "winner": "a"}\nnot json\n')
    with pytest.raises(VoteError, match="v.jsonl:2"):
        read_votes(path)
    path.write_text('{"a": "x", "b": "x", "winner": "a"}\n')
    with pytest.raises(VoteError, match="with itself"):
        read_votes(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(VoteError, match="JSON object"):
        read_votes(path)


def test_counts_from_votes():
    votes = [Vote("x", "y", "a"), Vote("y", "x", "a"), Vote("x", "z", "tie"), Vote("z", "y", "b")]
    counts = Counts.from_votes(votes)
    assert counts.systems == ("x", "y", "z")
    assert counts.wins.tolist() == [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    assert counts.ties.tolist() == [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    assert counts.total == 4
    assert np.array_equal(counts.pair_totals, counts.pair_totals.T)
    assert counts.pair_totals[0, 1] == 2
    clone = counts.copy()
    clone.add(0, 1, "a")
    assert counts.total == 4 and clone.total == 5


def test_counts_reject_unknown_systems_and_outcomes():
    with pytest.raises(VoteError, match="not in the list"):
        Counts.from_votes([Vote("x", "q", "a")], systems=["x", "y"])
    with pytest.raises(VoteError):
        Counts.empty(["x"])
    with pytest.raises(VoteError):
        Counts.empty(["x", "x"])
    counts = Counts.empty(["x", "y"])
    with pytest.raises(VoteError):
        counts.add(0, 1, "maybe")
    with pytest.raises(VoteError):
        counts.add(0, 0, "a")


def test_all_pairs_order():
    assert all_pairs(3) == [(0, 1), (0, 2), (1, 2)]
