"""Download the human votes of the MT-Bench human judgments dataset as a small JSONL file.

Source: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments (split "human",
licence CC BY 4.0 as stated on the dataset card). Only the vote columns are kept
(question_id, model_a, model_b, winner, judge, turn); the conversations are dropped.

    python examples/fetch_mt_bench.py data/mt_bench_human.jsonl

Uses only the standard library and the public dataset viewer API.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "lmsys/mt_bench_human_judgments"
KEEP = ("question_id", "model_a", "model_b", "winner", "judge", "turn")
PAGE = 100


def fetch_page(offset: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "human", "offset": offset,
         "length": PAGE}
    )
    last: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise SystemExit(f"giving up at offset {offset}: {last}")


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else "mt_bench_human.jsonl"
    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        page = fetch_page(len(rows))
        total = int(page["num_rows_total"])
        if not page["rows"]:
            break
        for item in page["rows"]:
            rows.append({key: item["row"][key] for key in KEEP})
        time.sleep(0.2)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} votes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
