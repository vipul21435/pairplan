"""Write examples/votes.jsonl: 400 simulated votes on six made-up systems.

The votes are drawn from the model in pairplan.model with the planted log-strengths below and
tie parameter nu = 0.4, on pairs chosen uniformly at random. Run from the repository root:

    python examples/make_votes.py
"""

import json
import math
from pathlib import Path

import numpy as np

from pairplan.model import outcome_probs

SYSTEMS = ["amber", "birch", "cedar", "dune", "ember", "flint"]
THETA = np.array([1.0, 0.6, 0.5, 0.0, -0.7, -1.4])
NU = 0.4
N_VOTES = 400


def main() -> None:
    rng = np.random.default_rng(1)
    win, tie = outcome_probs(THETA, math.log(NU))
    lines = []
    for n in range(N_VOTES):
        i, j = rng.choice(len(SYSTEMS), size=2, replace=False)
        outcome = rng.choice(["a", "b", "tie"], p=[win[i, j], win[j, i], tie[i, j]])
        record = {
            "a": SYSTEMS[i],
            "b": SYSTEMS[j],
            "winner": str(outcome),
            "judge": f"judge-{int(rng.integers(1, 6))}",
            "prompt": f"prompt-{n % 40 + 1}",
        }
        lines.append(json.dumps(record))
    out = Path(__file__).with_name("votes.jsonl")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} votes to {out}")


if __name__ == "__main__":
    main()
