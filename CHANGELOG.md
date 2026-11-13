# Changelog

## 0.1.0

First version.

- Votes from JSONL, CSV or TSV, tabulated per pair.
- Bradley-Terry fit with Davidson's tie parameter by Newton's method: strengths, standard
  errors, covariance, and the probability that each adjacent pair of the ranking is ordered
  correctly. With `--ridge 0`, whether the maximum-likelihood estimate exists is decided from
  the pattern of results before fitting.
- Bootstrap intervals over votes, judges or prompts.
- Schedulers: uniform, round-robin, adjacent and uncertainty, with batch mode and a floor.
- Stopping rules for the full order or a top-k set, with no correction, a union bound over
  pairs, or a union bound over pairs and looks.
- `simulate` and `plan` with planted strengths; `replay` on real votes without reusing a vote.
- Command line: `pairplan fit`, `next`, `simulate`, `replay`, `plan`, with Markdown or JSON
  output and an optional replay chart.
- Replay and simulation results on the MT-Bench human judgments in the README.
