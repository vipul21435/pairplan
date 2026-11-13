# pairplan

Plan pairwise comparisons for ranking systems from human votes: which pairs to show next, how many
votes are needed, and when the ranking is settled.

This is an early version. The section "What works so far" is the honest scope.

## Why the problem is hard

With m systems there are m(m-1)/2 pairs. Votes are noisy and cost money. Two systems that are far
apart need almost no votes to order, two that are close need many, and nobody knows which is which
before voting starts. A schedule that looks at the votes so far to decide what to ask next, or
when to stop, can also bias the result unless that is accounted for.

## Install

```
pip install git+https://github.com/vipul21435/pairplan
```

Python 3.10 or newer. The only runtime dependency is numpy.

## What works so far

- Reading votes from JSONL, CSV or TSV (`a`, `b`, `winner`, optional `judge` and `prompt`
  columns; `model_a` / `model_b` and a winner given as a system name are understood).
- Fitting a Bradley-Terry model with a tie parameter by Newton's method, with standard errors,
  the covariance of the strengths, and for each adjacent pair of the ranking the
  normal-approximation probability that its order is right.
- `pairplan fit` with Markdown or JSON output.

Not implemented yet: bootstrap intervals, the schedulers (`pairplan next`), stopping rules,
`simulate`, `replay`, `plan`, and the run on real public votes.

## Quickstart

`examples/votes.jsonl` holds 400 simulated votes on six made-up systems (written by
`examples/make_votes.py` with planted log-strengths 1.0, 0.6, 0.5, 0.0, -0.7, -1.4 and tie
parameter 0.4).

```
$ pairplan fit examples/votes.jsonl
400 votes, 6 systems, model davidson (tie parameter nu = 0.414), ridge 0.01

| rank | system | theta | se |
| --- | --- | --- | --- |
| 1 | amber | +1.083 | 0.189 |
| 2 | birch | +0.884 | 0.180 |
| 3 | cedar | +0.257 | 0.169 |
| 4 | dune | -0.072 | 0.183 |
| 5 | ember | -0.694 | 0.173 |
| 6 | flint | -1.459 | 0.214 |

Adjacent pairs of the ranking:

| upper | lower | difference | se | P(order correct) |
| --- | --- | --- | --- | --- |
| amber | birch | 0.199 | 0.268 | 0.771 |
| birch | cedar | 0.628 | 0.259 | 0.992 |
| cedar | dune | 0.328 | 0.268 | 0.890 |
| dune | ember | 0.622 | 0.271 | 0.989 |
| ember | flint | 0.765 | 0.285 | 0.996 |
```

## The model

For systems i and j with log-strengths theta_i and theta_j and a tie parameter nu >= 0, one vote
has three outcomes:

```
P(i wins) = exp(theta_i) / D
P(j wins) = exp(theta_j) / D
P(tie)    = nu * exp((theta_i + theta_j) / 2) / D
D         = exp(theta_i) + exp(theta_j) + nu * exp((theta_i + theta_j) / 2)
```

With nu = 0 this is the Bradley-Terry model, and nu is fixed at 0 when the votes contain no
ties. Strengths are reported centred (they sum to zero). The fit maximises the log-likelihood
minus a small quadratic penalty `ridge / 2 * (|theta|^2 + (log nu)^2)` (default 0.01), which keeps
the estimate finite when a system has won or lost every vote; `--ridge 0` gives the plain
maximum-likelihood estimate and fails with a message when it does not exist. Standard errors come
from the inverse of the (penalised) information matrix.

## Limitations

- The model has no judge or prompt effects and no position (first-shown) effect.
- "P(order correct)" is a large-sample normal approximation; with few votes it is rough.
- Everything listed under "Not implemented yet" above.

## Licence

MIT. See `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
