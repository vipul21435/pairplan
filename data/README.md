# Data

`mt_bench_human.jsonl` holds the 3,355 human votes of the MT-Bench human judgments dataset,
split `human`, from https://huggingface.co/datasets/lmsys/mt_bench_human_judgments, licensed
CC BY 4.0 as stated on the dataset card. The dataset accompanies:

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena". NeurIPS 2023 Datasets and Benchmarks Track.
https://arxiv.org/abs/2306.05685

Changes made: only the columns `question_id`, `model_a`, `model_b`, `winner`, `judge` and `turn`
are kept; the conversation texts are dropped. Rows and their order are unchanged. The file is
written by `examples/fetch_mt_bench.py`:

```
python examples/fetch_mt_bench.py data/mt_bench_human.jsonl
```

System names are the dataset's own `model_a` / `model_b` values.
