# Phase 2 GPU experiments

Launched 2026-09-20 on RTX 3080 Ti (12 GiB), provider Python 3.12.3,
PyTorch 2.12.1+cu130 (deviation from the project pin; clean pinned CPU reproduction
still required). Training source remains commit ac980ba; no architecture edits.
Remote output: /root/autodl-tmp/mp1-phase2-20260920
Runner PID: 4304. Read-only 15-minute status timer PID is recorded in remote check-pid.
Both use nohup, closed stdin and redirected stdout/stderr.

The phase-1 no-local student achieved validation BPB 1.733567 at 30,015,488
training targets. EMA .999 lagged substantially (2.4279 at update 916).

Phase 2 tests EMA .99 at matched size, then no-local 8-block models with
LR .001/.002 at dropout .05, an 8-block dropout .1 variant, and a 10-block
LR .001/dropout .05 variant. All use 916 updates, seed 17 and 30,015,488 targets.
The EMA control retains dropout zero to match phase 1. Every completed model
receives full CPU FP32 validation. Eligibility requires scoring time <=24 s and
<=4x a fresh local baseline, plus serialized inference assets <56 MiB.
The lowest validation BPB eligible candidate gets two from-scratch 3,052-update
runs (100,007,936 targets each), seeds 17 and 42. No test scoring occurs.
Training/evaluation errors halt the queue. Final memory and clean-environment
reproduction gates remain necessary; this is not a frozen submission.

The 900-second timer writes status-15min.json on the server. It does not wake
Codex or notify the user. Check COMPLETE/FAILED, nohup.log, per-run logs,
search_results.json and selection.json. The exact commands are in commands.jsonl.

Research consulted (not claims of SOTA on this benchmark):
- https://arxiv.org/abs/2002.05202 — GLU variants improve Transformer FFNs;
  SwiGLU is already implemented.
- https://arxiv.org/abs/2305.16264 — repeated data has diminishing returns;
  prefer measured validation curves over indiscriminately longer training.
- https://arxiv.org/abs/1708.02182 — regularization and weight averaging helped
  small-data language modeling; LSTM results are not directly transferable.
- https://arxiv.org/abs/2302.13971 — LLaMA architecture context, not external
  weights or training data.
