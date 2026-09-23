# Parallel dense validation search — 22 September 2026

Remote: `ssh -p 38916 root@connect.nmb2.seetacloud.com` (public-key access).

- Supervisor PID: `1774`, started with `nohup`.
- Output: `/root/autodl-tmp/mp1-dense-search-20260922`
- Main log: `/root/autodl-tmp/mp1-dense-search-20260922.nohup.log`
- Frozen source: `/root/autodl-tmp/mp1-dense-search-source-20260922`
- Three GPU workers; 11 candidates; 43,260 updates / 1,417,543,680 new
  student targets plus three 3-update GPU smoke runs. Prior student/teacher
  training cost remains in `parent_provenance.json` and per-run metrics.

## Candidates

- 2,000-update continuations of best self-distilled weights, LR .0002/.0005,
  retaining original seed-42 teacher; new optimizer, EMA and sampler (seed 1701).
- Original-teacher 6,000-update dense control.
- Second-generation teacher 6,000-update run.
- Second-generation 3,052-update control and one-factor variations:
  alpha .1/.4 at T=2; T=1.5/3 at alpha=.25.
- 10- and 12-block dense, 6,000 updates with second-generation teacher.

Default alpha .25, T=2; LR .0015, AdamW betas (.9,.999), dropout .05,
EMA .99, batch 32 x accumulation 4. Full validation every 100 updates.
Original teacher: `mp1-phase2-20260920/long-s42/checkpoint.pt`. New teacher:
`mp1-ideas-20260922/self-distill/checkpoint.pt` (incumbent BPB 1.556994).

17 unit tests passed locally and remotely. Three simultaneous BF16 GPU smoke
runs (continuation, dense10, dense12) passed before actual candidates launched.
First three workers active at 22:42 CST; observed 5,416 MiB and 100% utilization.

## Handoff

The supervisor automatically starts remaining jobs as workers become available.
Only after all GPU training finishes, sequential fresh-process CPU FP32 scoring
runs three times per checkpoint, including baseline and incumbent. Evaluator,
data and tokenizer are unchanged; no test scores are computed.

Inspect `STATUS`, `logs/`, `training_status.json`, `results.json`,
`selection.json`, `costs.json`. `COMPLETE` or `FINISHED_WITH_FAILURES` marks the end;
`FAILED` marks a fatal supervisor/preflight failure. `ACTIVE.json` lists children.

Create `STOP` in the output directory to request worker termination. This is an
abort, not an automatic pause/resume; last recovery states remain available.
Do not edit the frozen source during training. Exact-resume checks are preserved.
