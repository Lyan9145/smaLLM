# CPU smoke verification

This implements the code and local verification phase of `Plan.md`. It does
not complete the rental-GPU experiment campaign or establish final submission
quality. No candidate was evaluated on test.

## Method and scope

The student uses RMSNorm, rotary causal attention, SwiGLU, tied embeddings,
and a gated, left-padded depthwise convolution in each block. The matched
configuration has 1,088,128 parameters, versus 1,088,256 in the original baseline.
The no-local configuration removes only the convolution and its gate, leaving
1,086,080 parameters. The scaled models have 3,387,824 and 4,144,624 parameters.

All measurements use the supplied data/tokenizer and unchanged scorer. Training
samples only the training token stream. Every validation pass scores all 376,599
targets across 1,148,007 UTF-8 bytes. The CPU is an AMD Ryzen AI 9 H 465 under
WSL2, Python 3.12.3, PyTorch 2.7.1+cpu, NumPy 2.5.3, and tokenizers 0.21.4.
Training and scoring use FP32 and four PyTorch threads.

Run `python run_cpu_smoke.py` from `code/` using the pinned environment and a
fresh run directory. `commands.json` contains every exact command used here;
per-run metrics and evaluator JSON files contain full numerical results and
checkpoint/source hashes. Actual checkpoints remain in
`code/runs/cpu-smoke/<configuration>/checkpoint.pt` locally (ignored by Git).
There is no final frozen checkpoint bundle yet.

## Checks

- All 11 tests pass: fixed contract tests, convolution impulse response and
  channel isolation, local-branch gradients, ablation structure, tied-storage
  asset accounting, EMA validation selection and safe loading, baseline LR
  compatibility, and complete resume equivalence.
- The recovery test compares an uninterrupted four-update run with a two-update
  segment plus recovery. It uses dropout, two micro-batches per update and EMA.
  Raw weights, EMA, optimizer tensors, and CPU/sampler RNG states match exactly.
- Every trained checkpoint is reconstructed and scored by `evaluate.py` in a
  fresh process, loaded with `weights_only=True`. Its full-validation BPB must
  match the selected training validation result within 1e-6.
- Smoke assertions require falling training loss, BPB below 3.5 (random
  initialization is about 3.62), exact processed-target accounting, inference
  assets below 56 MiB and training/evaluation process peak RSS below 4 GiB.
- The original `common.py`, `evaluate.py`, `model.py`, baseline config, data,
  tokenizer, and supplied contract tests remain unchanged.

## Training recipe

The baseline, matched student and local ablation each train for 200 updates,
with micro-batch 8 and accumulation 2: **819,200 processed targets each**.
They share seed 17, LR 0.001, warmup 10, minimum LR ratio 0.1, AdamW betas
(0.9, 0.95), weight decay 0.1, clipping 1.0, dropout 0, and EMA decay 0.9.
Raw and EMA are evaluated at updates 100 and 200. The lowest validation BPB
selects the frozen weights; all 200 updates count toward training cost.

Each scaled model trains for 40 updates with micro-batch 2, accumulation 2,
and config dropout 0.05: **40,960 targets each**. Other optimization settings
are the same; validation runs at updates 20 and 40. These smaller-budget runs
only establish that the larger configurations train, serialize and reconstruct.
They are not a quality comparison against the longer small-model runs.

The smoke uses a shorter, explicitly recorded recipe, not the original
1,200-update baseline or the planned 30M/100M-target experiment budgets.
Individual same-seed models can have different initial weights because their
module structures consume initialization randomness differently. A one-seed
short run cannot isolate a small mechanism benefit from initialization variation.

## Results

| Configuration | Targets | Validation BPB | Final loss | CPU score time | Time ratio | Assets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 819,200 | 2.6746 | 5.7001 | 5.78 s | 1.00x | 5.17 MiB |
| Matched + local | 819,200 | 2.6095 | 5.5689 | 7.77 s | 1.34x | 4.17 MiB |
| Matched, no local | 819,200 | 2.6004 | 5.5415 | 7.21 s | 1.25x | 4.16 MiB |
| 8 blocks (short) | 40,960 | 3.0594 | 6.4963 | 16.85 s | 2.91x | 12.95 MiB |
| 10 blocks (short) | 40,960 | 3.0808 | 6.5259 | 20.88 s | 3.61x | 15.84 MiB |

## Interpretation and remaining work

These checks establish a working causal implementation, actual short-run
learning, recoverable training, and practical CPU resource headroom. A smoke
score is not expected to reach the published approximately 2.10 **test** BPB;
that reference uses much more training and a different split. This report only
uses validation BPB.

The no-local ablation is slightly better in this short run: **2.6004 BPB**
versus **2.6095** with local mixing (a 0.0091 BPB difference). Thus this
smoke does **not** establish a quality benefit for the named local mechanism.
Both improve on the matched baseline, but the longer paired ablation is needed
before retaining local mixing in a final predictor. No final architecture is
selected here.

Before choosing a final predictor, run the plan's full baseline, schedule
pilots, equal-target architecture comparison, local-branch ablation, scaled
GPU runs and second seed. Keep or remove the local branch based on those
validation experiments, not this short smoke alone. CUDA BF16/TF32 and long-run
GPU behavior have not been exercised on this CPU-only environment. Recheck
four-thread FP32 resource limits on the final selected weights, freeze source
and checkpoint hashes, then score test and verify the standalone inference
bundle in a clean CPU environment. No GPU rental or final-test result is claimed.

The evidence records the parent Git revision plus `code_dirty: true`, with
SHA-256 values for the implemented Python/config files. This is intentional:
the requested commit follows local verification. Reproduction should use the
commit containing this report; the source hashes identify the measured code.

OpenAI Codex provided substantive implementation, testing, and smoke-analysis
assistance; acknowledge this in the eventual coursework report.
