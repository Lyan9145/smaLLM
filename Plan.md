# MP1 Implementation Plan

## Objective and operating decisions

Build a reproducible student language model that improves validation BPB on the
fixed WikiText-2/BPE-2048 protocol and remains comfortably inside the final
CPU evaluation limits. The first priority is a reliable, well-evidenced
submission rather than an aggressive leaderboard-only search.

- Train on a rental NVIDIA RTX 4090 (24 GiB) or RTX 5090. Design the training
  recipe to fit in 24 GiB, so either GPU can run it.
- Use more than 15 GPU-hours for measured experiments and final reproduction.
- Defer self-distillation. Revisit it only if the compact final model has passed
  all correctness and CPU-resource checks and the validation result is still
  materially short of the target.
- Use validation exclusively for architecture, seed, checkpoint, schedule, and
  mixture decisions. Do not evaluate candidate student checkpoints on test.
- Use only `code/data/wikitext_train.txt` as training input. Do not download
  external text, use pretrained weights, alter the tokenizer, or create test or
  validation lookup assets. Python and PyTorch package downloads are allowed;
  they are not model data.

The final predictor will be a normal, stateless causal model. It will reset on
each `predict_log_probs()` call, making it compatible with the evaluator's
independent 256-token windows.

## Constraints to preserve

Do not modify the benchmark contract:

- Leave `code/common.py`, `code/evaluate.py`, `code/data/`, and the supplied
  tokenizer unchanged.
- Keep `context=256`, `vocab=2048`, the fixed scorer, and the prediction shape
  `[batch, time, 2048]`.
- Predictions at input position `t` may depend only on input tokens through
  `t`; neither temporary cross-window state nor future inputs are permitted.
- Final scoring must be CPU, four threads, FP32. The reference baseline takes
  5.92 seconds, so the formal limit is 29.6 seconds. Target at most 24 seconds
  on the same CPU, leaving margin for hardware and software differences.
- Keep final uncompressed inference assets below 56 MiB, rather than treating
  64 MiB as a target. A 4-5M parameter FP32 model is about 16-20 MiB of weights
  and leaves substantial margin. Keep peak CPU evaluation RAM below 4 GiB.

The evaluator records only the implementation module and the model config in
the checkpoint. Every source module imported by `student.py`, the matching
config, and the final checkpoint must be included in the final repository or
bundle.

## Proposed method

Replace the baseline returned by `student.build_model()` with a compact,
pre-normalized decoder. It combines established small-model improvements while
retaining ordinary dense CPU inference:

1. RMSNorm before attention and MLP sublayers.
2. Causal multi-head self-attention with rotary positional embeddings (RoPE).
3. SwiGLU feed-forward layers, sized to keep the parameter count controlled.
4. A short, left-padded depthwise causal convolutional local-mixing branch.
   Its output is gated and added within each decoder block. It gives the model
   an inexpensive direct path for local BPE and punctuation patterns while the
   attention path retains full-window dependencies.
5. Tied input/output token embeddings and a final RMSNorm.

The local-mixing branch is the named mechanism for the report. It is causal by
construction: use left padding only, never symmetric padding, and add no
cached state. The primary ablation removes this branch while holding depth,
width, optimizer, targets, seed, and all other architectural choices fixed.

Start the scaled model at 10 blocks, width 176, four attention heads, and a
SwiGLU hidden width near 480. It should have roughly 4.1M parameters. This is
an initial CPU-safe candidate, not a promised final configuration: retain it
only after measuring the actual four-thread FP32 timing ratio against the
baseline. A smaller 8-block, width-176 configuration is the immediate fallback
if the timing ratio exceeds 4x.

## Code changes

1. Update `code/student.py` to expose the new `build_model(config)` factory.
   Put substantial model code in `code/student_model.py` if that makes the
   factory readable. Both files remain part of the submission.
2. Add explicit student configs under `code/configs/`, including:
   - a parameter-matched 4-block, width-128 experiment config;
   - a scaled 8-block fallback config;
   - the initial 10-block, width-176 final candidate config.
3. Extend `code/train.py` without breaking its existing baseline command.
   Add explicit arguments for micro-batch size, gradient accumulation, total
   updates or target budget, warmup, minimum learning-rate ratio, AdamW betas,
   weight decay, dropout, validation interval, EMA, save interval, and resume
   checkpoint. Compute and record the exact processed-target count as
   `updates * micro_batch * grad_accum * 256`.
4. Save separate resumable training state (optimizer, scheduler, RNG state,
   current/EMA weights) during GPU runs. The evaluator-facing final
   `checkpoint.pt` contains only the frozen model state and simple metadata;
   it must load through `torch.load(..., weights_only=True)`.
5. Record immutable metadata in `metrics.json`: code commit, config, seed,
   hardware, CUDA/PyTorch versions, precision, wall time, processed targets,
   parent checkpoint, validation history, checkpoint SHA-256, and model
   parameter count. Append the same material results to `RUN_LOG_TEMPLATE.csv`.
6. Add focused tests for the local branch, model asset size calculation, and
   EMA checkpoint selection where they cover behavior not already checked by
   `tests/test_contract.py`.

## Rental-GPU setup

On persistent disk, clone or upload the repository and create a Python 3.12
environment from `code/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX 4090: use the README-pinned CUDA 12.6 wheel.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
nvidia-smi
```

For an RTX 5090, first use the provider image and CUDA wheel recommended for
that GPU (often the CUDA 12.8 index, if the 12.6 wheel does not contain a
usable kernel). Keep the package at the required PyTorch 2.7.1 version where
available, record the exact wheel and driver in `metrics.json`, and perform a
clean CPU installation with the pinned requirements before submission.

Use a terminal multiplexer and write all run directories to persistent storage.
Copy source, configs, metrics, final checkpoints, and run logs off the instance
after each completed experiment. Keep intermediate resume checkpoints separate
from the submission artifact. The provided runner caps PyTorch allocation at
20 GB, so a 4090 has enough memory for the planned models.

Train with CUDA BF16, enabled only after confirming the GPU reports BF16
support. Use TF32 where PyTorch permits it for FP32 matrix products. Inference
code must remain standard eager PyTorch and must not require CUDA or
`torch.compile` to run.

## Training recipe

Use random contiguous 257-token spans from the supplied training-token stream.
The first 256 tokens are inputs and the shifted sequence supplies the 256
next-token targets. This matches the task's causal training interface.

The initial recipe uses a micro-batch of 32 sequences and four accumulation
steps, for an effective batch of 128 sequences (32,768 targets per optimizer
update). Use AdamW with betas `(0.9, 0.95)`, gradient clipping at 1.0, weight
decay around 0.1, a 3-5% linear warmup, cosine decay, and modest dropout
(initially 0.05-0.1). Store both raw and EMA validation scores; select one
frozen weight set only from validation.

Tune the learning rate with a short pilot before committing long runs. Do not
reuse a checkpoint without recording its ancestry and all targets processed to
create it. Resume checkpoints are for fault recovery, not a way to hide
training cost.

## Experiment sequence

Run experiments in this order. Record each in the run log before starting the
next decision.

1. **Environment and baseline check.** Run the supplied contract tests. Train
   `--implementation model` with the exact baseline recipe: seed 17, 1,200
   updates, batch size 32, and 9,830,400 processed targets. Score validation
   and measure CPU FP32 validation time. Retain the published approximate
   2.10 test BPB as the initial reference; do not use candidate test scores
   during development.

2. **Training-schedule pilot.** On the parameter-matched baseline, test two or
   three learning rates and dropout settings for about 10M processed targets.
   Choose the schedule by validation BPB and training stability. This controls
   for any improvement caused by training changes alone.

3. **Fair architecture comparison.** At approximately 30M processed targets
   (916 updates with the stated effective batch), run the selected schedule on
   the original baseline and on the parameter-matched modern decoder with the
   local branch. Keep the seed, target count, data sampler, and evaluator
   identical. This is the required same-training-target comparison.

4. **Mechanism ablation.** Repeat the modern-decoder run with only the causal
   local-mixing branch disabled. Compare validation BPB, parameter count,
   training seconds, and CPU FP32 scoring time with the full version. Do not
   change other settings in this pair.

5. **Scale and CPU gate.** Train the 8-block and 10-block candidates for about
   30M targets. For each candidate, run full validation on GPU and on CPU FP32
   with four threads. Reject candidates over the timing guardrail, then select
   the smallest configuration within the guardrail that has the best validation
   curve.

6. **Long training and seed confirmation.** Train the selected configuration
   to about 100M processed targets (3,052 updates at the stated effective
   batch), evaluating validation at roughly 10M, 30M, 60M, 80M, and 100M
   targets. Run a second seed with the same fixed recipe. Choose the final
   seed, raw/EMA state, and training endpoint solely by validation BPB.

7. **Freeze and final test.** Commit the selected source/config, write a
   freeze manifest with all hashes, and prevent further architecture or
   hyperparameter changes. Then run the full CPU FP32 test scorer once on the
   frozen checkpoint. Additional test runs may only reproduce timing or the
   frozen score, never influence selection.

The first 10-15 GPU-hours cover the baseline, pilots, matched comparison,
ablation, scale check, and at least one long run. Use remaining time for the
second final seed, a repaired rerun, or a schedule repeat. Consider
self-distillation only after this sequence is complete and only with a written
cost accounting and a fresh, validation-only comparison.

## Acceptance gates

Do not advance a candidate merely because it trains quickly. It must pass all
relevant gates:

| Gate | Evidence |
| --- | --- |
| Correctness | `python -m unittest discover -s tests -v` passes; the student model passes causality, normalization, example-independence, reset, and gradient checks. |
| Validation selection | Full validation BPB improves over the matched baseline at the same target count; the local-branch ablation establishes its contribution. |
| Inference performance | Four-thread CPU FP32 validation time is no more than 4x the locally measured baseline before considering the final candidate. |
| Resource limits | Peak CPU RAM below 4 GiB; uncompressed inference assets below 56 MiB; final model comfortably below 5x baseline CPU time. |
| Reproduction | A clean environment recreates the model from the committed source, config, and checkpoint without any resume state, external data, or network access at evaluation. |

If the 10-block model fails the CPU gate, use the 8-block fallback. If the
local-mixing branch does not improve validation at matched size, remove it from
the final method, preserve the negative ablation in the report, and select the
best validated RoPE/SwiGLU decoder instead. If long training overfits, select
the best validation checkpoint or EMA point; do not change the selection rule
after test evaluation.

## Final verification and submission evidence

For the frozen predictor, run:

```bash
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

Bundle the exact final checkpoint, student source modules, configs, installation
and training commands, run log, and report. The report must include the
provided baseline, the equal-target comparison, the local-mixing ablation,
validation-selection rule, final test BPB, inference resources, GPU/search
cost, and an acknowledgement of substantive AI assistance. Verify the bundle
from a clean CPU environment before making the immutable submission link.
