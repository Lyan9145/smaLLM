# MP1 code — installation and usage

Read [the project guide](../GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## 7. Planned student implementation and CPU smoke checks

The student is a stateless pre-normalized decoder with RMSNorm, rotary causal
attention, SwiGLU, tied embeddings, and a gated depthwise convolution. Each
convolution pads only on the left. Attention and convolution both reset with
every independent window. `student.py` imports `student_model.py`; distribute
both alongside the selected checkpoint. Neither requires training helpers.

Configurations:

| Config | Blocks / width / MLP width | Parameters | Purpose |
| --- | --- | ---: | --- |
| `student_matched.json` | 4 / 128 / 365 | 1,088,128 | Within 128 parameters of the 1,088,256 baseline |
| `student_matched_no_local.json` | 4 / 128 / 365 | 1,086,080 | Local-branch ablation; all other settings unchanged |
| `student_8.json` | 8 / 176 / 480 | 3,387,824 | Scaled fallback |
| `student_10.json` | 10 / 176 / 480 | 4,144,624 | Initial scaled candidate |

All student configs use dropout 0.05. Pass `--dropout 0` for the short comparison
with the original dropout-free baseline. The preserved `model.py` does not
implement dropout, and the trainer rejects a nonzero baseline dropout override
rather than silently ignoring it. Choose an explicit student config: the
legacy default config remains `configs/baseline.json` for command compatibility.

### Distillation experiments

The trainer supports training-only teacher distillation. The teacher is loaded
from a checkpoint, frozen, and evaluated only on sampled training spans; the
submitted student checkpoint does not contain or require the teacher:

```bash
python train.py --implementation student --config configs/student_8.json \
  --teacher-checkpoint runs/teacher/checkpoint.pt \
  --distill-temperature 2 --distill-alpha .4 \
  --device cuda --precision bf16 --tf32 --threads 4 \
  --micro-batch-size 32 --grad-accum 4 --updates 3052 \
  --lr .0015 --warmup 122 --ema .99 --eval-every 100 \
  --run-dir runs/distilled-student
```

The teacher must itself be trained only on the supplied training text. Record
its checkpoint hash, configuration, and training cost. Select the student by
validation BPB; never use teacher outputs from validation or test.

Run the fixed FP32 CPU evaluator for each candidate and reject it if it breaks
the five-times baseline time, 4 GiB RAM, or 64 MiB asset limits. Speculative
decoding is not part of the scored implementation because this evaluator
requires probabilities at every position of every independent window.

Run the full local smoke suite (several minutes on a recent CPU):

```bash
python run_cpu_smoke.py --run-root runs/cpu-smoke --evidence evidence
```

This runs all tests, measures all model sizes on full validation, trains the
baseline/matched student/ablation for 200 updates each at the same target count,
and trains both scaled models for 40 updates. It then reconstructs each selected
checkpoint in a separate process and scores full validation using the fixed
four-thread FP32 evaluator. It checks finite learning, improving loss, BPB below
the random-weight level, exact target accounting, checkpoint reproduction, and
asset/RAM limits. Timing ratios are recorded as a hardware-dependent gate; a
failing 10-block timing gate means the 8-block fallback must be considered.
No candidate is scored on test. Repeats need a fresh `--run-root` and should use
a separate evidence directory to preserve prior measurements.

`evidence/SMOKE_REPORT.md` describes the measured results and their limits.
Checkpoints and recovery state live in ignored `runs/` directories. These are
smoke artifacts, not a frozen final submission. Source, metrics, exact commands,
and logs are committed; a final trained checkpoint still needs to be bundled
after the longer validation-selected experiments in `../Plan.md`.

### Training controls and recovery

For a new low-learning-rate phase, use `--init-checkpoint PATH/checkpoint.pt`.
This loads selected weights from an internally trained checkpoint with the same
implementation/configuration, then starts a **new** optimizer, schedule, sampler
and EMA. It does not extend an old cosine schedule through `--resume`. The
initial weights remain eligible for validation selection if further training
does not help. `--resume` and `--init-checkpoint` are mutually exclusive; exact
resume still requires identical source and recipe.

`processed_targets` in metrics counts the current recipe (including recovered
segments); `processed_targets_including_ancestry` additionally includes warm-start
parents, without double-counting exact-resume segments. Teacher checkpoint costs
are disclosed separately. Inference checkpoints carry the cumulative student
target count so further warm-start phases retain this accounting.

`../scripts/gpu_dense_search.py` runs a bounded 11-candidate dense-only search
with up to three concurrent GPU jobs. It uses the existing best self-distilled
student as a new training-only teacher, plus its original teacher for controls.
After all GPU jobs finish, it runs CPU/FP32 validation sequentially (three fresh
processes per checkpoint), measuring the baseline on the same host and reporting
peak process RAM, serialized inference assets, and timing ratios. It never scores
test. `plan.json`, `commands.jsonl`, parent provenance, source hashes, per-job
logs, and GPU telemetry preserve the search cost and reproduction evidence.
Create `STOP` in its output directory to terminate its workers gracefully.
The warm-start support and search orchestration were developed with AI assistance.

The original `--implementation model --steps 1200 --batch-size 32` recipe
retains its LR formula, sampler, optimizer defaults, and 9,830,400-target budget.
The extended trainer accepts `--micro-batch-size` (alias of `--batch-size`),
`--grad-accum`, `--updates` (alias of `--steps`), `--target-budget`, `--lr`,
`--warmup`, `--min-lr-ratio`, `--betas`, `--weight-decay`, `--dropout`,
`--eval-every`, `--ema`, `--save-every`, and `--resume`.
`--target-budget` must divide exactly by `micro_batch * grad_accum * 256`;
nondivisible budgets fail instead of silently rounding. Gradient clipping is 1.0.
`--tf32` explicitly enables CUDA training TF32; validation always restores FP32
matmul precision. CUDA BF16 is permitted only on supported devices by `setup()`.

Example GPU pilot (not yet a validated final recipe):

```bash
python train.py --implementation student --config configs/student_matched.json \
  --device cuda --precision bf16 --tf32 --threads 4 --seed 17 \
  --micro-batch-size 32 --grad-accum 4 --updates 916 \
  --lr .001 --warmup 36 --min-lr-ratio .1 --betas .9 .95 \
  --weight-decay .1 --ema .999 --eval-every 100 --save-every 100 \
  --run-dir runs/pilot-matched-s17
```

Use the same arguments and seed for the no-local config. For the baseline
schedule control, use `--implementation model --config configs/baseline.json`
and omit dropout. The 916-update budget is exactly 30,015,488 targets;
3,052 updates process 100,007,936 targets. Adjust the LR only after measured
validation pilots. See `../Plan.md` for the full experiment and selection order.

Each completed segment writes:

- `checkpoint.pt`: the best raw or EMA weights over all measured validation
  points, plus simple config/protocol metadata. Load with `weights_only=True`.
- `resume.pt`: latest raw and EMA weights, optimizer, scheduler recipe/position,
  sampler and torch CPU/CUDA RNG states, best candidate, history, and ancestry.
  Save intervals atomically replace this recovery file. Keep it out of inference
  bundles. Recovery files also load with `weights_only=True`.
- `metrics.json`: effective recipe, exact targets including ancestry, selected
  step/weight type, raw and EMA validation history, train/process seconds,
  hardware/software details, asset bytes, SHA-256 hashes, and Git provenance.
- A row appended to `RUN_LOG_TEMPLATE.csv` (or `--run-log PATH`), linking the
  immutable metrics and recording the material cost and validation result.

Resume into a **new output directory**, with `--resume old-run/resume.pt` and
exactly the same original recipe, including the total update budget, seed,
precision, thread count and validation interval. All Python/config source hashes
must match. `--stop-after UPDATE` allows a planned partial segment without
changing the full LR schedule. Do not set the total budget to the remaining
updates. The selected checkpoint may come from an earlier validation step;
reported training cost always includes all updates actually performed.
The tests verify bit-identical CPU recovery with dropout, accumulation and EMA.
CUDA nondeterministic kernels may still prevent bit-identical GPU recovery.

Runs from an uncommitted working tree explicitly record `code_dirty: true`
and source/config hashes; the recorded Git commit identifies the parent revision,
not an assertion that the working tree was clean. Commit the experiment source
before long GPU runs. Peak CPU RAM is Linux process high-water RSS, including
training and validation; GPU allocation is reported separately. Inference asset
accounting deduplicates tied tensor storage and also reports the serialized
checkpoint plus required source/config size.

Substantive AI assistance: OpenAI Codex implemented the planned model and
training tooling, wrote tests and reproduction scripts, and ran local CPU smoke
checks. The author must review and understand the method and report this
assistance in the final coursework report.
