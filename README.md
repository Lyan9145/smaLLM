# MP1 Small Language Model

Student submission for the DASE7506 WikiText-2/BPE-2048 benchmark.

- **Final test BPB:** `1.5620576288` (CPU, FP32, protocol `7506-mp1-wt2-v2`)
- **Validation BPB:** `1.5285584688`
- **Model:** 10-block RoPE/SwiGLU decoder, 4.1M parameters, EMA weights
- **Checkpoint:** [final-checkpoint.pt](https://github.com/Lyan9145/smaLLM/blob/main/checkpoints/final-checkpoint.pt)
- **Code and reproduction:** [code/README.md](code/README.md)
- **Guide:** [GUIDE.md](GUIDE.md)

Run from `code/` after installing the pinned requirements:

```bash
python evaluate.py --checkpoint ../checkpoints/final-checkpoint.pt \
  --device cpu --precision fp32 --threads 4 --split test
```

The implementation and training tooling were developed with OpenAI Codex assistance.
