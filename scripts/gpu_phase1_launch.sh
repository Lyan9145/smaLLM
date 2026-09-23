#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MP1_student_starter/code
PYTHON=/root/autodl-tmp/mp1-venv/bin/python
OUTPUT=/root/autodl-tmp/mp1-phase1-20260920
mkdir -p "$OUTPUT"
"$PYTHON" -m pip freeze > "$OUTPUT/packages.txt"
nvidia-smi -q > "$OUTPUT/gpu.txt"
git rev-parse HEAD > "$OUTPUT/source_commit.txt"
sha256sum /root/autodl-tmp/gpu_phase1.py /root/autodl-tmp/mp1-gpu-preflight.py > "$OUTPUT/launcher_sha256.txt"
"$PYTHON" -m unittest discover -s tests -v > "$OUTPUT/tests.log" 2>&1
PYTHONPATH="$PWD" "$PYTHON" -u /root/autodl-tmp/mp1-gpu-preflight.py > "$OUTPUT/gpu-preflight.log" 2>&1
exec "$PYTHON" -u /root/autodl-tmp/gpu_phase1.py --output "$OUTPUT"
