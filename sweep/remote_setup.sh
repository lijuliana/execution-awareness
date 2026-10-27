#!/usr/bin/env bash
# Environment for the sweep on one L40S: the watching package at the commit used for the replication,
# the sidechan code at f73d3a7, the pinned Qwen3-8B revision, and the Alpaca prompts.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PIN=0b3e7445b226a8a09d5bb96892b12a96d8afebe4
SC=f73d3a7021912bea7b29c564c5191bcd7b3197d2
REV=b968826d9c46dd6066d109eabc6255188de91218
TAIL_SHA=8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a
WORK=$HOME/sweep; OUT=$WORK/out
mkdir -p "$WORK" "$OUT/env" "$OUT/logs"; cd "$WORK"
for i in $(seq 1 120); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done   # GCP images install the driver on first boot
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/env/gpu.csv" 2>&1
uname -a > "$OUT/env/uname.txt"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[[ -d src ]] || git clone --quiet https://github.com/SauersML/watching.git src
git -C src checkout --quiet "$PIN"
[[ -d sc ]] || git -C src worktree add --quiet --detach "$WORK/sc" "$SC"
git -C sc rev-parse HEAD > "$OUT/env/sidechan_commit.txt"; git -C src rev-parse HEAD > "$OUT/env/watching_commit.txt"
[[ -x venv/bin/python ]] || uv venv --quiet --python 3.12 venv
uv pip install --quiet --python venv/bin/python "./src[dev,analysis]" datasets 2>&1 | tail -3
export PATH="$WORK/venv/bin:$PATH"
uv pip freeze --python venv/bin/python > "$OUT/env/environment.txt"
python - <<'PY' | tee "$OUT/env/torch.txt"
import torch, transformers, triton
print("torch", torch.__version__, "transformers", transformers.__version__, "triton", triton.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "available", torch.cuda.is_available())
PY
grep -q "available True" "$OUT/env/torch.txt" || { echo "CUDA not available"; exit 3; }
python - <<PY | tee "$OUT/env/model_path.txt"
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-8B", revision="$REV"))
from datasets import load_dataset
print("alpaca rows", len(load_dataset("tatsu-lab/alpaca")["train"]))
PY
mkdir -p artifacts/sft-034
for f in native_tail.pt; do
  [[ -f artifacts/sft-034/$f ]] || curl --fail --location --retry 3 --silent \
    "https://github.com/SauersML/watching/releases/download/sft-034/$f" --output "artifacts/sft-034/$f"
done
got=$(sha256sum artifacts/sft-034/native_tail.pt | cut -d' ' -f1)
[[ $got == "$TAIL_SHA" ]] || { echo "released checkpoint SHA mismatch"; exit 4; }
ls "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots" > "$OUT/env/snapshots.txt"
touch "$WORK/SETUP_OK"; echo "setup ok"
