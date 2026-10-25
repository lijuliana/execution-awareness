#!/usr/bin/env bash
# Fresh-process replication of SauersML/watching run 034 (probe detector) on one L40S.
# Stages: A validate released detector; B compile-to-compile control; C retrain the 034 recipe;
#         D identical-input null on the retrain; E validate the retrained tail in a fresh process.
# Every stage is a new process. Upstream code is pinned to the commit used for the seven-GPU check (run 062).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
COMMIT=0b3e7445b226a8a09d5bb96892b12a96d8afebe4
REV=b968826d9c46dd6066d109eabc6255188de91218
TAIL_SHA=8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a
WORK=$HOME/watching-rep
OUT=$WORK/out
mkdir -p "$WORK" "$OUT/env" "$OUT/logs"
cd "$WORK"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/stages.log"; }
stage_status() { echo "{\"stage\":\"$1\",\"exit\":$2,\"ended\":\"$(date -u +%FT%TZ)\"}" >> "$OUT/stage_status.jsonl"; }

log "stage 0: environment"
nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/env/gpu.csv" 2>&1
uname -a > "$OUT/env/uname.txt"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; fi
if [[ ! -d src ]]; then git clone --quiet https://github.com/SauersML/watching.git src; fi
git -C src checkout --quiet "$COMMIT"
git -C src rev-parse HEAD > "$OUT/env/source_commit.txt"
if [[ ! -x venv/bin/python ]]; then uv venv --quiet --python 3.12 venv; fi
uv pip install --quiet --python venv/bin/python "./src[dev,analysis]" 2>&1 | tail -3
export PATH="$WORK/venv/bin:$PATH"
python -m pip freeze > "$OUT/env/environment.txt" 2>/dev/null || uv pip freeze --python venv/bin/python > "$OUT/env/environment.txt"
python - <<'PY' | tee -a "$OUT/env/torch.txt"
import torch, transformers, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "triton", triton.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
grep -q "available True" "$OUT/env/torch.txt" || { log "CUDA not available; aborting"; exit 3; }

log "stage 0b: model and artifact downloads"
MODEL_PATH=$(python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-8B", revision="$REV"))
PY
)
echo "$MODEL_PATH" > "$OUT/env/model_path.txt"
mkdir -p artifacts/sft-034
for f in native_tail.pt SHA256SUMS; do
  [[ -f artifacts/sft-034/$f ]] || curl --fail --location --retry 3 --silent \
    "https://github.com/SauersML/watching/releases/download/sft-034/$f" --output "artifacts/sft-034/$f"
done
got=$(sha256sum artifacts/sft-034/native_tail.pt | cut -d' ' -f1)
echo "native_tail.pt sha256 $got expected $TAIL_SHA" | tee "$OUT/env/artifact_sha.txt"
[[ $got == "$TAIL_SHA" ]] || { log "released checkpoint SHA mismatch; aborting"; exit 4; }
sha256sum src/data/validation_v2/dataset.json | tee "$OUT/env/data_sha.txt"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4

log "stage A: released 034 detector, fresh process, whole model compiled, 64 held-out prompts, probe on/off"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
python -m watching.validate_tail --checkpoint Qwen/Qwen3-8B --revision "$REV" \
  --tail artifacts/sft-034/native_tail.pt --data src/data/validation_v2/dataset.json \
  --out "$OUT/A_fresh034" --split test --max-prompts 64 --length 96 > "$OUT/logs/A_fresh034.log" 2>&1
rc=$?; stage_status A_fresh034 $rc; log "stage A exit $rc"; rm -rf "$TORCHINDUCTOR_CACHE_DIR"

log "stage B: compile-to-compile control (run 048 procedure), unprobed model compiled independently"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
python -m watching.compile_control --checkpoint Qwen/Qwen3-8B --revision "$REV" \
  --data src/data/validation_v2/dataset.json --out "$OUT/B_compile_control" \
  --prompts 4 --probe-layer 14 --length 96 > "$OUT/logs/B_compile_control.log" 2>&1
rc=$?; stage_status B_compile_control $rc; log "stage B exit $rc"; rm -rf "$TORCHINDUCTOR_CACHE_DIR"

log "stage C: retrain the 034 recipe from original weights (256 prompts, 1024 updates, 8 pairs, lr 5e-6, seed 20261017)"
bash src/scripts/run_tail_train.sh "$OUT/C_retrain" "$MODEL_PATH" scaled > "$OUT/logs/C_retrain.log" 2>&1
rc=$?; stage_status C_retrain $rc; log "stage C exit $rc"

log "stage D: identical-input null on the retrain's cached states (must stay at chance)"
python -m watching.null_tail --source-results "$OUT/C_retrain/fit" --revision "$REV" \
  --out "$OUT/D_null" > "$OUT/logs/D_null.log" 2>&1
rc=$?; stage_status D_null $rc; log "stage D exit $rc"

log "stage E: the retrained tail in a fresh process on the 64 held-out validation_v2 prompts"
if [[ -f "$OUT/C_retrain/fit/native_tail.pt" ]]; then
  export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
  python -m watching.validate_tail --checkpoint Qwen/Qwen3-8B --revision "$REV" \
    --tail "$OUT/C_retrain/fit/native_tail.pt" --data src/data/validation_v2/dataset.json \
    --out "$OUT/E_fresh_retrain" --split test --max-prompts 64 --length 96 > "$OUT/logs/E_fresh_retrain.log" 2>&1
  rc=$?; rm -rf "$TORCHINDUCTOR_CACHE_DIR"
else
  rc=99
fi
stage_status E_fresh_retrain $rc; log "stage E exit $rc"

log "stage F: package results (tensors excluded)"
cd "$WORK"
tar --exclude='*.pt' --exclude='*.safetensors' -czf results.tgz -C "$WORK" out
ls -la results.tgz | tee -a "$OUT/logs/stages.log"
log "ALL DONE"
