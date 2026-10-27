#!/usr/bin/env bash
set -uo pipefail
WORK=$HOME/sweep; D=$WORK/out/dry3; rm -rf "$D"; mkdir -p "$D"
export PATH="$WORK/venv/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export SIDECHAN_HF_ROOT=$HOME/.cache/huggingface/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$WORK/sc/sidechan"; export PYTHONPATH="$WORK/sc/sidechan:$WORK/code"
X="python $WORK/code/extra_cond.py"
t() { local n=$1; shift; local t0=$(date +%s); "$@" > "$D/$n.log" 2>&1; echo "$n exit $? seconds $(( $(date +%s) - t0 ))" >> "$D/summary.txt"; }
TINY="--n-train 48 --n-dev 8 --n-test 16 --steps 16 --eval-every 8 --test-perturb-seeds 1 --test-gauss 1e-3"
t tiny_w-int4 $X tailexp --channel cond --cond w-int4 --keep-ordinary 32 $TINY --out "$D/tiny_w-int4"
t selftest $X selftest
find "$D" -name '*.safetensors' -delete
touch "$WORK/DRY3_DONE"
