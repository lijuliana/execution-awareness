#!/usr/bin/env bash
set -uo pipefail
WORK=$HOME/sweep; D=$WORK/out/dry2; rm -rf "$D"; mkdir -p "$D"
export PATH="$WORK/venv/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export SIDECHAN_HF_ROOT=$HOME/.cache/huggingface/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$WORK/sc/sidechan"; export PYTHONPATH="$WORK/sc/sidechan:$WORK/code"
X="python $WORK/code/extra_cond.py"
t() { local n=$1; shift; local t0=$(date +%s); "$@" > "$D/$n.log" 2>&1; echo "$n exit $? seconds $(( $(date +%s) - t0 ))" >> "$D/summary.txt"; }
t selftest $X selftest
TINY="--n-train 48 --n-dev 8 --n-test 16 --steps 16 --eval-every 8 --test-perturb-seeds 1 --test-gauss 1e-3"
t tiny_prec32 $X tailexp --channel cond --cond prec-fp32 $TINY --out "$D/tiny_prec32"
t tiny_compiled $X tailexp --channel cond --cond compiled $TINY --out "$D/tiny_compiled"
t tiny_lora $X tailexp --channel cond --cond lora-0.03 $TINY --out "$D/tiny_lora"
t tiny_w-int8 $X tailexp --channel cond --cond w-int8 $TINY --out "$D/tiny_w-int8"
SWEEP_ALT_PROMPTS=1 t tiny_alt $X tailexp --channel cond --cond attn-eager $TINY --out "$D/tiny_alt"
t diag_new $X cond --conds prec-fp32,attn-math,kv-fp8,w-int4 --n 8 --out "$D/diag_new"
find "$D" -name '*.safetensors' -delete
touch "$WORK/DRY2_DONE"
