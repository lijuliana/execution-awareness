#!/usr/bin/env bash
# Dry run on the dev instance: self-test, small step-0 scan, tiny end-to-end runs for timing.
set -uo pipefail
WORK=$HOME/sweep; D=$WORK/out/dry; mkdir -p "$D"
export PATH="$WORK/venv/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export SIDECHAN_HF_ROOT=$HOME/.cache/huggingface/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$WORK/sc/sidechan"; export PYTHONPATH="$WORK/sc/sidechan:$WORK/code"
X="python $WORK/code/extra_cond.py"
t() { local n=$1; shift; local t0=$(date +%s); "$@" > "$D/$n.log" 2>&1; echo "$n exit $? seconds $(( $(date +%s) - t0 ))" >> "$D/summary.txt"; }
t selftest $X selftest
t diag_base $X cond --conds train-mode,kv-noop,w-int8,w-fp8,attn-eager,attn-math,attn-flash,attn-cudnn,tp-4,dropout-0.1,prec-fp32mm,prec-fp16mm --n 8 --out "$D/diag_base"
t diag_lora $X cond --conds lora-0.03 --n 8 --out "$D/diag_lora"
t diag_compiled $X cond --conds compiled,compiled-epc --n 8 --out "$D/diag_compiled"
TINY="--n-train 48 --n-dev 8 --n-test 16 --steps 16 --eval-every 8 --test-perturb-seeds 1 --test-gauss 1e-3"
t tiny_w-int8 $X tailexp --channel cond --cond w-int8 $TINY --out "$D/tiny_w-int8"
t tiny_batch $X tailexp --channel batch $TINY --out "$D/tiny_batch"
t tiny_prec $X tailexp --channel cond --cond prec-fp32mm $TINY --out "$D/tiny_prec"
t tiny_lora $X tailexp --channel cond --cond lora-0.03 $TINY --out "$D/tiny_lora"
t tiny_compiled $X tailexp --channel cond --cond compiled $TINY --out "$D/tiny_compiled"
t tiny_dropout $X tailexp --channel cond --cond dropout-0.1 --keep-ordinary 32 $TINY --out "$D/tiny_dropout"
find "$D" -name '*.safetensors' -delete
touch "$WORK/DRY_DONE"
