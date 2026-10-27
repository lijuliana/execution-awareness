#!/usr/bin/env bash
# Run named jobs one after another on this instance. Each job is "name|kind|args":
#   kind tailexp / cond : python extra_cond.py <kind> <args> --out $OUT/<name>   (from sidechan/)
#   kind rate           : the buffer-hook and re-compilation rate job (watching, pinned commit)
# Writes $OUT/status.jsonl and touches $WORK/DONE at the end, whatever happened.
set -uo pipefail
WORK=$HOME/sweep; OUT=$WORK/out; REV=b968826d9c46dd6066d109eabc6255188de91218
export PATH="$WORK/venv/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export SIDECHAN_HF_ROOT=$HOME/.cache/huggingface/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
status() { echo "{\"job\":\"$1\",\"exit\":$2,\"seconds\":$3,\"ended\":\"$(date -u +%FT%TZ)\"}" >> "$OUT/status.jsonl"; }
rate_job() {
  local o=$OUT/rate; mkdir -p "$o"; cd "$WORK"
  local i
  for i in 1 2 3 4 5 6; do   # default compilation: how often does the buffer hook change the logits?
    TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) python -m watching.validate_tail --checkpoint Qwen/Qwen3-8B --revision "$REV" \
      --tail artifacts/sft-034/native_tail.pt --data src/data/validation_v2/dataset.json \
      --out "$o/validate_default_$i" --split test --max-prompts 64 --length 96 > "$o/validate_default_$i.log" 2>&1
    echo "validate_default_$i exit $?" >> "$o/exits.txt"
  done
  for i in 1 2 3 4; do       # Inductor deterministic mode
    TORCHINDUCTOR_DETERMINISTIC=1 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) python -m watching.validate_tail --checkpoint Qwen/Qwen3-8B --revision "$REV" \
      --tail artifacts/sft-034/native_tail.pt --data src/data/validation_v2/dataset.json \
      --out "$o/validate_determ_$i" --split test --max-prompts 64 --length 96 > "$o/validate_determ_$i.log" 2>&1
    echo "validate_determ_$i exit $?" >> "$o/exits.txt"
  done
  local mode flag ext
  for i in 1 2 3; do         # compile control: two unprobed compilations + reference hook; extended adds buffer copy,
    for mode in default determ; do   # an opaque materialization, and materialization + retained reference
      flag=""; [[ $mode == determ ]] && flag="--deterministic"
      for ext in plain ext; do
        local e=""; [[ $ext == ext ]] && e="--extended-controls"
        TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) python -m watching.compile_control --checkpoint Qwen/Qwen3-8B --revision "$REV" \
          --data src/data/validation_v2/dataset.json --out "$o/cc_${ext}_${mode}_$i" --prompts 4 --probe-layer 14 --length 96 $flag $e > "$o/cc_${ext}_${mode}_$i.log" 2>&1
        echo "cc_${ext}_${mode}_$i exit $?" >> "$o/exits.txt"
      done
    done
  done
  find "$o" -name '*.pt' -delete; find "$o" -name '*.safetensors' -delete
  ! grep -qv "exit 0" "$o/exits.txt"
}
for spec in "$@"; do
  [[ -z ${spec// /} || $spec == \#* ]] && continue
  IFS='|' read -r name kind args <<< "$spec"
  case "$kind" in rate|tailexp|cond) ;; *) echo "bad job spec: $spec" | tee -a "$OUT/logs/queue.log"; status "${name:-bad}" 98 0; continue ;; esac
  [[ -e $WORK/skip/$name ]] && { echo "[$(date -u +%FT%TZ)] skip $name (reassigned)" | tee -a "$OUT/logs/queue.log"; continue; }
  t0=$(date +%s); mkdir -p "$OUT/$name"
  echo "[$(date -u +%FT%TZ)] start $name" | tee -a "$OUT/logs/queue.log"
  if [[ $kind == rate ]]; then
    rate_job; rc=$?
  else
    ( cd "$WORK/sc/sidechan" && PYTHONPATH="$WORK/sc/sidechan:$WORK/code" python "$WORK/code/extra_cond.py" "$kind" $args --out "$OUT/$name" ) > "$OUT/logs/$name.log" 2>&1; rc=$?
  fi
  status "$name" "$rc" "$(( $(date +%s) - t0 ))"
  echo "[$(date -u +%FT%TZ)] end $name exit $rc" | tee -a "$OUT/logs/queue.log"
done
touch "$WORK/DONE"
