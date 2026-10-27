#!/usr/bin/env bash
# Poll every instance in instances.tsv: pull results, including trained tails (about 0.9 GB per run), and when
# its queue is done pull once more, terminate it, and record it. Exits when every instance is finished or after
# <max_minutes>. Tolerates network drops: an unreachable instance is simply retried on the next pass.
set -uo pipefail
export AWS_PAGER=""
HERE=$(cd "$(dirname "$0")" && pwd); MAXMIN=${1:-900}; R=us-east-1; START=$(date +%s)
KEY="$HOME/.ssh/cs2881r.pem"; KH="$HERE/.known_hosts"
SSH="ssh -i $KEY -o UserKnownHostsFile=$KH -o ConnectTimeout=10 -o BatchMode=yes"
PULL() { rsync -az --partial --timeout=120 -e "$SSH" --exclude '*.pt' --exclude 'capture*.safetensors' ubuntu@"$1":sweep/out/ "$HERE/results/$2/" </dev/null 2>/dev/null; }
touch "$HERE/finished.tsv"
while :; do
  left=0
  while IFS=$'\t' read -r name iid ip t0 <&3; do
    [[ -z $iid ]] && continue
    grep -q "	$iid	" "$HERE/finished.tsv" && continue
    left=$((left+1)); mkdir -p "$HERE/results/$name"
    PULL "$ip" "$name"
    $SSH ubuntu@"$ip" "cat ~/sweep/setup.log ~/sweep/queue.log 2>/dev/null | tail -c 20000" </dev/null > "$HERE/results/$name/remote_tail.log" 2>/dev/null
    if $SSH ubuntu@"$ip" "test -f ~/sweep/DONE" </dev/null 2>/dev/null; then
      if PULL "$ip" "$name"; then
        aws ec2 terminate-instances --region $R --instance-ids "$iid" --query 'TerminatingInstances[0].CurrentState.Name' --output text \
          && printf '%s\t%s\tdone\t%s\n' "$name" "$iid" "$(date -u +%FT%TZ)" >> "$HERE/finished.tsv"
      fi
    else
      st=$(aws ec2 describe-instances --region $R --instance-ids "$iid" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
      case "$st" in
        terminated|shutting-down) printf '%s\t%s\tgone-%s\t%s\n' "$name" "$iid" "$st" "$(date -u +%FT%TZ)" >> "$HERE/finished.tsv" ;;
        stopped|stopping) printf '%s\t%s\tSTOPPED-by-timer-results-on-disk\t%s\n' "$name" "$iid" "$(date -u +%FT%TZ)" >> "$HERE/finished.tsv" ;;
      esac
    fi
  done 3< "$HERE/instances.tsv"
  [[ $left -eq 0 ]] && { echo "all instances finished"; break; }
  [[ $(( ($(date +%s) - START) / 60 )) -ge $MAXMIN ]] && { echo "watch timeout with $left instances left"; break; }
  sleep 240
done
echo "--- finished.tsv ---"; cat "$HERE/finished.tsv"
echo "--- project instances not terminated ---"
aws ec2 describe-instances --region $R --filters Name=tag:project,Values=tvd-sweep Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags[?Key==`Name`]|[0].Value]' --output text
echo "(end)"
