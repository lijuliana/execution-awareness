#!/usr/bin/env bash
# Same as watch.sh for the GCP instances in gcp_instances.tsv: pull results, delete each VM when its queue is done.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); MAXMIN=${1:-900}; START=$(date +%s)
KEY="$HOME/.ssh/google_compute_engine"; KH="$HERE/.known_hosts_gcp"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -o ConnectTimeout=10 -o BatchMode=yes"
PULL() { rsync -az --partial --timeout=120 -e "$SSH" --exclude '*.pt' --exclude 'capture*.safetensors' "$1":sweep/out/ "$HERE/results/$2/" </dev/null 2>/dev/null; }
touch "$HERE/gcp_finished.tsv"
while :; do
  left=0
  while IFS=$'\t' read -r name vm ip t0 zone user <&3; do
    [[ -z $vm ]] && continue
    grep -q "	$vm	" "$HERE/gcp_finished.tsv" && continue
    left=$((left+1)); mkdir -p "$HERE/results/$name"
    PULL "$user@$ip" "$name"
    $SSH "$user@$ip" "cat ~/sweep/setup.log ~/sweep/queue.log 2>/dev/null | tail -c 20000" </dev/null > "$HERE/results/$name/remote_tail.log" 2>/dev/null
    if $SSH "$user@$ip" "test -f ~/sweep/DONE" </dev/null 2>/dev/null; then
      if PULL "$user@$ip" "$name"; then
        gcloud compute instances delete "$vm" --zone "$zone" --quiet >/dev/null 2>&1 \
          && printf '%s\t%s\tdone\t%s\n' "$name" "$vm" "$(date -u +%FT%TZ)" >> "$HERE/gcp_finished.tsv"
      fi
    else
      st=$(gcloud compute instances describe "$vm" --zone "$zone" --format='value(status)' 2>/dev/null)
      case "$st" in
        "") printf '%s\t%s\tgone\t%s\n' "$name" "$vm" "$(date -u +%FT%TZ)" >> "$HERE/gcp_finished.tsv" ;;
        TERMINATED|STOPPING|SUSPENDED) printf '%s\t%s\tSTOPPED-by-timer-results-on-disk\t%s\n' "$name" "$vm" "$(date -u +%FT%TZ)" >> "$HERE/gcp_finished.tsv" ;;
      esac
    fi
  done 3< "$HERE/gcp_instances.tsv"
  [[ $left -eq 0 ]] && { echo "all GCP instances finished"; break; }
  [[ $(( ($(date +%s) - START) / 60 )) -ge $MAXMIN ]] && { echo "watch timeout with $left GCP instances left"; break; }
  sleep 240
done
echo "--- gcp_finished.tsv ---"; cat "$HERE/gcp_finished.tsv"
echo "--- GCP instances still existing ---"; gcloud compute instances list --format='value(name,zone.basename(),status)'; echo "(end)"
