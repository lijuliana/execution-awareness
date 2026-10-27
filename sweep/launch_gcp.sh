#!/usr/bin/env bash
# usage: launch_gcp.sh <name> <max_hours> <jobs-file> [zone]
# One a2-ultragpu-1g (A100-80GB) in us-central1. GCP stops the VM itself after <max_hours> (disk kept).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); NAME=$1; H=$2; JOBS=$3; ZONE=${4:-us-central1-a}; VM=sweep-$(echo "$NAME" | tr 'A-Z' 'a-z')
MT=${MTYPE:-a2-ultragpu-1g}; SSDFLAG=""; [[ $MT == a2-ultragpu* ]] && SSDFLAG="--discard-local-ssds-at-termination-timestamp=true"
gcloud compute instances create "$VM" --zone "$ZONE" --machine-type "$MT" --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 150GB --boot-disk-type pd-ssd --metadata install-nvidia-driver=True \
  --max-run-duration "${H}h" --instance-termination-action STOP $SSDFLAG --labels project=tvd-sweep --quiet >/dev/null 2>"$HERE/.gcp_$NAME.err" \
  || { echo "$NAME: create failed in $ZONE: $(tail -2 "$HERE/.gcp_$NAME.err" | tr '\n' ' ' | cut -c1-220)"; exit 1; }
fail() { echo "$NAME: $1; deleting $VM"; gcloud compute instances delete "$VM" --zone "$ZONE" --quiet >/dev/null 2>&1; exit 2; }
ok=0; for i in $(seq 1 40); do gcloud compute ssh "$VM" --zone "$ZONE" --command true --quiet </dev/null >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[[ $ok == 1 ]] || fail "ssh never came up"
gcloud compute ssh "$VM" --zone "$ZONE" --command "mkdir -p ~/sweep/code" --quiet </dev/null >/dev/null 2>&1 || fail "mkdir failed"
gcloud compute scp "$HERE"/extra_cond.py "$HERE"/remote_setup.sh "$HERE"/remote_queue.sh "$HERE"/remote_start.sh "$VM":sweep/code/ --zone "$ZONE" --quiet >/dev/null 2>&1 || fail "scp code failed"
gcloud compute scp "$JOBS" "$VM":sweep/code/jobs.txt --zone "$ZONE" --quiet >/dev/null 2>&1 || fail "scp jobs failed"
U=$(gcloud compute ssh "$VM" --zone "$ZONE" --command "bash ~/sweep/code/remote_start.sh run >/dev/null; whoami" --quiet </dev/null 2>/dev/null | tail -1) || fail "start failed"
IP=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "gcp-$NAME" "$VM" "$IP" "$(date -u +%FT%TZ)" "$ZONE" "$U" >> "$HERE/gcp_instances.tsv"
echo "$NAME launched: $VM $ZONE $IP user $U"
