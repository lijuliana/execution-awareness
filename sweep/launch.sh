#!/usr/bin/env bash
# usage: launch.sh <name> <backstop_minutes> <jobs-file | setup-only>
# Launches one g6e.2xlarge (one L40S), copies the code, and starts it.
# The hard shutdown timer is set by the instance itself at boot (user data), before any SSH step, and a
# shutdown STOPS the instance (disk kept) so results survive if nothing has pulled them yet.
# watch.sh terminates instances after pulling; anything left stopped is terminated in the final sweep.
set -uo pipefail
export AWS_PAGER=""
HERE=$(cd "$(dirname "$0")" && pwd); NAME=$1; BACKSTOP=$2; JOBS=$3; R=us-east-1
KEY="$HOME/.ssh/cs2881r.pem"; KH="$HERE/.known_hosts"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -o ConnectTimeout=10 -o BatchMode=yes"
UD=$(mktemp); printf '#!/bin/bash\nshutdown -h +%s\n' "$BACKSTOP" > "$UD"
IID=$(aws ec2 run-instances --region $R --image-id ami-012ba162b9cd2729c --instance-type "${ITYPE:-g6e.2xlarge}" --key-name cs2881r \
  ${SUBNET:+--subnet-id $SUBNET} \
  --security-group-ids sg-007ca52c2b6cedb1f --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sweep-$NAME},{Key=project,Value=tvd-sweep}]" "ResourceType=volume,Tags=[{Key=project,Value=tvd-sweep}]" \
  --query 'Instances[0].InstanceId' --output text) || { rm -f "$UD"; echo "$NAME: run-instances failed"; exit 1; }
rm -f "$UD"
fail() { echo "$NAME: $1 ($IID); terminating"; aws ec2 terminate-instances --region $R --instance-ids "$IID" --query 'TerminatingInstances[0].CurrentState.Name' --output text; exit 2; }
aws ec2 wait instance-running --region $R --instance-ids "$IID" || fail "never reached running"
IP=$(aws ec2 describe-instances --region $R --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ok=0; for i in $(seq 1 45); do $SSH ubuntu@"$IP" true </dev/null 2>/dev/null && { ok=1; break; }; sleep 8; done
[[ $ok == 1 ]] || fail "ssh never came up"
$SSH ubuntu@"$IP" "mkdir -p ~/sweep/code" </dev/null || fail "mkdir failed"
scp -q -i "$KEY" -o UserKnownHostsFile="$KH" "$HERE"/extra_cond.py "$HERE"/remote_setup.sh "$HERE"/remote_queue.sh "$HERE"/remote_start.sh ubuntu@"$IP":sweep/code/ || fail "scp code failed"
if [[ $JOBS == setup-only ]]; then
  $SSH ubuntu@"$IP" "bash ~/sweep/code/remote_start.sh setup-only" </dev/null || fail "start failed"
else
  scp -q -i "$KEY" -o UserKnownHostsFile="$KH" "$JOBS" ubuntu@"$IP":sweep/code/jobs.txt || fail "scp jobs failed"
  $SSH ubuntu@"$IP" "bash ~/sweep/code/remote_start.sh run" </dev/null || fail "start failed"
fi
printf '%s\t%s\t%s\t%s\n' "$NAME" "$IID" "$IP" "$(date -u +%FT%TZ)" >> "$HERE/instances.tsv"
echo "$NAME launched: $IID $IP"
