#!/usr/bin/env bash
# usage: start_queue.sh <name>     Push the current code and jobs/<name>.txt to that instance and start its queue.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); NAME=$1
IP=$(awk -F'\t' -v n="$NAME" '$1==n{ip=$3} END{print ip}' "$HERE/instances.tsv"); [[ -n $IP ]] || { echo "$NAME: no such instance"; exit 1; }
KEY="$HOME/.ssh/cs2881r.pem"; KH="$HERE/.known_hosts"
SSH="ssh -i $KEY -o UserKnownHostsFile=$KH -o ConnectTimeout=15 -o BatchMode=yes"
$SSH ubuntu@"$IP" 'test -f ~/sweep/SETUP_OK' </dev/null || { echo "$NAME: setup not finished"; exit 2; }
scp -q -i "$KEY" -o UserKnownHostsFile="$KH" "$HERE"/extra_cond.py "$HERE"/remote_queue.sh ubuntu@"$IP":sweep/code/ || { echo "$NAME: scp failed"; exit 3; }
scp -q -i "$KEY" -o UserKnownHostsFile="$KH" "$HERE/jobs/$NAME.txt" ubuntu@"$IP":sweep/code/jobs.txt || { echo "$NAME: scp jobs failed"; exit 3; }
$SSH ubuntu@"$IP" 'rm -f ~/sweep/DONE; nohup bash -c "mapfile -t S < ~/sweep/code/jobs.txt; bash ~/sweep/code/remote_queue.sh \"\${S[@]}\" > ~/sweep/queue.log 2>&1; touch ~/sweep/DONE" >/dev/null 2>&1 & echo started' </dev/null && echo "$NAME queue started on $IP"
