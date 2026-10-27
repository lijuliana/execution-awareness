#!/usr/bin/env bash
# usage: remote_start.sh setup-only | run     (the shutdown timer was set at boot by user data)
mkdir -p ~/sweep
if [[ ${1:-} == setup-only ]]; then
  nohup bash ~/sweep/code/remote_setup.sh > ~/sweep/setup.log 2>&1 &
else
  nohup bash -c 'bash ~/sweep/code/remote_setup.sh > ~/sweep/setup.log 2>&1 && { mapfile -t S < ~/sweep/code/jobs.txt; bash ~/sweep/code/remote_queue.sh "${S[@]}" > ~/sweep/queue.log 2>&1; }; touch ~/sweep/DONE' >/dev/null 2>&1 &
fi
echo started
