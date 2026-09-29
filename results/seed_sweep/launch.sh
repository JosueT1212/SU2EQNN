#!/bin/bash
# Start one detached worker per GPU on teogpu02 (setsid: survives ssh / tmux loss).
# Refuses to start if workers are already running.
if pgrep -f "seed_sweep/worker.sh" >/dev/null; then echo "workers already running"; exit 1; fi
cd /home/arivero/SU2EQNN-main || exit 1
for gpu in ${GPUS:-0 1 2 3}; do
  setsid nohup bash results/seed_sweep/worker.sh "$gpu" >/dev/null 2>&1 < /dev/null &
  echo "started worker GPU=$gpu"
done
