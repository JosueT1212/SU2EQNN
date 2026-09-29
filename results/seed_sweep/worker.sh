#!/bin/bash
# One worker per GPU: claims the next unclaimed job of jobs.txt and runs it.
# Usage: worker.sh GPU   (start with: setsid nohup bash worker.sh 0 >/dev/null 2>&1 &)
# A job is claimed atomically by mkdir of its claim dir; a finished job
# (EXIT_STATUS=0 in its log) is never rerun. To retry a failed job, delete its
# claim dir. Logs: $LOGS/seed_<seed>/<config>.log; checkpoints: checkpoints/seed_<seed>/<config>.pt
GPU=$1
TREE=/home/arivero/SU2EQNN-main
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
SWEEP=$TREE/results/seed_sweep
LOGS=/home/arivero/official_paper_logs
CLAIMS=$LOGS/.claims
mkdir -p "$CLAIMS"
cd "$TREE" || exit 1
COMMIT=$(git rev-parse --short HEAD)

grep -v '^#' "$SWEEP/jobs.txt" | while read -r cfg seed; do
  [ -z "$cfg" ] && continue
  name=${cfg}_s${seed}
  mkdir -p "$LOGS/seed_$seed" "checkpoints/seed_$seed"
  log=$LOGS/seed_$seed/$cfg.log
  grep -q "^EXIT_STATUS=0" "$log" 2>/dev/null && continue
  mkdir "$CLAIMS/$name" 2>/dev/null || continue          # another worker has it
  echo "COMMIT=$COMMIT GPU=$GPU HOST=$(hostname) START=$(date -Is)" > "$log"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$TREE $PY -m experiments.top_tagging.experiment \
    --config-name="$cfg" training.seed="$seed" \
    training.checkpoint_path="checkpoints/seed_${seed}/${cfg}.pt" \
    hydra.run.dir="outputs/official_paper/seed_${seed}/${cfg}" >> "$log" 2>&1 < /dev/null
  echo "EXIT_STATUS=$? at $(date -Is)" >> "$log"
done
echo "worker GPU=$GPU: no jobs left at $(date -Is)" >> "$LOGS/workers.log"
