#!/bin/bash
#SBATCH -J sweep
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --gres=gpu:teslaga102:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /home/arivero/official_paper_logs/slurm/task_%A_%a.out
# One array task = one line of jobs.txt (comments skipped), in file order.
# Submit: sbatch --array=0-49%4 --dependency=afterok:<prep> slurm_task.sh
# A task whose log already ends with EXIT_STATUS=0 exits immediately.
TREE=/home/arivero/SU2EQNN-main
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
LOGS=/home/arivero/official_paper_logs
cd "$TREE" || exit 1
read -r cfg seed < <(grep -v '^#' results/seed_sweep/jobs.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$cfg" ] && { echo "no job for index $SLURM_ARRAY_TASK_ID"; exit 1; }
mkdir -p "$LOGS/seed_$seed" "checkpoints/seed_$seed"
log=$LOGS/seed_$seed/$cfg.log
if grep -q "^EXIT_STATUS=0" "$log" 2>/dev/null; then echo "already done: $log"; exit 0; fi
echo "COMMIT=$(git rev-parse --short HEAD) JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES START=$(date -Is)" > "$log"
PYTHONPATH=$TREE $PY -m experiments.top_tagging.experiment \
  --config-name="$cfg" training.seed="$seed" \
  training.checkpoint_path="checkpoints/seed_${seed}/${cfg}.pt" \
  hydra.run.dir="outputs/official_paper/seed_${seed}/${cfg}" >> "$log" 2>&1 < /dev/null
status=$?
echo "EXIT_STATUS=$status at $(date -Is)" >> "$log"
exit $status
