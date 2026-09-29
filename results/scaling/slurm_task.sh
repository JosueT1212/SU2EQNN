#!/bin/bash
#SBATCH -J scaling
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --gres=gpu:teslaga102:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /home/arivero/official_paper_logs/slurm/scaling_%A_%a.out
# One array task = one line of jobs.txt (config size seed). Same aligned *_full_b12
# config as the seed sweep; only the training set changes. 21 epochs at every size,
# cosine over the whole run, 5 validations. Done tasks (EXIT_STATUS=0) are skipped.
TREE=/home/arivero/SU2EQNN-scaling  # own clone: pulling here never moves the sweep tree
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
LOGS=/home/arivero/official_paper_logs/scaling
cd "$TREE" || exit 1
read -r cfg tag seed < <(grep -v '^#' results/scaling/jobs.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$cfg" ] && { echo "no job for index $SLURM_ARRAY_TASK_ID"; exit 1; }
case $tag in 10k) n=10000;; 30k) n=30000;; 100k) n=100000;; 300k) n=300000;; 1M) n=1000000;; *) exit 1;; esac
# baseline reads kinematics only; with aux keys multi_channel would add the axis channels
[ "$cfg" = top_tagging_r1_full_b12 ] && data=data/toptagging_${tag}.npz || data=data/toptagging_${tag}_aux.npz
epochs=21; bs=128
total=$(( epochs * ((n + bs - 1) / bs) ))   # loader keeps the last partial batch
val_every=$(( total / 5 ))
run=${cfg}_${tag}_seed${seed}
mkdir -p "$LOGS" checkpoints/scaling
log=$LOGS/$run.log
if grep -q "^EXIT_STATUS=0" "$log" 2>/dev/null; then echo "already done: $log"; exit 0; fi
echo "COMMIT=$(git rev-parse --short HEAD) JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES N=$n STEPS=$total VAL_EVERY=$val_every START=$(date -Is)" > "$log"
PYTHONPATH=$TREE $PY -m experiments.top_tagging.experiment \
  --config-name="$cfg" training.seed="$seed" data.path="$data" \
  training.epochs=$epochs training.batch_size=$bs \
  training.scheduler_t0_steps=$total training.validate_every_n_steps=$val_every \
  training.checkpoint_path="checkpoints/scaling/${run}.pt" \
  hydra.run.dir="outputs/scaling/${run}" >> "$log" 2>&1 < /dev/null
status=$?
echo "EXIT_STATUS=$status at $(date -Is)" >> "$log"
exit $status
