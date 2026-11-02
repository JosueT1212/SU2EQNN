#!/bin/bash
#SBATCH -J scaling
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --gres=gpu:teslaga102:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /home/arivero/official_paper_logs/slurm/scaling_%A_%a.out
# One array task = one line of $JOBS (default jobs.txt; config size, tag 10k..1M or full, seed, optional
# depth = model.num_blocks, default 12). Same aligned *_full_b12
# config as the seed sweep; only the training set changes. 21 epochs at every size,
# cosine over the whole run, 5 validations. Tag `full` runs as a full-b12 run instead
# (seed-sweep log layout and protocol, see below). Done tasks (EXIT_STATUS=0) are skipped.
TREE=/home/arivero/SU2EQNN-scaling  # own clone: pulling here never moves the sweep tree
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
LOGS=/home/arivero/official_paper_logs/scaling
cd "$TREE" || exit 1
JOBS=${JOBS:-results/scaling/jobs.txt}   # submit_little.sh passes jobs_little.txt
read -r cfg tag seed depth < <(grep -v '^#' "$JOBS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$cfg" ] && { echo "no job for index $SLURM_ARRAY_TASK_ID"; exit 1; }
depth=${depth:-12}
case $tag in 10k) n=10000;; 30k) n=30000;; 100k) n=100000;; 300k) n=300000;; 1M) n=1000000;; full) n=;; *) exit 1;; esac
# baselines read kinematics only (aux keys would add axis channels); naive variants the lab
# aux file; little-group variants the rest-frame Durham labels
case $cfg in
  top_tagging_r1_full_b12|top_tagging_r1_xyplane_full_b12|top_tagging_rest_xyplane_full_b12|\
  top_tagging_r1_patch_xyplane_full_b12|top_tagging_rest_xyplane_patch_full_b12) data=data/toptagging_${tag}.npz ;;
  top_tagging_rest_*) data=data/toptagging_${tag}_durham.npz ;;
  *) data=data/toptagging_${tag}_aux.npz ;;
esac
[ -z "$n" ] && n=$(PYTHONPATH=$TREE $PY -c "import numpy as np; print(len(np.load('$data')['labels_train']))")
epochs=21; bs=128
total=$(( epochs * ((n + bs - 1) / bs) ))   # loader keeps the last partial batch
if [ "$tag" = full ]; then
  # Full-dataset runs are full-b12 runs: same layout and protocol as the seed sweep
  # (results/seed_sweep/slurm_task.sh): full_runs/<config>.log (seed 0; other seeds get a
  # _seed<s> suffix), the config's own cosine
  # length and validation every 5000 steps. A later sweep relaunch skips them as done.
  run=$cfg
  log=/home/arivero/official_paper_logs/full_runs/$cfg.log
  [ "$seed" = 0 ] || log=/home/arivero/official_paper_logs/full_runs/${cfg}_seed${seed}.log
  ckpt=checkpoints/seed_${seed}/${cfg}.pt
  outdir=outputs/seed_${seed}/${cfg}
  schedule=()
else
  run=${cfg}_${tag}_seed${seed}
  log=$LOGS/$run.log
  ckpt=checkpoints/scaling/${run}.pt
  outdir=outputs/scaling/${run}
  schedule=(training.scheduler_t0_steps=$total training.validate_every_n_steps=$(( total / 5 )))
fi
# Depth ablation: same config and data, only model.num_blocks differs. The _b<depth> suffix keeps
# the run off the official b12 log/checkpoint, and collect.py's name pattern skips it.
if [ "$depth" != 12 ]; then
  log=${log%.log}_b${depth}.log; ckpt=${ckpt%.pt}_b${depth}.pt; outdir=${outdir}_b${depth}
  schedule+=(model.num_blocks=$depth)
fi
mkdir -p "$(dirname "$log")" "$(dirname "$ckpt")"
if grep -q "^EXIT_STATUS=0" "$log" 2>/dev/null; then echo "already done: $log"; exit 0; fi
echo "COMMIT=$(git rev-parse --short HEAD) JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES N=$n STEPS=$total TAG=$tag DEPTH=$depth START=$(date -Is)" > "$log"
PYTHONPATH=$TREE $PY -m experiments.top_tagging.experiment \
  --config-name="$cfg" training.seed="$seed" data.path="$data" \
  training.epochs=$epochs training.batch_size=$bs "${schedule[@]}" \
  training.checkpoint_path="$ckpt" \
  hydra.run.dir="$outdir" >> "$log" 2>&1 < /dev/null
status=$?
echo "EXIT_STATUS=$status at $(date -Is)" >> "$log"
exit $status
