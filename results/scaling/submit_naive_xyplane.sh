#!/bin/bash
# Submit from linuxge: the 8 naive xy-plane full runs (official, seed_0 layout). No prep:
# they read toptagging_full.npz / toptagging_full_aux.npz, already on teogpu02.
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
n=$(grep -vc '^#' jobs_naive_xyplane.txt)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 \
  --export=ALL,JOBS=results/scaling/jobs_naive_xyplane.txt slurm_task.sh)
echo "array=$arr tasks=$n"
