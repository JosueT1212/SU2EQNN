#!/bin/bash
# Submit from linuxge: the 26 seed + depth runs (4 at a time). No prep: they read the plain full
# toptagging_full.npz. Pass any job ids that must finish first.
#   ./submit_seeds_depth.sh
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
deps=$(IFS=:; echo "$*")
after=${deps:+--dependency=afterok:$deps}
n=$(grep -vc '^#' jobs_seeds_depth.txt)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 $after \
  --export=ALL,JOBS=results/scaling/jobs_seeds_depth.txt slurm_task.sh)
echo "array=$arr tasks=$n"
