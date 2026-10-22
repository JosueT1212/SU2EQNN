#!/bin/bash
# Submit from linuxge: the 26 baseline patch runs (4 at a time). No prep: they read the plain
# toptagging_<size>.npz files built by the earlier preps. Pass any prep job ids still pending.
#   ./submit_patch_baseline.sh
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
deps=$(IFS=:; echo "$*")
after=${deps:+--dependency=afterok:$deps}
n=$(grep -vc '^#' jobs_patch_baseline.txt)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 $after \
  --export=ALL,JOBS=results/scaling/jobs_patch_baseline.txt slurm_task.sh)
echo "array=$arr tasks=$n"
