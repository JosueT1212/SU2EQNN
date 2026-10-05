#!/bin/bash
# Submit from linuxge: the little-group 10k point. Prep (idempotent: only builds the 10k Durham
# subset and its link) after the given prep job, then 24 tasks (4 at a time).
# Usage: ./submit_little_10k.sh <little_prep job id>
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
after=${1:+--dependency=afterok:$1}
n=$(grep -vc '^#' jobs_little_10k.txt)
prep=$(sbatch --parsable $after slurm_prep_little.sh)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 --dependency=afterok:$prep \
  --export=ALL,JOBS=results/scaling/jobs_little_10k.txt slurm_task.sh)
echo "prep=$prep array=$arr tasks=$n"
