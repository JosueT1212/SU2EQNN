#!/bin/bash
# Submit from linuxge: little-group prep, then the 96-task array (4 at a time) once prep succeeds.
# Pass a running array id to queue behind it: ./submit_little.sh 3475876
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
mkdir -p /home/arivero/official_paper_logs/slurm
after=${1:+--dependency=afterany:$1}
n=$(grep -vc '^#' jobs_little.txt)
prep=$(sbatch --parsable $after slurm_prep_little.sh)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 --dependency=afterok:$prep \
  --export=ALL,JOBS=results/scaling/jobs_little.txt slurm_task.sh)
echo "prep=$prep array=$arr tasks=$n"
