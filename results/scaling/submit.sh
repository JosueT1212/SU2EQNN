#!/bin/bash
# Submit from linuxge: prep, then the 45-task array (4 at a time) once prep succeeds.
# Pass the seed-sweep array id to queue behind it: ./submit.sh 3475805
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
mkdir -p /home/arivero/official_paper_logs/slurm
after=${1:+--dependency=afterany:$1}
n=$(grep -vc '^#' jobs.txt)
prep=$(sbatch --parsable $after slurm_prep.sh)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 --dependency=afterok:$prep slurm_task.sh)
echo "prep=$prep array=$arr tasks=$n"
