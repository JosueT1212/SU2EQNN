#!/bin/bash
# Submit the sweep from linuxge: prep job, then the 50-task array (4 at a time)
# once prep succeeds. teogpu02 no longer accepts direct ssh; SLURM only.
set -e
cd /home/arivero/SU2EQNN-main/results/seed_sweep
mkdir -p /home/arivero/official_paper_logs/slurm
n=$(grep -vc '^#' jobs.txt)
prep=$(sbatch --parsable slurm_prep.sh)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 --dependency=afterok:$prep slurm_task.sh)
echo "prep=$prep array=$arr tasks=$n"
