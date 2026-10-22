#!/bin/bash
# Submit from linuxge: the 96 naive xy-plane subset runs (4 at a time). No prep: they read the
# toptagging_<size>[_aux].npz files built by the earlier preps. Pass any prep job ids still pending.
#   ./submit_naive_xyplane_subsets.sh
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
deps=$(IFS=:; echo "$*")
after=${deps:+--dependency=afterok:$deps}
n=$(grep -vc '^#' jobs_naive_xyplane_subsets.txt)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 $after \
  --export=ALL,JOBS=results/scaling/jobs_naive_xyplane_subsets.txt slurm_task.sh)
echo "array=$arr tasks=$n"
