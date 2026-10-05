#!/bin/bash
# Submit from linuxge: the 150 patch-version runs (4 at a time). No prep: they read the naive
# _aux and little-group _durham files built by the earlier preps. Pass the prep job ids still
# pending (the 10k Durham subset comes from 3492218):
#   ./submit_patch.sh 3492218
set -e
cd /home/arivero/SU2EQNN-scaling/results/scaling
deps=$(IFS=:; echo "$*")
after=${deps:+--dependency=afterok:$deps}
n=$(grep -vc '^#' jobs_patch.txt)
arr=$(sbatch --parsable --array=0-$((n - 1))%4 $after \
  --export=ALL,JOBS=results/scaling/jobs_patch.txt slurm_task.sh)
echo "array=$arr tasks=$n"
