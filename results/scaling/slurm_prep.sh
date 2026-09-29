#!/bin/bash
#SBATCH -J scaling_prep
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /home/arivero/official_paper_logs/slurm/scaling_prep_%j.out
# Build the nested training subsets on teogpu02's local disk and link them into the tree. Idempotent.
set -e
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
TREE=/home/arivero/SU2EQNN-scaling  # own clone: pulling here never moves the sweep tree
cd "$TREE"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) $(date -Is)"
[ -f /var/tmp/arivero/toptagging_full_aux.npz ] || { echo "aux dataset missing: run seed_sweep prep first"; exit 1; }
df -h /var/tmp
free_gb=$(df -Pk /var/tmp | awk 'NR==2 {print int($4/1048576)}')
if [ ! -f /var/tmp/arivero/toptagging_1M.npz ]; then
  [ "$free_gb" -ge 45 ] || { echo "need ~35 GB in /var/tmp, have ${free_gb} GB"; exit 1; }
  $PY results/scaling/make_subsets.py   # writes *.tmp.npz then renames, so a present file is complete
fi
for tag in 10k 30k 100k 300k 1M; do
  ln -sf /var/tmp/arivero/toptagging_${tag}.npz data/toptagging_${tag}.npz
  ln -sf /var/tmp/arivero/toptagging_${tag}_aux.npz data/toptagging_${tag}_aux.npz
done
ls -la /var/tmp/arivero/ data/
