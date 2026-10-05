#!/bin/bash
#SBATCH -J little_prep
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o /home/arivero/official_paper_logs/slurm/little_prep_%j.out
# Durham labels for the full file, nested little-group subsets, data/ links (incl. the naive
# full aux file for the naive full-set partners). Idempotent.
# Needs the naive scaling prep (toptagging_full.npz and the toptagging_<tag>.npz subsets).
set -e
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
TREE=/home/arivero/SU2EQNN-scaling
V=/var/tmp/arivero
cd "$TREE"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) $(date -Is)"
[ -f $V/toptagging_full.npz ] && [ -f $V/toptagging_1M.npz ] && [ -f $V/toptagging_full_aux.npz ] || { echo "naive scaling prep missing"; exit 1; }
free_gb=$(df -Pk /var/tmp | awk 'NR==2 {print int($4/1048576)}')
if [ ! -f $V/toptagging_full_durham.npz ]; then
  [ "$free_gb" -ge 30 ] || { echo "need ~25 GB in /var/tmp, have ${free_gb} GB"; exit 1; }
  PYTHONPATH=$TREE $PY scripts/precompute_features.py --features durham \
    $V/toptagging_full.npz $V/toptagging_full_durham.tmp.npz
  mv $V/toptagging_full_durham.tmp.npz $V/toptagging_full_durham.npz
fi
[ -f $V/toptagging_1M_durham.npz ] || (cd results/scaling && $PY make_durham_subsets.py)
mkdir -p data
ln -sf $V/toptagging_full.npz data/toptagging_full.npz
ln -sf $V/toptagging_full_aux.npz data/toptagging_full_aux.npz
for tag in 30k 100k 300k 1M full; do
  ln -sf $V/toptagging_${tag}_durham.npz data/toptagging_${tag}_durham.npz
done
ls -la $V/ data/
