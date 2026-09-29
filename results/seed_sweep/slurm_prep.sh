#!/bin/bash
#SBATCH -J sweep_prep
#SBATCH -p teogpu
#SBATCH -w teogpu02
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /home/arivero/official_paper_logs/slurm/prep_%j.out
# Ensure the raw and aux datasets exist on teogpu02's local disk (home quota
# cannot hold them) and are linked into the tree. Idempotent.
set -e
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
TREE=/home/arivero/SU2EQNN-main
mkdir -p /var/tmp/arivero
cd "$TREE"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) $(date -Is)"
if [ ! -f /var/tmp/arivero/toptagging_full.npz ]; then
  echo "raw dataset missing: rebuilding"; $PY /home/arivero/make_raw_timelike.py
fi
if [ ! -f /var/tmp/arivero/toptagging_full_aux.npz ]; then
  echo "aux dataset missing: building"; $PY results/seed_sweep/make_aux_dataset.py
fi
ln -sf /var/tmp/arivero/toptagging_full.npz data/toptagging_full.npz
ln -sf /var/tmp/arivero/toptagging_full_aux.npz data/toptagging_full_aux.npz
ls -la /var/tmp/arivero/ data/
