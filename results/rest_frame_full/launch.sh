#!/bin/bash
# Launch the two rest-frame runs (beams none / boosted) on teogpu02, one A40 each, each in
# its own tmux session so a dropped ssh cannot take them down. Same pattern as
# results/b12_full/launch_lgatrfull.sh. All three run from ONE main commit.
# Idempotent: refuses to double-launch a session that already exists.
#
# Prerequisites on teogpu02 (checked below, the script refuses to launch otherwise):
#   - $TREE is a clean checkout of main containing the rest-frame configs.
#     A FRESH clone, not ~/SU2EQNN: that tree is the patched older fork behind
#     the b12 logs and must stay as it is for provenance.
#   - $TREE/data/toptagging_full.npz is the complete raw file (no aux keys),
#     and every jet has a timelike total momentum (frame: rest raises otherwise).
# GPU0 is left idle: it throttles to ~25% clock under load (teogpu02 thermal).
TREE=/home/arivero/SU2EQNN-main
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$TREE" || exit 1
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ABORT: $TREE has uncommitted changes"; exit 1
fi
COMMIT=$(git rev-parse --short HEAD)
$PY - <<'PY' || { echo "ABORT: data/toptagging_full.npz missing, not raw, or has non-timelike jets"; exit 1; }
import numpy as np
f = np.load("data/toptagging_full.npz")
need = {f"{k}_{m}" for k in ("kinematics", "labels") for m in ("train", "val", "test")}
assert set(f.files) == need, sorted(f.files)
bad = 0
for mode in ("train", "val", "test"):
    kin = f[f"kinematics_{mode}"]
    n_bad, min_m, max_gamma = 0, np.inf, 0.0
    for i in range(0, len(kin), 100_000):          # chunks: float64 copy of 1.2M jets is ~8 GB
        P = kin[i:i + 100_000].astype(np.float64).sum(axis=1)   # padding rows are zero
        m2 = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1)
        n_bad += int((m2 <= 0).sum())
        ok = m2 > 0
        m = np.sqrt(m2[ok])
        min_m = min(min_m, m.min())
        max_gamma = max(max_gamma, (P[ok, 0] / m).max())
    print(f"{mode}: {len(kin)} jets, non-timelike {n_bad}, min m {min_m:.3f} GeV, max gamma {max_gamma:.1f}")
    bad += n_bad
assert bad == 0, f"{bad} non-timelike jets: frame: rest would raise mid-run"
PY

launch () {  # name config gpu
  local name=$1 cfg=$2 gpu=$3
  local log=/home/arivero/${name}_full_b12_teogpu.log
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "SKIP $name (tmux session already exists)"; return
  fi
  tmux new-session -d -s "$name" \
    "cd $TREE && echo \"COMMIT=$COMMIT\" > $log && \
     CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$TREE \
     setsid $PY -m experiments.top_tagging.experiment --config-name=$cfg \
     >> $log 2>&1; \
     echo \"EXIT_STATUS=\$? at \$(date -Is)\" >> $log; \
     sleep 604800"
  echo "LAUNCHED $name gpu=$gpu commit=$COMMIT -> $log"
}

launch rest_none        top_tagging_rest_frame_full_b12                2
launch rest_boosted     top_tagging_rest_frame_boosted_beams_full_b12  3

if ! tmux has-session -t watchdog_rest 2>/dev/null; then
  tmux new-session -d -s watchdog_rest \
    "while true; do
       { echo \"### \$(date -Is)\";
         nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm --format=csv,noheader;
         ps -u arivero -o pid,etime,rss,stat,cmd --sort=-rss | grep -E \"experiment\" | head -5;
         dmesg -T 2>/dev/null | tail -3;
       } >> /home/arivero/watchdog_rest_${STAMP}.log 2>&1;
       sleep 600;
     done"
  echo "LAUNCHED watchdog -> /home/arivero/watchdog_rest_${STAMP}.log"
fi

sleep 3
echo "=== SESSIONS ==="; tmux ls 2>&1
