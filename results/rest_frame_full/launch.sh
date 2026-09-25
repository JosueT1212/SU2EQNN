#!/bin/bash
# Launch the three rest-frame comparison runs on teogpu02, one per A40, each in
# its own tmux session so a dropped ssh cannot take them down. Same pattern as
# results/b12_full/launch_lgatrfull.sh. All three run from ONE main commit.
# Idempotent: refuses to double-launch a session that already exists.
#
# Prerequisites on teogpu02 (checked below, the script refuses to launch otherwise):
#   - $TREE is a clean checkout of main containing the rest-frame configs
#   - $TREE/data/toptagging_full.npz is the complete raw file (no aux keys)
TREE=/home/arivero/SU2EQNN
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$TREE" || exit 1
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ABORT: $TREE has uncommitted changes"; exit 1
fi
COMMIT=$(git rev-parse --short HEAD)
$PY - <<'PY' || { echo "ABORT: data/toptagging_full.npz is not the raw full file"; exit 1; }
import numpy as np
f = np.load("data/toptagging_full.npz")
need = {f"{k}_{m}" for k in ("kinematics", "labels") for m in ("train", "val", "test")}
assert set(f.files) == need, sorted(f.files)
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

launch r1_main          top_tagging_r1_full_b12                        0
launch rest_none        top_tagging_rest_frame_full_b12                1
launch rest_boosted     top_tagging_rest_frame_boosted_beams_full_b12  2

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
