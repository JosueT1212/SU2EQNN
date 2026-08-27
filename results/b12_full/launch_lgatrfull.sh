#!/bin/bash
# Launch the four b12 lgatrfull runs on teogpu02, one per A40, each in its own
# tmux session so a dropped ssh cannot take them down.
# Idempotent: refuses to double-launch a session that already exists.
PY=/home/arivero/SU2EQNN-r8/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

launch () {  # name tree config gpu
  local name=$1 tree=$2 cfg=$3 gpu=$4
  local log=/home/arivero/${name}_b12_lgatrfull_teogpu.log
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "SKIP $name (tmux session already exists)"; return
  fi
  tmux new-session -d -s "$name" \
    "cd /home/arivero/$tree && \
     CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/home/arivero/$tree \
     setsid $PY -m experiments.top_tagging.experiment --config-name=$cfg \
     > $log 2>&1; \
     echo \"EXIT_STATUS=\$? at \$(date -Is)\" >> $log; \
     sleep 604800"
  echo "LAUNCHED $name gpu=$gpu -> $log"
}

launch r1        SU2EQNN     top_tagging_r1_full_b12_lgatrfull       0
launch r7_patch  SU2EQNN     top_tagging_r7_patch_full_b12_lgatrfull 1
launch r8a       SU2EQNN-r8  top_tagging_r8a_full_b12_lgatrfull      2
launch r8b       SU2EQNN-r8  top_tagging_r8b_full_b12_lgatrfull      3

if ! tmux has-session -t watchdog 2>/dev/null; then
  tmux new-session -d -s watchdog \
    "while true; do
       { echo \"### \$(date -Is)\";
         nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader;
         ps -u arivero -o pid,etime,rss,stat,cmd --sort=-rss | grep -E \"experiment\" | head -5;
         dmesg -T 2>/dev/null | tail -3;
       } >> /home/arivero/watchdog_lgatrfull_${STAMP}.log 2>&1;
       sleep 600;
     done"
  echo "LAUNCHED watchdog -> /home/arivero/watchdog_lgatrfull_${STAMP}.log"
fi

sleep 3
echo "=== SESSIONS ==="; tmux ls 2>&1
