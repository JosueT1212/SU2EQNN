# 12-block full-dataset runs (teogpu02)

Logs and as-run configs for the `num_blocks: 12` full-dataset (Kasieczka top tagging,
1.2M/0.4M/0.4M) experiments on the INFN Genova `teogpu02` node (4x NVIDIA A40).

## Contents

- `logs/*_full_b12_teogpu.log` — **completed** runs. These back the b12 results table.
- `logs/*_b12_lgatrfull_teogpu.log` — **in progress as of 2026-08-27 22:43 CEST.**
  Each holds only the step-1000 validation point. They are NOT finished runs and are
  NOT runs that stopped at step 1000.
- `logs/*_full_b12_lgatrcfg_teogpu.log` — runs killed by an undetermined external
  signal on 2026-08-17/18, each ~30 s after its step-1000 checkpoint write. Cause never
  established; node rebooted and kernel logs are gone. Kept as a record of the incident.
- `logs/watchdog_lgatrfull_*.log` — per-600 s `nvidia-smi` + `ps` snapshots for the
  in-progress runs.
- `configs_as_run/` — the exact configs used on the cluster. See caveat below.
- `launch_lgatrfull.sh` — the launcher (one run per GPU, tmux + setsid, idempotent).

## Completed b12 results (test AUC / accuracy)

| run | AUC | acc |
|-----|-----|-----|
| r1 baseline    | 0.9859 | 0.9400 |
| r7 patch       | 0.9859 | 0.9399 |
| r8a            | 0.9858 | 0.9396 |
| r8b            | 0.9858 | 0.9395 |
| r5b            | 0.9855 | 0.9392 |
| r4             | 0.9854 | 0.9392 |
| r5a            | 0.9854 | 0.9392 |
| r5d            | 0.9845 | 0.9371 |

Single seed each. The 0.9845–0.9859 spread is far wider than L-GATr's reported
±0.0001 seed error, so ordering within this table is not resolved.

## Caveat: `configs_as_run/` is a record, not a runnable config set

These four configs were run against an **older fork** of `experiments/top_tagging/`
that predates the unified embedding registry. `top_tagging_r8{a,b}_*.yaml` set
`embedding: r8`, which is not a layout name on `main` (valid names:
`multi_channel`, `single_channel`, `invariant_subjet_axis`, `patch`). They are stored
here verbatim to document what was executed, and are deliberately not placed in
`config/`. Porting them to the registry is unfinished work.

## Caveat: wall-clock in these logs is thermally throttled

Running four jobs concurrently pins every A40 at ~88 C with
`sw_thermal_slowdown = Active`. Measured mid-run:

| gpu | sm clock | % of max | power | job |
|-----|----------|----------|-------|-----|
| 0 | 435 MHz | 25% | 150/300 W | r1 |
| 1 | 975 MHz | 56% | 179/300 W | r7_patch |
| 2 | 1155 MHz | 66% | 202/300 W | r8a |
| 3 | 555 MHz | 32% | 174/300 W | r8b |

Power sits at 50–67% of limit, so this is cooling-limited, not power-limited. First
1000-step+validation cycle took 55.8 min on GPU2 but 103.4 min on GPU0 — a 1.85x
spread from throttling alone, despite r1 having strictly less compute than r7_patch.
Do not read per-run timings here as architecture differences. See the watchdog log.
