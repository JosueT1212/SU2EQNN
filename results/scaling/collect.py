"""Scaling-study logs -> CSV for plot_scaling.py (model,size,seed,test_auc).

    python results/scaling/collect.py ~/official_paper_logs/scaling scaling.csv ~/official_paper_logs
The optional third argument is the root holding full_runs/<config>.log: the full-dataset
(tag full) points, which slurm_task.sh writes in the seed-sweep layout. Only logs whose
header says TAG=full are taken from there. Only finished runs (EXIT_STATUS=0 with a Test AUC line) are kept; the rest are listed.
"""
import csv
import re
import sys
from pathlib import Path

SIZES = {"10k": 10_000, "30k": 30_000, "100k": 100_000, "300k": 300_000, "1M": 1_000_000}
# model = config name between top_tagging_ and _full_b12, e.g. r1, bivector_patch;
# little-group configs map to <naive counterpart>_little so plots pair them up
NAME = re.compile(
    r"top_tagging_(?P<cfg>.+)_full_b12_(?P<tag>[0-9]+[kM]|full)_seed(?P<seed>[0-9]+)\.log$")
LITTLE = {
    "rest_xyplane": "r1_little",
    "rest_durham_axes": "grade2_bivectors_little",
    "rest_durham_patch": "bivector_patch_little",
    "rest_durham_masses": "subjet_mass_little",
    "rest_durham_handedness": "pseudoscalar_handedness_little",
    "rest_durham_plane_normal": "pseudoscalar_triad_little",
    "rest_durham_single_channel": "single_channel_axis_little",
    "rest_durham_single_channel_b": "single_channel_axis_b_little",
    "rest_xyplane_patch": "r1_patch_little",
    "rest_durham_masses_patch": "subjet_mass_patch_little",
    "rest_durham_handedness_patch": "pseudoscalar_handedness_patch_little",
    "rest_durham_plane_normal_patch": "pseudoscalar_triad_patch_little",
    "rest_durham_single_channel_patch": "single_channel_axis_patch_little",
    "rest_durham_single_channel_b_patch": "single_channel_axis_b_patch_little",
}


def collect(log_dir):
    rows, missing = [], []
    for log in sorted(Path(log_dir).glob("*.log")):
        m = NAME.match(log.name)
        name = m and official_full_name(m["cfg"])
        if not name:  # naive +-e3 runs are superseded by the xy-plane reruns
            continue
        text = log.read_text()
        auc = re.findall(r"Test AUC=([0-9.]+)", text)
        if "EXIT_STATUS=0" not in text or not auc:
            missing.append(log.name)
            continue
        # "full" = the whole training set: its size comes from the N= header slurm_task.sh writes
        size = SIZES[m["tag"]] if m["tag"] in SIZES else int(re.search(r"\bN=(\d+)", text)[1])
        rows.append((name, size, int(m["seed"]), float(auc[-1])))
    return rows, missing


FULL_NAME = re.compile(r"top_tagging_(?P<cfg>.+?)_full_b12(?:_seed(?P<seed>[0-9]+))?\.log$")


def official_full_name(cfg):
    """Paper name of a config (full or subset), or None if it is not an official run.

    Naive official runs use the xy-plane spurion (<v>_xyplane configs) and take the
    standard naive name <v>; naive +-e3 full runs are not official. Little group: LITTLE.
    """
    if cfg.startswith("rest_"):
        return LITTLE.get(cfg, cfg)
    return cfg.removesuffix("_xyplane") if cfg.endswith("_xyplane") else None


def collect_full(root):
    """Full-dataset points from <root>/full_runs/top_tagging_<cfg>_full_b12[_seed<s>].log (TAG=full only)."""
    rows, missing = [], []
    for log in sorted(Path(root).glob("full_runs/top_tagging_*_full_b12*.log")):
        m = FULL_NAME.match(log.name)
        text = log.read_text()
        if not m or "TAG=full" not in text.split("\n", 1)[0] or official_full_name(m["cfg"]) is None:
            continue
        auc = re.findall(r"Test AUC=([0-9.]+)", text)
        if "EXIT_STATUS=0" not in text or not auc:
            missing.append(f"{log.parent.name}/{log.name}")
            continue
        size = int(re.search(r"\bN=(\d+)", text)[1])
        seed = int(m["seed"] or 0)
        rows.append((official_full_name(m["cfg"]), size, seed, float(auc[-1])))
    return rows, missing


def main(log_dir, out, full_root=None):
    rows, missing = collect(log_dir)
    if full_root:
        full_rows, full_missing = collect_full(full_root)
        rows, missing = rows + full_rows, missing + full_missing
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "size", "seed", "test_auc"])
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out}; unfinished: {missing or 'none'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
