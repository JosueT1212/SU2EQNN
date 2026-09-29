"""Scaling-study logs -> CSV for plot_scaling.py (model,size,seed,test_auc).

    python results/scaling/collect.py ~/official_paper_logs/scaling scaling.csv
Only finished runs (EXIT_STATUS=0 with a Test AUC line) are kept; the rest are listed.
"""
import csv
import re
import sys
from pathlib import Path

SIZES = {"10k": 10_000, "30k": 30_000, "100k": 100_000, "300k": 300_000, "1M": 1_000_000}
# model = config name between top_tagging_ and _full_b12, e.g. r1, bivector_patch
NAME = re.compile(r"top_tagging_(?P<cfg>.+)_full_b12_(?P<tag>[0-9]+[kM])_seed(?P<seed>[0-9]+)\.log$")


def collect(log_dir):
    rows, missing = [], []
    for log in sorted(Path(log_dir).glob("*.log")):
        m = NAME.match(log.name)
        if not m:
            continue
        text = log.read_text()
        auc = re.findall(r"Test AUC=([0-9.]+)", text)
        if "EXIT_STATUS=0" not in text or not auc:
            missing.append(log.name)
            continue
        rows.append((m["cfg"],SIZES[m["tag"]], int(m["seed"]), float(auc[-1])))
    return rows, missing


def main(log_dir, out):
    rows, missing = collect(log_dir)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "size", "seed", "test_auc"])
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out}; unfinished: {missing or 'none'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
