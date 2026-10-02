"""AUC vs training size, styled after Fig. 4 of Li et al., "Does Lorentz-symmetric
design boost network performance in jet physics?" (log x, mean line with std
errorbars plus a ±std band, dashed grid, legend lower right).

Input: long CSV with columns model,size,seed,test_auc (one row per training).
Four panels: naive/little frame x sencilla (inject once at input) / patch
(re-inject every block). Each panel shows every variant of its kind against the
plain baseline of the same frame. A panel with no rows says "pending".
Model names are the config names between top_tagging_ and _full_b12 (collect.py).

    python results/scaling/plot_scaling.py scaling.csv scaling.pdf
"""
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# fixed color + marker per model, so a model looks the same in every panel
STYLE = {
    "r1":                      ("Baseline",                       "tab:blue",   "o", "-"),
    "grade2_bivectors":        ("+ axis bivectors",               "tab:orange", "^", "-"),
    "subjet_mass":             ("+ subjet masses",                "tab:green",  "s", "-"),
    "pseudoscalar_triad":      ("+ triad handedness",             "tab:red",    "v", "-"),
    "pseudoscalar_handedness": ("+ local handedness",             "tab:purple", "P", "-"),
    "single_channel_axis":     ("jet axis in channel",            "tab:brown",  "X", "-"),
    "single_channel_axis_b":   ("jet axis in channel + lead ps.", "tab:pink",   "*", "-"),
    "bivector_patch":          ("+ axis bivectors, patch",        "tab:orange", "D", "--"),
}
SENCILLA = ["r1", "grade2_bivectors", "subjet_mass", "pseudoscalar_triad",
            "pseudoscalar_handedness", "single_channel_axis", "single_channel_axis_b"]
PANELS = [
    ("Naive, sencilla", "darkred", SENCILLA),
    ("Naive, patch",    "darkred", ["r1", "bivector_patch"]),
    ("Little group, sencilla", "navy", []),  # rest-frame injection not implemented yet
    ("Little group, patch",    "navy", []),
]


def load(path):
    runs = defaultdict(lambda: defaultdict(list))  # model -> size -> [auc]
    with open(path) as f:
        for r in csv.DictReader(f):
            runs[r["model"]][int(r["size"])].append(float(r["test_auc"]))
    return runs


def stats(by_size):
    sizes = np.array(sorted(by_size))
    aucs = [np.array(by_size[s]) for s in sizes]
    mean = np.array([a.mean() for a in aucs])
    std = np.array([a.std(ddof=1) if len(a) > 1 else 0.0 for a in aucs])
    return sizes, mean, std, min(len(a) for a in aucs)


def plot(runs, out, watermark=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    nseeds = set()
    for ax, (title, tcolor, models) in zip(axes.flat, PANELS):
        ax.set_title(title, loc="left", color=tcolor, fontweight="bold", fontsize=11)
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
        ax.set_xscale("log")
        if not any(m in runs for m in models):
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                    va="center", color="0.5", fontsize=12)
            continue
        for m in models:
            if m not in runs:
                continue
            label, color, marker, ls = STYLE[m]
            x, y, e, n = stats(runs[m])
            nseeds.add(n)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.2, lw=0)
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=4, ls=ls, lw=1.2,
                        capsize=0, label=label)
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    ticks = sorted({s for m in runs.values() for s in m})
    axes[0, 0].set_xticks(ticks, [f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k" for t in ticks])
    axes[0, 0].minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("Training sample size")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")
    fig.tight_layout()
    if watermark:  # e.g. "MOCK": mock figures must never pass for results
        fig.text(0.5, 0.5, watermark, ha="center", va="center", rotation=30,
                 fontsize=90, color="red", alpha=0.15, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}; seeds per point (min): {sorted(nseeds)}")


def demo(out):
    """Synthetic data: checks the pipeline runs and missing panels say pending."""
    rng = np.random.default_rng(0)
    path = out.rsplit(".", 1)[0] + "_demo.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "size", "seed", "test_auc"])
        for i, m in enumerate(STYLE):
            for s in [10_000, 30_000, 100_000, 300_000, 1_000_000]:
                for seed in range(3):
                    auc = 0.986 - 0.02 * (1e4 / s) ** 0.5 - 0.001 * i + rng.normal(0, 0.0015 * (1e4 / s) ** 0.3)
                    w.writerow([m, s, seed, f"{auc:.5f}"])
    runs = load(path)
    assert set(runs["bivector_patch"]) == {10_000, 30_000, 100_000, 300_000, 1_000_000}
    x, y, e, n = stats(runs["bivector_patch"])
    assert n == 3 and (e > 0).all() and list(x) == sorted(x)
    plot(runs, out)


if __name__ == "__main__":
    if sys.argv[1] == "--demo":
        demo(sys.argv[2])
    else:
        plot(load(sys.argv[1]), sys.argv[2])
