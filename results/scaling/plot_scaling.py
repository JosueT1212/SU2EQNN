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
from pathlib import Path

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
PATCH = ["r1", "bivector_patch", "subjet_mass_patch", "pseudoscalar_triad_patch",
         "pseudoscalar_handedness_patch", "single_channel_axis_patch", "single_channel_axis_b_patch"]
for _m in SENCILLA[2:]:  # patch twins: same colour and marker as the input version, dashed
    STYLE[_m + "_patch"] = (STYLE[_m][0] + ", patch", STYLE[_m][1], STYLE[_m][2], "--")
PANELS = [
    ("Naive, sencilla", "darkred", SENCILLA),
    ("Naive, patch",    "darkred", PATCH),
    ("Little group, sencilla", "navy", [m + "_little" for m in SENCILLA]),
    ("Little group, patch",    "navy", [m + "_little" for m in PATCH]),
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


PAIR_TITLES = {  # one panel per variant, baseline vs variant (Li et al. Fig. 4 layout)
    "grade2_bivectors":        "Subjet-axis bivectors (sencilla)",
    "bivector_patch":          "Subjet-axis bivectors (patch)",
    "subjet_mass":             "Subjet masses",
    "pseudoscalar_triad":      "Triad handedness",
    "pseudoscalar_handedness": "Local handedness",
    "single_channel_axis":     "Jet axis in channel",
    "single_channel_axis_b":   "Jet axis in channel + lead pseudoscalar",
}
PAIR_TITLES.update({m + "_patch": PAIR_TITLES[m] + " (patch)" for m in SENCILLA[2:]})
LI_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 11}


def _li_axis(ax, title, ticks):
    ax.set_title(title, loc="left", color="#8B1A1A", fontweight="bold", fontsize=14)
    ax.grid(True, ls="--", lw=0.8, color="0.6")
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:d}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("AUC")


def _li_curve(ax, by_size, label, color):
    x, y, e, _ = stats(by_size)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.2, lw=0)
    ax.errorbar(x, y, yerr=e, color=color, fmt="-", lw=1.8, elinewidth=1.8, capsize=0,
                label=label)


def plot_pairs(runs, outdir, baseline="r1"):
    """Li et al. style: one panel per variant against the baseline, shared y-range.

    Writes pairs.pdf/.png (all panels) and pair_<variant>.png (one per panel).
    """
    outdir = Path(outdir)
    variants = [m for m in PAIR_TITLES if m in runs]
    ticks = sorted({s for m in runs.values() for s in m})
    lo = min(stats(runs[m])[1].min() - stats(runs[m])[2].max() for m in [baseline, *variants])
    hi = max(stats(runs[m])[1].max() + stats(runs[m])[2].max() for m in [baseline, *variants])
    ylim = (lo - 0.0005, hi + 0.0005)
    with plt.rc_context(LI_RC):
        ncol = 2
        nrow = -(-len(variants) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.6 * nrow), squeeze=False)
        for ax in axes.flat[len(variants):]:
            ax.set_visible(False)
        for ax, m in zip(axes.flat, variants):
            _li_axis(ax, PAIR_TITLES[m], ticks)
            _li_curve(ax, runs[baseline], "Baseline", "tab:blue")
            _li_curve(ax, runs[m], f"+ {PAIR_TITLES[m].lower()}", "tab:orange")
            ax.set_ylim(*ylim)
            ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"pairs.{ext}", bbox_inches="tight", dpi=200)
        plt.close(fig)
        for m in variants:  # single panels
            fig, ax = plt.subplots(figsize=(6.5, 4.6))
            _li_axis(ax, PAIR_TITLES[m], ticks)
            _li_curve(ax, runs[baseline], "Baseline", "tab:blue")
            _li_curve(ax, runs[m], f"+ {PAIR_TITLES[m].lower()}", "tab:orange")
            ax.set_ylim(*ylim)
            ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(outdir / f"pair_{m}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
    print(f"wrote {outdir}/pairs.pdf/.png and {len(variants)} single panels")


FOUR = [  # (file stem, title, colour, models): one PNG per panel, Li et al. style, no grid
    ("naive_input", "Naive, input injection", "#8B1A1A", SENCILLA),
    ("naive_patch", "Naive, patch injection", "#8B1A1A", PATCH),
    ("little_group_input", "Little group, input injection", "#1F3A8A",
     [m + "_little" for m in SENCILLA]),
    ("little_group_patch", "Little group, patch injection", "#1F3A8A", [m + "_little" for m in PATCH]),
]
LITTLE_LABELS = {  # rest-frame ports; e/f/g are analogs, not the same observables
    "r1": "Baseline", "grade2_bivectors": "+ subjet-axis bivectors",
    "subjet_mass": "+ subjet masses", "pseudoscalar_handedness": "+ local handedness",
    "pseudoscalar_triad": "+ out-of-plane pseudoscalar",
    "single_channel_axis": "leading subjet in channel",
    "single_channel_axis_b": "leading subjet in channel + lead ps.",
    "bivector_patch": "+ subjet-axis bivectors, patch",
}
LITTLE_LABELS.update({m + "_patch": LITTLE_LABELS[m] + ", patch" for m in SENCILLA[2:]})
for _m, _label in LITTLE_LABELS.items():
    STYLE[_m + "_little"] = (_label, *STYLE[_m][1:])
FOUR_COLORS = {"r1": "tab:blue", "grade2_bivectors": "tab:orange", "subjet_mass": "tab:green",
               "pseudoscalar_triad": "tab:red", "pseudoscalar_handedness": "tab:purple",
               "single_channel_axis": "tab:brown", "single_channel_axis_b": "tab:pink",
               "bivector_patch": "tab:orange"}
FOUR_COLORS.update({m + "_patch": FOUR_COLORS[m] for m in SENCILLA[2:]})


def plot_four(runs, outdir):
    """Four PNGs (naive/little group x input/patch injection), shared y-range, no grid.

    Curves in a crowded panel are dodged a few percent along log-x so error bars
    do not sit on top of each other; a panel without data says "pending".
    """
    outdir = Path(outdir)
    ticks = sorted({s for m in runs.values() for s in m})
    present = [m for m in STYLE if m in runs]
    lo = min(stats(runs[m])[1].min() - stats(runs[m])[2].max() for m in present)
    hi = max(stats(runs[m])[1].max() + stats(runs[m])[2].max() for m in present)
    with plt.rc_context(LI_RC):
        for stem, title, tcolor, models in FOUR:
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            ax.set_xscale("log")
            ax.set_xticks(ticks, [f"{t:d}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("Training sample size", fontsize=18)
            ax.set_ylabel("AUC", fontsize=18)
            ax.set_ylim(lo - 0.0005, hi + 0.0005)
            models = [m for m in models if m in runs]
            if not models:
                ax.set_xlim(ticks[0] / 1.5, ticks[-1] * 1.5)
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", color="0.5", fontsize=14)
            for k, m in enumerate(models):
                x, y, e, _ = stats(runs[m])
                x = x * 1.035 ** (k - (len(models) - 1) / 2)   # dodge
                c = FOUR_COLORS[m.removesuffix("_little")]
                ax.fill_between(x, y - e, y + e, color=c, alpha=0.12, lw=0)
                ax.errorbar(x, y, yerr=e, color=c, fmt=STYLE[m][3], lw=1.8, elinewidth=1.6,
                            capsize=0, label=STYLE[m][0])
            if models:
                leg = ax.legend(loc="lower right", fontsize=10, framealpha=0.9, title=title,
                                title_fontproperties={"weight": "bold", "size": 11})
                leg.get_title().set_color("black")
                leg._legend_box.align = "left"
            fig.tight_layout()
            fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
    print(f"wrote {len(FOUR)} PNGs to {outdir}")


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
