"""plot_T_sweep.py -- accuracy against d for the pullback grids T, averaged over K (paper Fig. 6 style).

Reads output/<run>/<process>/<variant>/T<T>/results.csv (no GPU, no recomputation) and writes

    experiment_results/figures2/
        README.md
        <variant>/                      unweighted | weighted mixture
            ddim_vs_flow/               DDIM (top row) against flow matching (bottom row)
                anchors_<b>.pdf/.png    one file per anchor budget b (anchors per mode)
            ddim/  flow/  heun/  rk45/  dpmpp2m/
                anchors_<b>.pdf/.png    one sampler, one row

Columns are the classifiers, lines are T = 250 / 500 / 750 (one blue ramp, light -> dark, plus a
marker per T), the line is the mean over K of the overall accuracy (each already a mean over the
runs) and the band is +-1 standard deviation across K. At d = 2 the average is over K = 2, 4, 8,
since K = 16 centres cannot be placed there.

    python experiment_results/plot_T_sweep.py            # run abc123
"""
from __future__ import annotations
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures2")

PROCESSES = [("ddim", "DDIM"), ("flow", "Flow matching"), ("heun", "Heun"), ("rk45", "RK45"),
             ("dpmpp2m", "DPM-Solver++(2M)")]
MODELS = [("knn", "$k$NN"), ("altered_knn", "w-$k$NN"), ("quadratic", "quadratic"), ("polar3", "polar")]
VARIANTS = ["unweighted", "weighted"]
TS = [250, 500, 750]
# one-hue ordinal ramp (validated: monotone lightness, visible step gaps, light end >= 2:1 on white)
COLORS = {250: "#86b6ef", 500: "#2a78d6", 750: "#104281"}
MARKERS = {250: "o", 500: "s", 750: "^"}


def load(run, proc, var, T):
    """{(budget, model, K, d): overall accuracy in %} of one sweep."""
    path = os.path.join(ROOT, "output", run, proc, var, f"T{T}", "results.csv")
    with open(path) as f:
        return {(int(r["n_per_mode"]), r["model"], int(r["K"]), int(r["d"])): 100 * float(r["full_acc"])
                for r in csv.DictReader(f)}


def curve(rows, b, model):
    """(ds, mean over K, std over K) of one classifier at one budget."""
    ds = sorted({k[3] for k in rows if k[0] == b and k[1] == model})
    vals = [[v for k, v in rows.items() if k[0] == b and k[1] == model and k[3] == d] for d in ds]
    return np.array(ds), np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])


def style():
    """Paper style: Times (the ICLR body font), 7-8.5 pt text at the final size, quiet axes."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": "#6b6b6b", "xtick.color": "#4a4a4a",
        "ytick.color": "#4a4a4a", "axes.labelcolor": "#2b2b2b", "text.color": "#2b2b2b",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, by_T, b, model, ds_all):
    for T in TS:
        ds, m, s = curve(by_T[T], b, model)
        ax.fill_between(ds, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=COLORS[T],
                        alpha=0.13, lw=0, zorder=1)
        ax.plot(ds, m, color=COLORS[T], lw=1.25, marker=MARKERS[T], ms=3.3, mec="white", mew=0.5,
                label=f"$T = {T}$", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds_all)
    ax.set_xticklabels([str(d) if i % 2 == 0 else "" for i, d in enumerate(ds_all)])   # 2, 8, 32, ...
    ax.minorticks_off()
    ax.set_xlim(ds_all[0] / 1.25, ds_all[-1] * 1.25)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, color="#e6e6e6", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


WIDTH = 5.5          # inches: the ICLR text width, so the figure is included at 100 %


def figure(rows_of, row_names, b, path):
    """rows_of: one {T: rows} per figure row; columns are the classifiers."""
    n_rows = len(rows_of)
    panel_h, gap, top_in, bottom_in = 0.85, 0.2, 0.44, 0.34        # inches: wide, short panels
    height = top_in + bottom_in + n_rows * panel_h + (n_rows - 1) * gap
    fig, axes = plt.subplots(n_rows, len(MODELS), figsize=(WIDTH, height), sharey=True,
                             sharex=True, squeeze=False)
    ds_all = sorted({k[3] for rows in rows_of for k in rows[TS[0]]})
    for i, (by_T, name) in enumerate(zip(rows_of, row_names)):
        for j, (model, title) in enumerate(MODELS):
            ax = axes[i, j]
            panel(ax, by_T, b, model, ds_all)
            if i == 0:
                ax.set_title(title, pad=3)
            if j == 0:
                ax.set_ylabel("accuracy (%)", labelpad=2)
                ax.text(-0.36, 0.5, name, transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TS), frameon=False, handlelength=2.2,
               columnspacing=1.8, bbox_to_anchor=(0.54, 1.0), borderaxespad=0.0)
    fig.supxlabel("dimension $d$", fontsize=8, x=0.54, y=0.0)
    fig.subplots_adjust(left=0.14, right=0.995, bottom=bottom_in / height, top=1 - top_in / height,
                        wspace=0.1, hspace=gap / panel_h)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


README = """# figures2: accuracy against d for the pullback grids T

Overall accuracy against the dimension d for the pullback grids T = 250 / 500 / 750, averaged over
K (line: mean over K; band: +-1 standard deviation across K; at d = 2 the mean is over K = 2, 4, 8).
Columns: kNN, w-kNN, quadratic, polar. Every file is a PDF (for the paper) with a PNG preview.
Regenerated by `python3 experiment_results/build.py` (or `plot_T_sweep.py` alone); reads
output/abc123, no GPU. presenting_results.tex includes all of them.

| folder | what it shows |
|---|---|
| `unweighted/` , `weighted/` | the mixture variant (uniform weights w_i = 1/K, or random weights) |
| `<variant>/ddim_vs_flow/` | DDIM (top row) against flow matching (bottom row) |
| `<variant>/ddim/`, `flow/`, `heun/`, `rk45/`, `dpmpp2m/` | one sampler, one row |
| `anchors_<b>.pdf` | the classifiers fit on b anchors per mode (b = 2000 ... 20000) |

Main-text Figure 6: `unweighted/ddim_vs_flow/anchors_20000.pdf` (the budget of Table S1).
"""


def write_all(run="abc123"):
    """Every figure of every variant, sampler and budget; returns [(variant, folder, b, relpath)]."""
    style()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(README)
    written = []
    for var in VARIANTS:
        data = {p: {T: load(run, p, var, T) for T in TS} for p, _ in PROCESSES}
        budgets = sorted({k[0] for k in data["ddim"][TS[0]]})
        folders = [("ddim_vs_flow", [data["ddim"], data["flow"]], ["DDIM", "Flow matching"])]
        folders += [(p, [data[p]], [name]) for p, name in PROCESSES]
        for folder, rows_of, names in folders:
            out_dir = os.path.join(OUT, var, folder)
            os.makedirs(out_dir, exist_ok=True)
            for b in budgets:
                figure(rows_of, names, b, os.path.join(out_dir, f"anchors_{b}"))
                written.append((var, folder, b, f"{var}/{folder}/anchors_{b}.pdf"))
    print(f"[figures2] wrote {len(written)} figures (.pdf + .png) to {OUT}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="abc123")
    write_all(ap.parse_args().run)


if __name__ == "__main__":
    main()
