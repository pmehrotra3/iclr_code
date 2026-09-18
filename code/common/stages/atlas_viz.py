"""stages/atlas_viz.py — figures for the ring atlas, written next to the numbers.

Layout, under output/<process>/:

  T_<T>/<model>/heatmap_{full_acc,mode_f1,hall_f1}.{png,pdf}
      one (K x d) heatmap per predictor per T. Each cell shows the metric at the BEST
      anchor budget for that cell, with the winning anchor count in brackets:
          87
        (6000)
      `analytic` uses no anchors, so its cells carry no bracket.
  T_<T>/summary_all_models_<key>.{png,pdf}
      the same metric for every predictor side by side, to compare at a glance.

  best_T_table.tex / best_T_table.png
      per predictor: its best T (by mean overall accuracy over cells), the three metrics
      there, and the modal winning anchor budget -- plus the full (K x d) grid for the
      single best (predictor, T), each cell "overall / mode F1 / hallucination F1".
  summary_vs_T.{png,pdf}
      mean over cells of the three metrics vs T, every predictor.
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from common.stages.atlas import run_dir, model_at, model_names

# ------------------------------------------------------------------ paper style
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7", "#52514e",
          "#0d366b", "#9c4a1a", "#0f6b4b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
BLUES = LinearSegmentedColormap.from_list(
    "paper_blues",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
     "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def _save(fig, viz_dir, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(viz_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    plt.close(fig)
    return paths

METRICS = [("full_acc", "overall accuracy"), ("mode_f1", "mode-basin $F_1$"),
           ("hall_f1", "hallucination $F_1$")]


def _grid(cells, ds, Ks, T, name, key):
    """(len(Ks), len(ds)) arrays of the metric and of the winning anchor count."""
    val = np.full((len(Ks), len(ds)), np.nan)
    nA = np.full((len(Ks), len(ds)), np.nan)
    for i, K in enumerate(Ks):
        for j, d in enumerate(ds):
            c = next((c for c in cells if c["d"] == d and c["K"] == K), None)
            if c is None:
                continue
            row = next((r for r in c["per_T"] if r["T"] == T), None)
            if row is None:
                continue
            best = model_at(row, name, key)      # best budget FOR THIS METRIC
            if best:
                val[i, j] = best[key]
                nA[i, j] = best["n_anchors"]
    return val, nA


def _heatmap(val, nA, ds, Ks, title, out_dir, stem):
    fig, ax = plt.subplots(figsize=(1.05 * len(ds) + 1.6, 0.62 * len(Ks) + 1.5))
    im = ax.imshow(val, vmin=0, vmax=1, aspect="auto", cmap=BLUES)
    ax.grid(False)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("modes $K$")
    ax.set_title(title, loc="left", color=INK2, fontsize=8)
    for i in range(len(Ks)):
        for j in range(len(ds)):
            if not np.isnan(val[i, j]):
                col = "white" if val[i, j] > 0.55 else INK
                ax.text(j, i - 0.13, f"{val[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=8, color=col)
                if not np.isnan(nA[i, j]) and nA[i, j] > 0:
                    ax.text(j, i + 0.22, f"({int(nA[i, j])})", ha="center", va="center",
                            fontsize=6, color=col, alpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    return _save(fig, out_dir, stem)


def _panels(cells, ds, Ks, T, names, key, label, out_dir, stem, ncol=4):
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.0 * len(Ks) * nrow + 1.0),
                             squeeze=False)
    im = None
    for i, ax in enumerate(axes.ravel()):
        if i >= len(names):
            ax.axis("off"); continue
        val, _ = _grid(cells, ds, Ks, T, names[i], key)
        im = ax.imshow(val, vmin=0, vmax=1, aspect="auto", cmap=BLUES)
        ax.grid(False)
        ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds, fontsize=7)
        ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks, fontsize=7)
        if i // ncol == nrow - 1:
            ax.set_xlabel("dimension $d$", fontsize=8)
        if i % ncol == 0:
            ax.set_ylabel("modes $K$", fontsize=8)
        ax.set_title(names[i], loc="left", color=INK2, fontsize=9)
        for r in range(len(Ks)):
            for c in range(len(ds)):
                if not np.isnan(val[r, c]):
                    ax.text(c, r, f"{val[r, c] * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if val[r, c] > 0.55 else INK)
    fig.suptitle(f"{label} (%) — best anchor budget per cell, true-score $T={T}$",
                 fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    cax = fig.add_axes((0.955, 0.12, 0.012, 0.76))
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    return _save(fig, out_dir, stem)


# ------------------------------------------------------------------ best-T summary
def _best_T(cells, Ts, name):
    """(T, mean metrics, modal winning budget, n_cells) maximising mean overall accuracy."""
    best = None
    for T in Ts:
        got = [model_at(next(r for r in c["per_T"] if r["T"] == T), name)
               for c in cells if any(r["T"] == T for r in c["per_T"])]
        got = [g for g in got if g]
        if not got:
            continue
        means = {k: sum(g[k] for g in got) / len(got) for k, _ in METRICS}
        counts = {}
        for g in got:
            counts[g["n_anchors"]] = counts.get(g["n_anchors"], 0) + 1
        modal = max(counts, key=counts.get) if counts else 0
        if best is None or means["full_acc"] > best[1]["full_acc"]:
            best = (T, means, modal, len(got))
    return best


def _best_T_tables(cfg, cells, ds, Ks, Ts, families, out_dir):
    pname = cfg.process
    rows = []
    for f in families:
        b = _best_T(cells, Ts, f)
        if b:
            rows.append((f, *b))
    rows.sort(key=lambda r: -r[2]["full_acc"])
    top = rows[0]
    bf, bT = top[0], top[1]
    hdr = (f"{pname}: best true-score $T$ per predictor, chosen by mean overall accuracy over "
           f"{top[4]} $(d, K)$ cells ($d \\in \\{{{', '.join(map(str, ds))}\\}}$, "
           f"$K \\in \\{{{', '.join(map(str, Ks))}\\}}$). Anchors = modal winning budget.")

    L = ["% Auto-generated by common/stages/atlas_viz.py -- requires \\usepackage{booktabs}",
         "\\begin{table}[t]", "\\centering", "\\caption{" + hdr + "}",
         f"\\label{{tab:atlas-bestT-{pname}}}",
         "\\begin{tabular}{l r rrr r}", "\\toprule",
         "Predictor & Best $T$ & Overall acc. & Mode $F_1$ & Halluc.\\ $F_1$ & Anchors \\\\",
         "\\midrule"]
    for f, T, m, modal, _ in rows:
        anc = "--" if modal == 0 else str(modal)
        L.append(f"{f.replace('_', chr(92) + '_')} & {T} & {m['full_acc'] * 100:.1f} & "
                 f"{m['mode_f1'] * 100:.1f} & {m['hall_f1'] * 100:.1f} & {anc} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    grids = {k: _grid(cells, ds, Ks, bT, bf, k)[0] for k, _ in METRICS}
    L += ["\\begin{table}[t]", "\\centering",
          "\\caption{" + f"{pname}, best predictor \\emph{{{bf.replace('_', chr(92) + '_')}}} at "
          f"$T={bT}$ (mean overall accuracy {top[2]['full_acc'] * 100:.1f}\\%). Each cell is "
          "overall accuracy / mode-basin $F_1$ / hallucination $F_1$ (\\%), at that cell's best "
          "anchor budget." + "}",
          f"\\label{{tab:atlas-bestgrid-{pname}}}",
          "\\begin{tabular}{l " + "r" * len(ds) + "}", "\\toprule",
          "$K \\backslash d$ & " + " & ".join(f"${d}$" for d in ds) + " \\\\", "\\midrule"]
    for i, K in enumerate(Ks):
        cellstr = []
        for j in range(len(ds)):
            v = [grids[k][i, j] for k, _ in METRICS]
            cellstr.append("--" if np.isnan(v[0]) else " / ".join(f"{x * 100:.1f}" for x in v))
        L.append(f"{K} & " + " & ".join(cellstr) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p = os.path.join(out_dir, "best_T_table.tex")
    with open(p, "w") as fh:
        fh.write("\n".join(L))

    # ---- the same content as a PNG table
    fig, axes = plt.subplots(2, 1, figsize=(1.7 * len(ds) + 5.0,
                                            0.36 * (len(rows) + len(Ks)) + 2.6),
                             gridspec_kw={"height_ratios": [len(rows) + 2, len(Ks) + 2]})
    for ax in axes:
        ax.axis("off")
    axes[0].set_title(
        f"{pname} — best true-score T per predictor (mean over {top[4]} (d, K) cells)\n"
        f"best overall: {bf} at T={bT}, mean overall accuracy {top[2]['full_acc'] * 100:.1f}%",
        loc="left", fontsize=10, color=INK)
    t1 = axes[0].table(
        cellText=[[f, str(T), f"{m['full_acc'] * 100:.1f}", f"{m['mode_f1'] * 100:.1f}",
                   f"{m['hall_f1'] * 100:.1f}", "—" if modal == 0 else str(modal)]
                  for f, T, m, modal, _ in rows],
        colLabels=["predictor", "best T", "overall acc (%)", "mode F1 (%)",
                   "halluc. F1 (%)", "anchors"],
        cellLoc="center", loc="upper center")
    t1.auto_set_font_size(False); t1.set_fontsize(8); t1.scale(1, 1.25)
    axes[1].set_title(
        f"{bf} at T={bT}: overall / mode F1 / hallucination F1 (%), best anchor budget per cell",
        loc="left", fontsize=10, color=INK)
    body = []
    for i, K in enumerate(Ks):
        r = [str(K)]
        for j in range(len(ds)):
            v = [grids[k][i, j] for k, _ in METRICS]
            r.append("—" if np.isnan(v[0]) else " / ".join(f"{x * 100:.1f}" for x in v))
        body.append(r)
    t2 = axes[1].table(cellText=body, colLabels=["K \\ d"] + [str(d) for d in ds],
                       cellLoc="center", loc="upper center")
    t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1, 1.25)
    fig.tight_layout()
    return [p] + _save(fig, out_dir, "best_T_table"), top


def run(cfg):
    pname = cfg.process
    out_dir = run_dir(cfg)
    with open(os.path.join(out_dir, "atlas_results.json")) as f:
        blob = json.load(f)
    cells = blob["cells"]
    ds = sorted({c["d"] for c in cells}); Ks = sorted({c["K"] for c in cells})
    Ts = sorted({r["T"] for c in cells for r in c["per_T"]})
    names = model_names(cells)
    has_an = all("analytic" in r for c in cells for r in c["per_T"])
    families = names + (["analytic"] if has_an else [])
    made = []

    for T in Ts:
        tdir = os.path.join(out_dir, f"T_{T}")
        for name in families:
            mdir = os.path.join(tdir, name)
            os.makedirs(mdir, exist_ok=True)
            anc = ("no anchors (analytic sampler)" if name == "analytic"
                   else "cell = metric, (brackets) = winning anchor count")
            for key, label in METRICS:
                val, nA = _grid(cells, ds, Ks, T, name, key)
                made += _heatmap(val, nA, ds, Ks,
                                 f"{pname} — {name}: {label} (%), $T={T}$\n{anc}",
                                 mdir, f"heatmap_{key}")
        for key, label in METRICS:
            made += _panels(cells, ds, Ks, T, families, key, label, tdir,
                            f"summary_all_models_{key}")

    fig, axes = plt.subplots(1, 3, figsize=(3 * 3.4, 2.6))
    for ax, (key, label) in zip(axes, METRICS):
        for i, name in enumerate(families):
            ys = [np.nanmean(_grid(cells, ds, Ks, T, name, key)[0]) for T in Ts]
            ax.plot(Ts, ys, marker=MARKERS[i % 10],
                    color=SERIES[i % 10] if name != "analytic" else INK2,
                    ls="--" if name == "analytic" else "-", label=name)
        ax.set_xlabel("true-score steps $T$"); ax.set_ylabel(label + " (mean over cells)")
        ax.set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", ncol=2)
    fig.suptitle(f"ring atlas, {pname}: fitted on true-score anchors (best budget per cell), "
                 f"scored on the learned sampler", fontsize=8, color=INK2)
    fig.tight_layout()
    made += _save(fig, out_dir, "summary_vs_T")

    tbl, top = _best_T_tables(cfg, cells, ds, Ks, Ts, families, out_dir)
    made += tbl
    print(f"[atlas_viz:{pname}] wrote {len(made)} files under {out_dir}; "
          f"best: {top[0]} at T={top[1]}, mean overall accuracy {top[2]['full_acc'] * 100:.1f}%")
    return {"figures": made, "best": {"model": top[0], "T": top[1],
                                      "mean_full_acc": top[2]["full_acc"]}}
