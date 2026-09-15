"""stages/visualize.py — results[_tag].json -> paper figures + LaTeX tables.

Reads only the results file (never re-runs models). Every plot is its own single-column
figure, saved as PDF + PNG under visualization/<process>/:

  acc_vs_dim_K{K}          full accuracy vs d: every classifier (best budget) + analytic
  acc_vs_budget_d{d}       primary classifier's full accuracy vs labeled-seed budget, per K
  hallf1_vs_dim            hallucination F1 of the primary classifier vs d, per K
  hallucination_vs_dim     ground-truth hallucination rate vs d, per K
  heatmap_{full_acc,mode_acc,hall_f1}   primary classifier over the (d, K) grid
  ladder                   mean full accuracy / hall F1 per classifier (only if > 1 model)
  table_<process>[_tag].tex          primary classifier, one row per (K, d)
  table_<process>[_tag]_ladder.tex   full accuracy of every classifier per cell
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import utils

# ------------------------------------------------------------------ paper style
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7", "#52514e",
          "#0d366b", "#9c4a1a", "#0f6b4b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
BLUES = LinearSegmentedColormap.from_list(
    "paper_blues",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
     "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
COL_W, COL_H = 3.4, 2.5

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


def _new_fig():
    return plt.subplots(figsize=(COL_W, COL_H))


def _log2_axis(ax, xs, label):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
    ax.set_xlabel(label)


class Blob:
    """Thin accessor over results.json."""
    def __init__(self, blob, primary):
        self.blob = blob
        self.results = blob["results"]
        self.dk = {(r["d"], r["K"]): r for r in self.results}
        self.Ks = sorted({r["K"] for r in self.results})
        self.ds = sorted({r["d"] for r in self.results})
        seen = []
        for r in self.results:
            for c in r["classifiers"]:
                if c["name"] not in seen:
                    seen.append(c["name"])
        self.models = seen
        self.primary = primary if primary in seen else (seen[0] if seen else None)
        self.has_analytic = all("analytic" in r for r in self.results)

    def best(self, d, K, name, key):
        r = self.dk.get((d, K))
        b = r["best"].get(name) if r else None
        return b[key] if b else np.nan

    def rows(self, d, K, name):
        r = self.dk.get((d, K))
        return [c for c in r["classifiers"] if c["name"] == name] if r else []


# ------------------------------------------------------------------ figures
def plot_accuracy_vs_dimension(B, viz_dir):
    made = []
    for K in B.Ks:
        xs = [d for d in B.ds if (d, K) in B.dk]
        fig, ax = _new_fig()
        for i, name in enumerate(B.models):
            ax.plot(xs, [B.best(d, K, name, "full_acc") for d in xs],
                    marker=MARKERS[i % 10], color=SERIES[i % 10], label=name)
        if B.has_analytic:
            ax.plot(xs, [B.dk[(d, K)]["analytic"]["full_acc"] for d in xs],
                    marker="x", color=INK2, ls="--", label="analytic field")
        ax.axhline(0.95, color=INK2, lw=0.6, ls=":")
        _log2_axis(ax, xs, "dimension $d$")
        ax.set_ylabel("full accuracy"); ax.set_ylim(0.6, 1.02)
        ax.set_title(f"$K = {K}$", loc="left", color=INK2)
        ax.legend(loc="lower left", ncol=2 if len(B.models) > 4 else 1)
        made += _save(fig, viz_dir, f"acc_vs_dim_K{K}")
    return made


def plot_accuracy_vs_budget(B, viz_dir):
    made = []
    for d in B.ds:
        fig, ax = _new_fig()
        any_line = False
        for i, K in enumerate(B.Ks):
            rows = sorted(B.rows(d, K, B.primary), key=lambda a: a["n_train"])
            if len(rows) < 2:
                continue
            any_line = True
            ax.plot([a["n_train"] for a in rows], [a["full_acc"] for a in rows],
                    marker=MARKERS[i], color=SERIES[i], label=f"$K={K}$")
        if not any_line:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_xlabel("labeled seeds $n$"); ax.set_ylabel("full accuracy")
        ax.set_title(f"$d = {d}$, {B.primary}", loc="left", color=INK2); ax.legend(loc="lower right")
        made += _save(fig, viz_dir, f"acc_vs_budget_d{d}")
    return made


def plot_metric_vs_dim(B, viz_dir, key, stem, ylabel, source="best"):
    fig, ax = _new_fig()
    for i, K in enumerate(B.Ks):
        xs = [d for d in B.ds if (d, K) in B.dk]
        ys = [B.best(d, K, B.primary, key) if source == "best" else B.dk[(d, K)][key] for d in xs]
        ax.plot(xs, ys, marker=MARKERS[i], color=SERIES[i], label=f"$K={K}$")
    _log2_axis(ax, B.ds, "dimension $d$"); ax.set_ylabel(ylabel)
    if source == "best":
        ax.set_ylim(0, 1.02); ax.set_title(B.primary, loc="left", color=INK2)
    ax.legend()
    return _save(fig, viz_dir, stem)


def heatmap_grid(A, ds, Ks, title, viz_dir, stem):
    """(len(Ks), len(ds)) array of a metric in [0, 1] -> annotated heatmap, saved as PDF + PNG."""
    fig, ax = plt.subplots(figsize=(COL_W, 0.45 * len(Ks) + 0.9))
    im = ax.imshow(A, vmin=0, vmax=1, aspect="auto", cmap=BLUES)
    ax.grid(False)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("modes $K$")
    ax.set_title(title, loc="left", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    for i in range(len(Ks)):
        for j in range(len(ds)):
            if A[i, j] == A[i, j]:
                ax.text(j, i, f"{A[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if A[i, j] > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    return _save(fig, viz_dir, stem)


def _heatmap(B, viz_dir, key, stem, title):
    A = np.array([[B.best(d, K, B.primary, key) for d in B.ds] for K in B.Ks])
    return heatmap_grid(A, B.ds, B.Ks, f"{title} — {B.primary}", viz_dir, stem)


def plot_heatmaps(B, viz_dir):
    return (_heatmap(B, viz_dir, "full_acc", "heatmap_full_acc", "full accuracy (%)")
            + _heatmap(B, viz_dir, "mode_acc", "heatmap_mode_acc", "mode accuracy (%)")
            + _heatmap(B, viz_dir, "hall_f1", "heatmap_hall_f1", "hallucination $F_1$ (%)"))


def plot_ladder(B, viz_dir):
    """Mean over cells of full accuracy and hallucination F1, per classifier (config order)."""
    if len(B.models) < 2:
        return []
    cells = list(B.dk)
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    xs = range(len(B.models))
    for i, (key, lab) in enumerate([("full_acc", "full accuracy"), ("hall_f1", "hallucination $F_1$")]):
        ys = [np.nanmean([B.best(d, K, m, key) for d, K in cells]) for m in B.models]
        ax.plot(list(xs), ys, marker=MARKERS[i], color=SERIES[i], label=lab)
    if B.has_analytic:
        ax.axhline(np.mean([B.dk[c]["analytic"]["full_acc"] for c in cells]), color=INK2, lw=0.6,
                   ls="--", label="analytic field (full acc.)")
    ax.set_xticks(list(xs)); ax.set_xticklabels(B.models, rotation=45, ha="right")
    ax.set_ylabel("mean over $(d, K)$ cells"); ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    ax.set_title("classifier capacity ladder", loc="left", color=INK2)
    return _save(fig, viz_dir, "ladder")


# ------------------------------------------------------------------ tables
def write_table(B, viz_dir, process, tag):
    lines = ["% Auto-generated by common/stages/visualize.py -- requires \\usepackage{booktabs}",
             "\\begin{table}[t]", "\\centering",
             f"\\caption{{Seed-fate prediction ({process}, classifier: {B.primary}). "
             "\\emph{Halluc.\\ rate} is the ground-truth hallucination rate on held-out seeds; "
             "\\emph{Analytic} is the full accuracy of the analytic-field forward pass; the remaining "
             "columns are the classifier at the labeled-seed budget with the best full accuracy.}",
             f"\\label{{tab:fate-{process}{'-' + tag if tag else ''}}}",
             "\\begin{tabular}{cc r r rrrr}", "\\toprule",
             "$K$ & $d$ & Halluc.\\ rate & Analytic & Full acc. & Seeds & Mode acc. & Halluc.\\ $F_1$ \\\\",
             "\\midrule"]
    prev_K = None
    for r in sorted(B.results, key=lambda r: (r["K"], r["d"])):
        b = r["best"].get(B.primary)
        if b is None:
            continue
        if prev_K is not None and r["K"] != prev_K:
            lines.append("\\midrule")
        prev_K = r["K"]
        an = f"{r['analytic']['full_acc'] * 100:.1f}" if "analytic" in r else "--"
        lines.append(f"{r['K']} & {r['d']} & {r['hall_gt'] * 100:.1f} & {an} & "
                     f"{b['full_acc'] * 100:.1f} & {b['n_train'] / 1e6:.1f}M & "
                     f"{b['mode_acc'] * 100:.1f} & {b['hall_f1'] * 100:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p = os.path.join(viz_dir, f"table_{process}{'_' + tag if tag else ''}.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return [p]


def write_ladder_table(B, viz_dir, process, tag):
    if len(B.models) < 2:
        return []
    cols = (["analytic"] if B.has_analytic else []) + B.models
    lines = ["% Auto-generated by common/stages/visualize.py -- requires \\usepackage{booktabs}",
             "\\begin{table}[t]", "\\centering",
             f"\\caption{{Full accuracy (\\%) of every seed-fate classifier ({process}), "
             "best labeled-seed budget per cell. Classifier families are ordered by capacity.}",
             f"\\label{{tab:ladder-{process}{'-' + tag if tag else ''}}}",
             "\\begin{tabular}{cc " + "r" * len(cols) + "}", "\\toprule",
             "$K$ & $d$ & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    prev_K = None
    for r in sorted(B.results, key=lambda r: (r["K"], r["d"])):
        if prev_K is not None and r["K"] != prev_K:
            lines.append("\\midrule")
        prev_K = r["K"]
        vals = ([f"{r['analytic']['full_acc'] * 100:.1f}"] if B.has_analytic else []) \
            + [f"{B.best(r['d'], r['K'], m, 'full_acc') * 100:.1f}" for m in B.models]
        lines.append(f"{r['K']} & {r['d']} & " + " & ".join(vals) + " \\\\")
    lines.append("\\midrule")
    means = ([f"{np.mean([r['analytic']['full_acc'] for r in B.results]) * 100:.1f}"] if B.has_analytic else []) \
        + [f"{np.nanmean([B.best(d, K, m, 'full_acc') for d, K in B.dk]) * 100:.1f}" for m in B.models]
    lines.append("\\multicolumn{2}{c}{mean} & " + " & ".join(means) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p = os.path.join(viz_dir, f"table_{process}{'_' + tag if tag else ''}_ladder.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return [p]


def run(cfg) -> dict:
    pname, tag = cfg.process, cfg.eval.tag
    viz_dir = utils.process_dir(cfg.paths.viz, pname)
    if tag:
        viz_dir = os.path.join(viz_dir, tag)
    os.makedirs(viz_dir, exist_ok=True)
    with open(utils.results_path(cfg.paths.output, pname, tag, "json")) as f:
        B = Blob(json.load(f), cfg.classifier.primary)
    made = []
    jobs = [lambda: plot_accuracy_vs_dimension(B, viz_dir),
            lambda: plot_accuracy_vs_budget(B, viz_dir),
            lambda: plot_metric_vs_dim(B, viz_dir, "hall_f1", "hallf1_vs_dim", "hallucination $F_1$"),
            lambda: plot_metric_vs_dim(B, viz_dir, "hall_gt", "hallucination_vs_dim",
                                       "hallucination rate (ground truth)", source="cell"),
            lambda: plot_heatmaps(B, viz_dir),
            lambda: plot_ladder(B, viz_dir),
            lambda: write_table(B, viz_dir, pname, tag),
            lambda: write_ladder_table(B, viz_dir, pname, tag)]
    for job in jobs:
        try:
            made += job()
        except Exception as e:  # one bad panel must not kill the rest
            print(f"[viz:{pname}] a figure failed: {e!r}")
    for p in made:
        print(f"[viz:{pname}] wrote {p}")
    return {"figures": made}
