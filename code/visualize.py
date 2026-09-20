"""
visualize.py — Stage 3. Turn a run's results.json into heatmaps and a table (no line plots).

One unified schema for every process: flat rows of (d, K, budget, model) with fate metrics.
Everything plotted is the "aggregate" block of results.json: each metric is the mean over the
repeat seeds and <metric>_std its std, so every cell reads mean +- std (in %). The best anchor
budget per cell is chosen by mean full accuracy.

Layout (output/<run_id>/<process>/T<T_true>/):

    results.json, results.csv
    table.tex                         the table (best anchor budget per cell)
    table.png
    <model>.png                       heatmaps per model, best anchor budget
    anchors_<b>/
        <model>.png                   heatmaps per model AT that anchor budget

Each <model>.png is a row of (d, K) heatmaps: full accuracy / mode F1 / hallucination F1,
coloured by the mean and annotated "mean+-std".
"""
from __future__ import annotations
import os
import json
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import resolve_sweep_dir

PANELS = [("full_acc", "full accuracy"), ("mode_f1", "mode F1"), ("hall_f1", "hallucination F1")]


def _load(cfg):
    d = resolve_sweep_dir(cfg.paths.output, cfg.run_id, cfg.process.name, cfg.process.T_true)
    if d is None:
        raise FileNotFoundError(
            f"no results found for {cfg.process.name} T={cfg.process.T_true} under {cfg.paths.output}")
    with open(os.path.join(d, "results.json")) as f:
        return json.load(f), d


def _rows(blob):
    """The mean +- std rows. Older results.json files (single seed, no aggregate block) are
    read as one-seed aggregates with std 0 so they still plot."""
    if blob.get("aggregate"):
        return blob["aggregate"]
    rows = []
    for r in blob["results"]:
        r = dict(r)
        for k in list(r):
            if isinstance(r[k], float) and not k.endswith("_std"):
                r.setdefault(k + "_std", 0.0)
        r.setdefault("n_seeds", 1)
        rows.append(r)
    return rows


def _fmt(mean, std, digits=1):
    """'87.3+-1.2' in percent; '--' for a missing cell."""
    if mean != mean:
        return "--"
    return f"{mean * 100:.{digits}f}$\\pm${std * 100:.{digits}f}"


def _axes(results):
    return sorted({r["d"] for r in results}), sorted({r["K"] for r in results})


def _models(results):
    out = []
    for r in results:
        if r["model"] not in out:
            out.append(r["model"])
    return out


def _budgets(results):
    return sorted({int(r["n_per_mode"]) for r in results})


def _cellmap(rows, model):
    """{(d, K): best-full_acc row} for one model over the given rows."""
    best = {}
    for r in rows:
        if r["model"] != model:
            continue
        key = (r["d"], r["K"])
        if key not in best or r["full_acc"] > best[key]["full_acc"]:
            best[key] = r
    return best


# --------------------------------------------------------------------------------------
# heatmaps
# --------------------------------------------------------------------------------------
def _heat(ax, A, S, ds, Ks, title):
    """Cells coloured by the mean A and annotated 'mean+-std' (%, S = std)."""
    im = ax.imshow(A, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("modes $K$")
    ax.set_title(title, fontsize=10)
    for i in range(len(Ks)):
        for j in range(len(ds)):
            v, sd = A[i, j], S[i, j]
            if v == v:
                txt = f"{v * 100:.0f}" if sd != sd else f"{v * 100:.0f}$\\pm${sd * 100:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    return im


def plot_model(cellmap, ds, Ks, model, suptitle, out_dir):
    """full_acc / mode_f1 / hall_f1 heatmaps for one model over the (d, K) grid."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 0.55 * len(Ks) + 3.0),
                             squeeze=False)
    im = None
    for ax, (key, label) in zip(axes[0], PANELS):
        A = np.full((len(Ks), len(ds)), np.nan)
        S = np.full((len(Ks), len(ds)), np.nan)
        for i, K in enumerate(Ks):
            for j, d in enumerate(ds):
                r = cellmap.get((d, K))
                if r is not None:
                    A[i, j] = r[key]
                    S[i, j] = r.get(key + "_std", float("nan"))
        im = _heat(ax, A, S, ds, Ks, label)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(suptitle, fontsize=12)
    p = os.path.join(out_dir, f"{model}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------------------
# table (best anchor budget per cell), tex + png
# --------------------------------------------------------------------------------------
def _table_rows(results, ds, Ks, models):
    """Per (K, d): best mean full accuracy of each model as (mean, std) fractions, at the anchor
    budget achieving the best mean."""
    rows = []
    for K in Ks:
        for d in ds:
            vals = []
            for m in models:
                cell = _cellmap([r for r in results if r["d"] == d and r["K"] == K], m).get((d, K))
                vals.append((cell["full_acc"], cell.get("full_acc_std", float("nan"))) if cell
                            else (float("nan"), float("nan")))
            rows.append((K, d, vals))
    return rows


def _n_seeds(results):
    return max(int(r.get("n_seeds", 1)) for r in results) if results else 1


def write_summary_tex(results, ds, Ks, models, sampler, T_true, out_dir):
    rows = _table_rows(results, ds, Ks, models)
    ns = _n_seeds(results)
    cols = "cc " + "r" * len(models)
    L = ["% Auto-generated by visualize.py -- requires \\usepackage{booktabs}",
         "\\begin{table}[t]", "\\centering",
         f"\\caption{{Seed-fate full accuracy (\\%, mean $\\pm$ std over {ns} seeds) on the "
         f"$(d,K)$ grid ({sampler}, $T={T_true}$), best anchor budget per cell.}}",
         f"\\label{{tab:atlas-{sampler}}}",
         f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         "$K$ & $d$ & " + " & ".join(models) + " \\\\", "\\midrule"]
    prev_K = None
    for (K, d, vals) in rows:
        if prev_K is not None and K != prev_K:
            L.append("\\midrule")
        prev_K = K
        cells = " & ".join("--" if mu != mu else f"{mu * 100:.1f} $\\pm$ {sd * 100:.1f}"
                           for (mu, sd) in vals)
        L.append(f"{K} & {d} & {cells} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p = os.path.join(out_dir, "table.tex")
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    return p


def write_summary_png(results, ds, Ks, models, sampler, T_true, out_dir):
    rows = _table_rows(results, ds, Ks, models)
    ns = _n_seeds(results)
    header = ["$K$", "$d$"] + models
    cells = [[str(K), str(d)] + [_fmt(mu, sd) for (mu, sd) in vals] for (K, d, vals) in rows]
    fig, ax = plt.subplots(figsize=(1.6 * len(header) + 1.5, 0.34 * len(cells) + 1.8))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
    for (row, col), cell in tb.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f6f6")
    ax.set_title(f"{sampler}  |  T = {T_true}  |  full accuracy % (mean $\\pm$ std, {ns} seeds), "
                 f"best budget", fontsize=10, pad=12)
    p = os.path.join(out_dir, "table.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run(cfg) -> dict:
    blob, src = _load(cfg)
    results = _rows(blob)
    if not results:
        print("[viz] results.json is empty, nothing to plot")
        return {"figures": []}

    sampler = blob.get("sampler", cfg.process.name)
    T_true = blob.get("config_sweep", {}).get("T_true", "NA")
    ds, Ks = _axes(results)
    models = _models(results)

    # src is already output/<run_id>/<process>/T<T> -- write straight into it (no figures/, no
    # inner T folder; d is a heatmap axis, not a directory).
    made = []

    def attempt(fn, *args):
        try:
            made.append(fn(*args))
        except Exception:
            print(f"[viz] {fn.__name__} failed:")
            traceback.print_exc()

    # table + best-budget heatmaps, directly under the T folder
    attempt(write_summary_tex, results, ds, Ks, models, sampler, T_true, src)
    attempt(write_summary_png, results, ds, Ks, models, sampler, T_true, src)
    for m in models:
        attempt(plot_model, _cellmap(results, m), ds, Ks, m,
                f"{m} (% correct, mean$\\pm$std over {_n_seeds(results)} seeds, best anchor budget)", src)

    # one folder per anchor budget, beside the table
    for b in _budgets(results):
        adir = os.path.join(src, f"anchors_{b}")
        os.makedirs(adir, exist_ok=True)
        rows_b = [r for r in results if int(r["n_per_mode"]) == b]
        for m in models:
            attempt(plot_model, _cellmap(rows_b, m), ds, Ks, m,
                    f"{m} (% correct, mean$\\pm$std over {_n_seeds(results)} seeds, {b} anchors/mode)", adir)

    for p in made:
        print(f"[viz] wrote {p}")
    return {"figures": made}
