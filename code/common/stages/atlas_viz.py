"""stages/atlas_viz.py — figures for the ring-atlas T sweep.

Reads output/<run_tag>/<process>/atlas_results.json and writes, under
visualization/<run_tag>/<process>/:
  T_<T>/heatmap_full_acc     overall accuracy of the primary predictor over the (d, K) grid
  T_<T>/heatmap_hall_f1      hallucination F1 (one-vs-rest F1 of the hallucination class)
  T_<T>/heatmap_mode_f1      mode-basin F1 (macro F1 over the K mode classes)
  T_<T>/heatmap_roundtrip    fraction of anchors that return to their own label (sanity)
  T_<T>/anchors_K{K}         d = 2: the anchors in data space (left) and, after backtracking, in
                             seed space over the learned sampler's fate map (right)
  summary_vs_T               mean over cells of the three metrics vs T, every predictor
  table_atlas.tex            primary predictor at every T: mean full acc / mode F1 / hall F1
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch

from common import checkpoint, utils
from common.process import make_process
from common.stages.visualize import heatmap_grid, _save, _new_fig, SERIES, MARKERS, INK2
from common.stages.atlas import run_dir, anchors_dir


def plot_anchors_d2(cfg, T, Ks, tdir):
    """d = 2: anchors in data space and, backtracked, in seed space over the learned fate map."""
    pname, made = cfg.process, []
    device = utils.get_device(cfg.device)
    n, L = 400, 3.5
    g = torch.linspace(-L, L, n, device=device)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    G = torch.stack([gx.ravel(), gy.ravel()], 1)
    for K in Ks:
        f = os.path.join(anchors_dir(cfg), f"d2_K{K}_T{T}.npz")
        ck_path = utils.ckpt_path(cfg.paths.checkpoints, pname, 2, K, int(cfg.sweep.T_train))
        if not (os.path.exists(f) and os.path.exists(ck_path)):
            continue
        z = np.load(f)
        P, y, A, M, R99 = z["P"], z["y"], z["A"], z["means"], float(z["R99"])
        model, ck = checkpoint.load(ck_path, device)
        proc = make_process(pname, ck["means"], ck["variance"], ck["T"], device, cfg)
        fate_map = proc.label(model, G, ck["R99"]).cpu().numpy().reshape(n, n)
        cmap = plt.get_cmap("tab20", K)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        ax = axes[0]
        ax.scatter(P[y < 0, 0], P[y < 0, 1], s=3, c="k", label="shell (hallucination)")
        ax.scatter(P[y >= 0, 0], P[y >= 0, 1], s=3, c=y[y >= 0], cmap=cmap, vmin=-0.5, vmax=K - 0.5)
        for k in range(K):
            ax.add_patch(plt.Circle(M[k], R99, fill=False, lw=0.8, color="k"))
        ax.set_aspect("equal"); ax.grid(False); ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"anchors in DATA space (K={K}): {int((y >= 0).sum())} disk + {int((y < 0).sum())} shell", fontsize=9)
        ax = axes[1]
        ext = [-L, L, -L, L]
        ax.imshow(np.where(fate_map < 0, np.nan, fate_map), origin="lower", extent=ext, cmap=cmap,
                  vmin=-0.5, vmax=K - 0.5, interpolation="nearest", alpha=0.35)
        ax.imshow(np.where(fate_map < 0, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r",
                  vmin=0, vmax=1, interpolation="nearest", alpha=0.35)
        ax.scatter(A[y < 0, 0], A[y < 0, 1], s=4, c="k")
        ax.scatter(A[y >= 0, 0], A[y >= 0, 1], s=4, c=y[y >= 0], cmap=cmap, vmin=-0.5, vmax=K - 0.5, edgecolors="none")
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(f"backtracked to SEED space (true score, T={T}) over the learned fate map", fontsize=9)
        fig.tight_layout()
        made += _save(fig, tdir, f"anchors_K{K}")
    return made


def run(cfg):
    pname = cfg.process
    with open(os.path.join(run_dir(cfg), "atlas_results.json")) as f:
        blob = json.load(f)
    cells = blob["cells"]
    ds = sorted({c["d"] for c in cells}); Ks = sorted({c["K"] for c in cells})
    Ts = sorted({r["T"] for c in cells for r in c["per_T"]})
    models = []
    for c in cells:
        for r in c["per_T"]:
            for m in r["classifiers"]:
                if m["name"] not in models:
                    models.append(m["name"])
    primary = cfg.classifier.primary if cfg.classifier.primary in models else models[0]
    has_an = all("analytic" in r for c in cells for r in c["per_T"])
    viz_root = os.path.join(cfg.paths.viz, cfg.run_tag, pname)
    made = []

    def metric(c, T, name, key):
        r = next((r for r in c["per_T"] if r["T"] == T), None)
        if r is None:
            return np.nan
        if name == "analytic":
            return r["analytic"][key] if "analytic" in r else np.nan
        if name == "roundtrip":
            return r.get("roundtrip_acc", np.nan)
        m = next((m for m in r["classifiers"] if m["name"] == name), None)
        return m[key] if m else np.nan

    grid = lambda T, name, key: np.array([[metric(next(c for c in cells if c["d"] == d and c["K"] == K), T, name, key)
                                          if any(c["d"] == d and c["K"] == K for c in cells) else np.nan
                                          for d in ds] for K in Ks])

    n_pm = int(cfg.anchors.n_per_mode); n_sh = int(round(float(cfg.anchors.shell_frac) * n_pm))
    # per-T heatmaps
    for T in Ts:
        tdir = os.path.join(viz_root, f"T_{T}")
        os.makedirs(tdir, exist_ok=True)
        sub = f"{primary}, true-score T={T}, anchors/mode: {n_pm} disk + {n_sh} shell"
        for key, title in [("full_acc", "overall accuracy (%)"), ("hall_f1", "hallucination $F_1$ (%)"),
                           ("mode_f1", "mode-basin $F_1$ (%)")]:
            made += heatmap_grid(grid(T, primary, key), ds, Ks, f"{title}\n{sub}", tdir, f"heatmap_{key}")
        if cfg.anchors.roundtrip:
            made += heatmap_grid(grid(T, "roundtrip", None), ds, Ks, f"anchor round-trip accuracy (%), T={T}", tdir, "heatmap_roundtrip")
        if 2 in ds:
            made += plot_anchors_d2(cfg, T, [K for K in Ks if any(c["d"] == 2 and c["K"] == K for c in cells)], tdir)

    # summary vs T
    fig, axes = plt.subplots(1, 3, figsize=(3 * 3.4, 2.5))
    for ax, (key, lab) in zip(axes, [("full_acc", "overall accuracy"), ("mode_f1", "mode-basin $F_1$"), ("hall_f1", "hallucination $F_1$")]):
        for i, name in enumerate(models + (["analytic"] if has_an else [])):
            ys = [np.nanmean(grid(T, name, key)) for T in Ts]
            ax.plot(Ts, ys, marker=MARKERS[i % 10], color=SERIES[i % 10] if name != "analytic" else INK2,
                    ls="--" if name == "analytic" else "-", label=name)
        ax.set_xlabel("true-score steps $T$"); ax.set_ylabel(lab + " (mean over cells)"); ax.set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"ring atlas, {pname}: predictor fit on true-score anchors, scored on the learned sampler",
                 fontsize=8, color=INK2)
    fig.tight_layout()
    made += _save(fig, viz_root, "summary_vs_T")

    # table
    lines = ["% Auto-generated by common/stages/atlas_viz.py -- requires \\usepackage{booktabs}",
             "\\begin{table}[t]", "\\centering",
             f"\\caption{{Ring atlas ({pname}): predictor \\emph{{{primary}}} fit on true-score anchors backtracked "
             "with $T$ steps, scored on the learned sampler's fate; mean over $(d, K)$ cells.}",
             f"\\label{{tab:atlas-T-{pname}}}", "\\begin{tabular}{r rrr" + (" r" if has_an else "") + "}", "\\toprule",
             "$T$ & Overall acc. & Mode $F_1$ & Halluc.\\ $F_1$" + (" & Analytic acc." if has_an else "") + " \\\\", "\\midrule"]
    for T in Ts:
        vals = [np.nanmean(grid(T, primary, k)) * 100 for k in ("full_acc", "mode_f1", "hall_f1")]
        an = f" & {np.nanmean(grid(T, 'analytic', 'full_acc')) * 100:.1f}" if has_an else ""
        lines.append(f"{T} & " + " & ".join(f"{v:.1f}" for v in vals) + an + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p = os.path.join(viz_root, "table_atlas.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    made.append(p)
    print(f"[atlas_viz:{pname}] wrote {len(made)} files under {viz_root}")
    return {"figures": made}
