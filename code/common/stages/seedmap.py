"""stages/seedmap.py — draw the seed -> fate map in d = 2.

One figure per K in cfg.seedmap.K, panels left to right:
  ground truth (learned sampler, or the analytic sampler when eval.labels=true)
  the other sampler, for comparison
  one panel per classifier in cfg.seedmap.models, fit at cfg.seedmap.budget seeds, with
  its mistakes overlaid in red and its Gaussian-weighted accuracy in the title.

Colours = mode reached, black = hallucination. Written to visualization/<process>/.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import fate, checkpoint, utils
from common.process import make_process
from common.stages.evaluate import _spec, _label


def _panel(ax, lab, K, extent, title):
    n = int(np.sqrt(lab.size))
    lab = lab.reshape(n, n)
    ext = [-extent, extent, -extent, extent]
    ax.imshow(np.where(lab < 0, np.nan, lab), origin="lower", extent=ext, interpolation="nearest",
              cmap=plt.get_cmap("tab20", K), vmin=-0.5, vmax=K - 0.5)
    ax.imshow(np.where(lab < 0, 1.0, np.nan), origin="lower", extent=ext, interpolation="nearest",
              cmap="gray_r", vmin=0, vmax=1)
    for r in (1, 2, 3):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.6, color="k"))
    ax.grid(False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("seed $x_1$"); ax.set_ylabel("seed $x_2$")


def run(cfg):
    device = utils.get_device(cfg.device)
    pname, d = cfg.process, 2
    viz_dir = utils.process_dir(cfg.paths.viz, pname)
    os.makedirs(viz_dir, exist_ok=True)
    n, L = int(cfg.seedmap.n_grid), float(cfg.seedmap.extent)
    g = torch.linspace(-L, L, n, device=device)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    G = torch.stack([gx.ravel(), gy.ravel()], 1)
    w = torch.exp(-(G ** 2).sum(1) / 2)                     # Gaussian weight of each grid point
    names = cfg.seedmap.models
    specs = [_spec(cfg, m) for m in cfg.classifier.models
             if names == "all" or m.name in list(names)]
    made = []
    for K in cfg.seedmap.K:
        K = int(K)
        path = utils.ckpt_path(cfg.paths.checkpoints, pname, d, K, int(cfg.sweep.T_train))
        if not os.path.exists(path):
            print(f"[seedmap:{pname}] d=2 K={K}: no checkpoint, skipped"); continue
        model, ck = checkpoint.load(path, device)
        proc = make_process(pname, ck["means"], ck["variance"], ck["T"], device, cfg)
        proc_true = make_process(pname, ck["means"], ck["variance"], cfg.sweep.T_true, device, cfg)
        use_true = cfg.eval.labels == "true"
        gt = (proc_true if use_true else proc).label(None if use_true else model, G, ck["R99"])
        other = (proc if use_true else proc_true).label(model if use_true else None, G, ck["R99"])
        panels = [(f"{pname} d=2 K={K}: " + ("analytic sampler (truth)" if use_true else "learned sampler (truth)"), gt, None),
                  ("learned sampler" if use_true else "analytic sampler", other, None)]
        if specs:
            X, y = _label(proc_true if use_true else proc, None if use_true else model, ck["R99"], d,
                          int(cfg.seedmap.budget), cfg.seed + cfg.eval.seed_offset)
        for spec in specs:
            nets = fate.train_ensemble(X, y, K, spec, device)
            pred = fate.predict_fate(nets, G)
            acc = float(((pred == gt).float() * w).sum() / w.sum())
            panels.append((f"{spec['name']}: {100 * acc:.1f}% (Gaussian-weighted)", pred, (pred != gt)))
            print(f"[seedmap:{pname}] K={K} {spec['name']}: {100 * acc:.2f}%  fit={nets[0].describe()}", flush=True)
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2), squeeze=False)
        for ax, (title, lab, wrong) in zip(axes[0], panels):
            _panel(ax, lab.cpu().numpy(), K, L, title)
            if wrong is not None:
                ax.contourf(g.cpu(), g.cpu(), wrong.cpu().numpy().reshape(n, n).astype(float),
                            levels=[0.5, 1.5], colors=["red"], alpha=0.6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(viz_dir, f"seedmap_K{K}.{ext}")
            fig.savefig(p, dpi=90 if ext == "png" else None, bbox_inches="tight"); made.append(p)
        plt.close(fig)
        print(f"[seedmap:{pname}] wrote {made[-2]}")
    return {"figures": made}
