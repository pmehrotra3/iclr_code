"""make_animations.py — 2-D animations of the Gaussian-mode experiment (d = 2, K = 4 by default).

Three animations per process (ddim, flow) and variant, from the trained checkpoints:

  <proc>_learned.gif   data -> noise, the LEARNED network: 5 000 seeds x_T ~ N(0, I) are sampled to
                       x_0, then their paths are played back from data to noise. Every point is
                       coloured by where it landed (its mode, or black x = hallucination), so each
                       hallucination can be followed from x_0 back to the seed it came from. Left
                       panel: the samples x_0 in data space by fate.
  <proc>_exact.gif     the same, with the EXACT (true) field instead of the network.
  <proc>_anchors.gif   data -> noise, the anchors: points uniform in each mode's R99 ball plus the
                       band just outside it (hallucination class), placed as evaluate.py does
                       (core.ball_anchors) and backtracked with the exact field from the balls out
                       to their seeds. Left panel: the anchors in data space.

Both fields use the learned sampler's T (T_train), so the pictures are directly comparable.
--forward plays the two seed animations the other way (noise -> data).

    python animations/make_animations.py                        # ddim + flow, both variants, K=4, d=2
    python animations/make_animations.py --procs ddim --K 8 --n 6000 --frames

Writes animations/out/: one GIF per (variant, process, animation) and, with --frames, every
frame as a PNG (for making a video elsewhere). Runs on the CPU in about a minute per animation.
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

import core                                    # noqa: E402
from processes.factory import make_process     # noqa: E402

HALL = "#111111"                               # hallucination colour
MODE_COLORS = plt.get_cmap("tab10").colors
LIM = 3.6                                      # axis half-width: covers N(0, I) and the modes


# ---- loading ------------------------------------------------------------------------------------
def load(proc_name, variant, d, K, seed):
    path = os.path.join(ROOT, "checkpoints", proc_name, variant, "checkpoints", f"model_d{d}_K{K}_s{seed}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    model = core.ScoreNet(ck["d"], a["h"], a["nb"], a["td"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cfg = OmegaConf.create(ck["config"])
    w = ck.get("weights")
    learned = make_process(proc_name, ck["means"], ck["variance"], ck["T"], "cpu", cfg, w)
    exact = make_process(proc_name, ck["means"], ck["variance"], ck["T"], "cpu", cfg, w)
    return model, ck, learned, exact


# ---- trajectories: a list of (time label, points) ------------------------------------------------
@torch.no_grad()
def learned_path(proc, model, X, every):
    """Seeds -> data under the learned network, recording every `every` steps."""
    T, out = proc.T, [("noise", X.clone())]
    if proc.name == "ddim":
        for n, i in enumerate(reversed(range(1, T))):
            eps = model(X, torch.full((X.shape[0],), i, dtype=torch.long))
            X = core._ddim_step(X, eps, proc.abar[i], proc.abar[i - 1])
            if (n + 1) % every == 0 or i == 1:
                out.append((f"DDIM step {i - 1} / {T - 1}  (0 = data)", X.clone()))
    else:
        def f(x, t):
            idx = max(0, min(T - 1, int(round(float(t) * (T - 1)))))
            return model(x, torch.full((x.shape[0],), idx, dtype=torch.long))
        dt = 1.0 / (T - 1)
        for i in range(T - 1):
            X = proc._step(f, X, float(proc.ts[i]), dt, proc.solver)
            if (i + 1) % every == 0 or i == T - 2:
                out.append((f"flow time t = {float(proc.ts[i + 1]):.2f}  (1 = data)", X.clone()))
    return out


@torch.no_grad()
def exact_path(proc, X, every):
    """Seeds -> data under the exact field, recording every `every` steps."""
    T, out = proc.T, [("noise", X.clone())]
    if proc.name == "ddim":
        for n, i in enumerate(reversed(range(1, T))):
            X = core._ddim_transport(X, proc.means_t, proc.true_order, [(proc.abar[i], proc.abar[i - 1])],
                                     proc.variance, logw=proc.logw)
            if (n + 1) % every == 0 or i == 1:
                out.append((f"DDIM step {i - 1} / {T - 1}  (0 = data)", X.clone()))
    else:
        dt = 1.0 / (T - 1)
        for i in range(T - 1):
            X = proc._step(proc._true_velocity, X, float(proc.ts[i]), dt, proc.true_solver)
            if (i + 1) % every == 0 or i == T - 2:
                out.append((f"flow time t = {float(proc.ts[i + 1]):.2f}  (1 = data)", X.clone()))
    return out


@torch.no_grad()
def backtrack_path(proc, X, every):
    """Data -> seeds under the exact field (as evaluate.py carries the anchors), recording."""
    T, out = proc.T, [("data (where the anchors are placed)", X.clone())]
    if proc.name == "ddim":
        for n, i in enumerate(range(1, T)):
            X = core._ddim_transport(X, proc.means_t, proc.true_order, [(proc.abar[i - 1], proc.abar[i])],
                                     proc.variance, logw=proc.logw)
            if (n + 1) % every == 0 or i == T - 1:
                out.append((f"DDIM step {i} / {T - 1}  ({T - 1} = noise)", X.clone()))
    else:
        dt = 1.0 / (T - 1)
        for n, i in enumerate(reversed(range(1, T))):
            X = proc._step(proc._true_velocity, X, float(proc.ts[i]), -dt, proc.true_solver)
            if (n + 1) % every == 0 or i == 1:
                out.append((f"flow time t = {float(proc.ts[i - 1]):.2f}  (0 = noise)", X.clone()))
    return out


# ---- drawing --------------------------------------------------------------------------------------
def colours(labels):
    lab = labels.numpy()
    return np.array([HALL if l < 0 else matplotlib.colors.to_hex(MODE_COLORS[l % 10]) for l in lab])


def style(ax, title):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def mode_circles(ax, means, R99, **kw):
    for k, m in enumerate(means.numpy()):
        ax.add_patch(plt.Circle(m, R99, fill=False, lw=1, color=MODE_COLORS[k % 10], **kw))


def noise_circles(ax):
    for r in (1, 2, 3):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.6, ls=":", color="0.6"))


def animate(path, labels, left, left_title, right_title, means, R99, out, fps, frames_dir, what):
    """left: static (points, labels) panel; right: `path` moving, coloured by `labels`."""
    col, hall = colours(labels), (labels < 0).numpy()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 5.4))
    lp, ll = left
    lc, lh = colours(ll), (ll < 0).numpy()
    axl.scatter(*lp[~lh].T.numpy(), s=3, c=lc[~lh], alpha=0.6, lw=0)
    axl.scatter(*lp[lh].T.numpy(), s=14, c=HALL, marker="x", lw=0.8)
    style(axl, left_title)
    noise_circles(axl) if "noise" in left_title else mode_circles(axl, means, R99, alpha=0.8)

    style(axr, right_title)
    noise_circles(axr)
    mode_circles(axr, means, R99, alpha=0.35)
    P0 = path[0][1].numpy()
    body = axr.scatter(P0[~hall, 0], P0[~hall, 1], s=3, c=col[~hall], alpha=0.6, lw=0)
    halls = axr.scatter(P0[hall, 0], P0[hall, 1], s=16, c=HALL, marker="x", lw=0.9, zorder=3)
    stamp = axr.text(0.02, 0.02, "", transform=axr.transAxes, fontsize=9, color="0.25")
    fig.suptitle(f"{int(hall.sum())} of {len(hall)} {what} (black ×)", fontsize=10)
    fig.tight_layout()
    stack = np.stack([p.numpy() for _, p in path])            # (frames, n, 2)
    hold = fps * 2                                            # rest on the last frame for 2 s
    n_frames = len(path) + hold

    def draw(f):
        f = min(f, len(path) - 1)
        P = stack[f]
        body.set_offsets(P[~hall]); halls.set_offsets(P[hall])
        stamp.set_text(path[f][0])
        return [body, halls, stamp]

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=fps), dpi=90)
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for f in range(len(path)):
            draw(f); fig.savefig(os.path.join(frames_dir, f"{f:04d}.png"), dpi=110)
    plt.close(fig)
    print(f"[anim] wrote {out}.gif ({len(path)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", nargs="+", default=["ddim", "flow"])
    ap.add_argument("--variants", nargs="+", default=["unweighted", "weighted"])
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="which trained model (0, 100, 200)")
    ap.add_argument("--n", type=int, default=5000, help="seeds in the learned-sampler animation")
    ap.add_argument("--anchors", type=int, default=600, help="anchors per mode in the ball (+ half as many in the band)")
    ap.add_argument("--every", type=int, default=5, help="record a frame every this many steps")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frames", action="store_true", help="also save every frame as a PNG")
    ap.add_argument("--forward", action="store_true", help="seed animations noise -> data instead of data -> noise")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    assert args.d == 2, "these are 2-D pictures"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    for variant in args.variants:
        for name in args.procs:
            render(args, name, variant)


def render(args, name, variant):
    """The three animations of one process and variant."""
    model, ck, learned, exact = load(name, variant, args.d, args.K, args.seed)
    means, R99, sigma = ck["means"], ck["R99"], ck["variance"] ** 0.5
    tag = f"{name}_K{args.K}_d{args.d}_s{args.seed}_{variant}"
    pretty = "DDIM" if name == "ddim" else "Flow matching"

    # 1) learned network, 2) exact (true) field: seeds sampled to x_0, every point coloured by its
    #    fate; played back from the samples in data space to the seeds they came from
    X = learned.seeds(args.n, args.d, args.seed + 1)
    for kind, path, field in (("learned", learned_path(learned, model, X, args.every), "learned network"),
                              ("exact", exact_path(exact, X, args.every), "exact (true) field")):
        fate = core.label_fate(path[-1][1], means, R99)
        if args.forward:
            left, left_title, arrow = (X, fate), "seeds x_T in noise space, coloured by fate", "noise → data"
        else:
            left, left_title, arrow = (path[-1][1], fate), "samples x_0 in data space, coloured by fate", "data → noise"
            path = path[::-1]
        animate(path, fate, left, left_title, f"{pretty}, {field}: {arrow}", means, R99,
                os.path.join(args.out, f"{tag}_{kind}"), args.fps,
                os.path.join(args.out, "frames", f"{tag}_{kind}") if args.frames else None,
                what="samples are hallucinations")

    # 3) anchors: uniform in each R99 ball + the band outside, backtracked with the exact field
    #    from the balls out to their seeds in noise (the direction evaluate.py carries them)
    P, y = core.ball_anchors(means, R99, args.anchors, 0.5, 2.0, sigma, seed=0)
    path = backtrack_path(exact, P, args.every)
    animate(path, y, (P, y), "anchors in data space (ball = mode, band = hallucination)",
            f"{pretty}, exact field: anchors data → noise", means, R99,
            os.path.join(args.out, f"{tag}_anchors"), args.fps,
            os.path.join(args.out, "frames", f"{tag}_anchors") if args.frames else None,
            what="anchors are in the hallucination band")

if __name__ == "__main__":
    main()
