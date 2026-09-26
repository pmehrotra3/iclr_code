"""make_animations.py — 2-D animations of the Gaussian-mode experiment (d = 2, K = 4 by default).

Three animations per process (ddim, flow) and variant, from the trained checkpoints:

  <proc>_learned.gif   the LEARNED score: seeds x_T ~ N(0, I) are sampled to x_0, then their paths
                       are played back from the generated samples x_0 to the noise x_T they came
                       from. Every point keeps the colour of the mode it lands in; hallucinations
                       (outside every mode's 99% circle) are red x, so each can be followed back to
                       its seed (the basin boundaries, plus a few far-out seeds). Hallucinations are
                       ~1% of samples, so they are drawn from a large pool (--pool) and shown in full
                       (up to --max-hall) beside a random subset of the other samples (--show); the
                       title states both counts. Left panel: the generated samples x_0.
  <proc>_true.gif      the same, with the TRUE score instead of the learned one.
  <proc>_anchors.gif   the anchor points (the set kNN, quadratic and polar are fit on, core.ball_anchors):
                       uniform inside each mode's 99% circle (labelled with the mode) plus the shell
                       just outside it, R99 .. R99 + 2 sigma (labelled hallucination), pulled back
                       with the true score from data space to noise. Left panel: the anchor points
                       where they are placed.

Conventions shared by every animation (and meant to match the paper): hallucination = red x, modes
= the colours in MODE_COLORS (none of them red), step 0 = data end, step T-1 = noise x_T, for
DDIM and flow alike. Both scores use the learned sampler's T (T_train). --forward plays the two
seed animations the other way (noise -> data).

    python animations/make_animations.py                        # ddim + flow, both variants, K=4, d=2
    python animations/make_animations.py --procs ddim --K 8 --frames

Writes animations/out/: one GIF per (variant, process, animation) and, with --frames, every
frame as a PNG (for making a video elsewhere). CPU only, about 2 minutes per process and variant.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

import core                                                        # noqa: E402
from processes.factory import checkpoint_process, make_process     # noqa: E402
from processes.pf_ode import PFODEProcess                          # noqa: E402

PRETTY = {"ddim": "DDIM", "flow": "Flow matching", "heun": "Heun", "rk45": "RK45",
          "dpmpp2m": "DPM-Solver++(2M)"}
STEP = {"ddim": "DDIM step", "flow": "flow step", "heun": "Heun step", "rk45": "RK45 step",
        "dpmpp2m": "DPM-Solver++ step"}

# ---- one visual language for every figure --------------------------------------------------------
HALL = "#D62728"                               # hallucination: red, and only hallucinations are red
MODE_COLORS = ["#0072B2", "#009E73", "#9467BD", "#E69F00",      # modes: no red, colour-blind safe
               "#56B4E9", "#8C564B", "#7F7F7F", "#BCBD22"]
DOT, DOT_A = 7, 0.65                           # mode points: marker size, alpha
X_S, X_LW = 46, 1.5                            # hallucination x: marker size, line width
ANCHOR = "anchor points"                       # the method's name for the labelled points
LIM = 3.6                                      # axis half-width: covers N(0, I) and the modes


def mode_colour(k):
    return MODE_COLORS[k % len(MODE_COLORS)]


# ---- loading ------------------------------------------------------------------------------------
def load(proc_name, variant, d, K, seed):
    path = os.path.join(ROOT, "checkpoints", checkpoint_process(proc_name), variant, "checkpoints",
                        f"model_d{d}_K{K}_s{seed}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    model = core.ScoreNet(ck["d"], a["h"], a["nb"], a["td"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cfg = OmegaConf.create(ck["config"])
    proc = make_process(proc_name, ck["means"], ck["variance"], ck["T"], "cpu", cfg, ck.get("weights"))
    return model, ck, cfg, proc


# ---- trajectories: a list of (s, points), s = steps from the data end (0 = data, T-1 = noise) -----
class _Recorder(torch.nn.Module):
    """Wraps the network and keeps, per grid index j, the points of the LAST call at exactly j.
    In heun / rk45 / dpmpp2m that is the solver's state x_j (the calls in between are Heun's
    predictor or RK45's stages at i-1 and fractional times; the next step's first call at j comes
    after them). Index 0 is never a state call: the caller takes x_0 from the solver's output."""
    def __init__(self, model, keep):
        super().__init__()
        self.model, self.keep, self.at = model, keep, {}

    def forward(self, x, t):
        v = float(t[0])
        if v.is_integer() and int(v) in self.keep:
            self.at[int(v)] = x.clone()
        return self.model(x, t)


@torch.no_grad()
def learned_path(proc, model, X, every):
    """Seeds -> data under the learned score, recording every `every` steps."""
    T, out = proc.T, [(proc.T - 1, X.clone())]
    if isinstance(proc, PFODEProcess):            # heun / rk45 / dpmpp2m: the solver's own loop
        keep = {j for j in range(1, T) if j % every == 0 or j == T - 1}
        rec = _Recorder(model, keep)
        X0 = proc._solve(rec, X.clone())          # the whole pool at once: no chunking
        return [(j, rec.at[j]) for j in sorted(rec.at, reverse=True)] + [(0, X0)]
    if proc.name == "ddim":
        for n, i in enumerate(reversed(range(1, T))):
            eps = model(X, torch.full((X.shape[0],), i, dtype=torch.long))
            X = core._ddim_step(X, eps, proc.abar[i], proc.abar[i - 1])
            if (n + 1) % every == 0 or i == 1:
                out.append((i - 1, X.clone()))
    else:
        def f(x, t):
            idx = max(0, min(T - 1, int(round(float(t) * (T - 1)))))
            return model(x, torch.full((x.shape[0],), idx, dtype=torch.long))
        dt = 1.0 / (T - 1)
        for i in range(T - 1):
            X = proc._step(f, X, float(proc.ts[i]), dt, proc.solver)
            if (i + 1) % every == 0 or i == T - 2:
                out.append((T - 2 - i, X.clone()))
    return out


@torch.no_grad()
def true_path(proc, X, every):
    """Seeds -> data under the true score, recording every `every` steps."""
    T, out = proc.T, [(proc.T - 1, X.clone())]
    if proc.name == "ddim":
        for n, i in enumerate(reversed(range(1, T))):
            X = core._ddim_transport(X, proc.means_t, proc.true_order, [(proc.abar[i], proc.abar[i - 1])],
                                     proc.variance, logw=proc.logw)
            if (n + 1) % every == 0 or i == 1:
                out.append((i - 1, X.clone()))
    else:
        dt = 1.0 / (T - 1)
        for i in range(T - 1):
            X = proc._step(proc._true_velocity, X, float(proc.ts[i]), dt, proc.true_solver)
            if (i + 1) % every == 0 or i == T - 2:
                out.append((T - 2 - i, X.clone()))
    return out


@torch.no_grad()
def pullback_path(proc, X, every):
    """Data -> seeds under the true score (as evaluate.py pulls the anchor points back), recording."""
    T, out = proc.T, [(0, X.clone())]
    if proc.name == "ddim":
        for n, i in enumerate(range(1, T)):
            X = core._ddim_transport(X, proc.means_t, proc.true_order, [(proc.abar[i - 1], proc.abar[i])],
                                     proc.variance, logw=proc.logw)
            if (n + 1) % every == 0 or i == T - 1:
                out.append((i, X.clone()))
    else:
        dt = 1.0 / (T - 1)
        for n, i in enumerate(reversed(range(1, T))):
            X = proc._step(proc._true_velocity, X, float(proc.ts[i]), -dt, proc.true_solver)
            if (n + 1) % every == 0 or i == 1:
                out.append((T - i, X.clone()))
    return out


def pick(fate, n_show, max_hall, seed):
    """Indices to draw: every hallucination (at most max_hall) and n_show random other points."""
    g = torch.Generator().manual_seed(seed)
    hall, other = torch.where(fate < 0)[0], torch.where(fate >= 0)[0]
    hall = hall[torch.randperm(len(hall), generator=g)[:max_hall]]
    other = other[torch.randperm(len(other), generator=g)[:n_show]]
    return torch.cat([other, hall])


# ---- drawing --------------------------------------------------------------------------------------
def style(ax, title, box=((0.0, 0.0), LIM)):
    set_view(ax, *box); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def set_view(ax, c, h):
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)


def data_box(means, R99, shell):
    """A square around the modes, their 99% circles and shells: the data-space view."""
    m = means.numpy()
    c = (m.max(0) + m.min(0)) / 2
    return c, float(np.abs(m - c).max() + R99 + shell) * 1.15


def views(path, box):
    """Per frame (centre, half-width), from the data box at s = 0 to the full noise view
    [-LIM, LIM]^2 at s = T-1: the centre slides linearly in s, the half-width is the larger of the
    linear one and what holds 98% of the points, never shrinking on the way to noise."""
    (c0, h0), T1 = box, max(s for s, _ in path)
    order = sorted(range(len(path)), key=lambda f: path[f][0])      # data end first
    out, h_run = {}, h0
    for f in order:
        s, P = path[f][0], path[f][1].numpy()
        p = s / T1
        c = (1 - p) * np.asarray(c0)
        spread = float(np.quantile(np.abs(P - c).max(1), 0.98)) * 1.08 if len(P) else h0
        h_run = min(LIM, max(h_run, (1 - p) * h0 + p * LIM, spread))
        out[f] = (c, LIM if s == T1 else h_run)
    return [out[f] for f in range(len(path))]


def scatter(ax, P, labels, zorder=2):
    """Mode points as dots in their mode's colour, hallucinations as red x on top."""
    P, lab = P.numpy(), labels.numpy()
    h = lab < 0
    body = ax.scatter(P[~h, 0], P[~h, 1], s=DOT, c=[mode_colour(k) for k in lab[~h]],
                      alpha=DOT_A, lw=0, zorder=zorder)
    halls = ax.scatter(P[h, 0], P[h, 1], s=X_S, c=HALL, marker="x", lw=X_LW, zorder=zorder + 1)
    return body, halls


def mode_regions(ax, means, R99, shell=None, alpha=1.0):
    """Each mode's 99% circle (solid, mode colour) and, if shell, the hallucination shell
    R99 .. R99 + shell shaded red with a dashed outer edge. Returns the patches (to fade them)."""
    patches = []
    for k, m in enumerate(means.numpy()):
        if shell:
            patches.append(Wedge(m, R99 + shell, 0, 360, width=shell, facecolor=HALL, edgecolor="none",
                                 alpha=0.13 * alpha, zorder=1))
            patches.append(Circle(m, R99 + shell, fill=False, lw=1.1, ls="--", color=HALL,
                                  alpha=0.9 * alpha, zorder=1))
        patches.append(Circle(m, R99, fill=False, lw=1.4, color=mode_colour(k), alpha=0.9 * alpha, zorder=1))
    for p in patches:
        p._base_alpha = p.get_alpha()
        ax.add_patch(p)
    return patches


def legend(ax, entries):
    handles = []
    for kind, text in entries:
        if kind == "dot":
            handles.append(Line2D([], [], ls="", marker="o", ms=5, color="0.35", label=text))
        elif kind == "x":
            handles.append(Line2D([], [], ls="", marker="x", ms=7, mew=X_LW, color=HALL, label=text))
        elif kind == "circle":
            handles.append(Line2D([], [], ls="-", lw=1.4, color="0.35", label=text))
        elif kind == "shell":
            handles.append(Patch(facecolor=HALL, alpha=0.25, edgecolor=HALL, ls="--", label=text))
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9,
              handletextpad=0.5, borderpad=0.5)


def animate(path, labels, left, right_title, suptitle, stamp_fmt, T, means, R99, box, out, fps,
            frames_dir, shell=None, entries=()):
    """left: (points, labels, title) drawn still, zoomed on the modes; right: `path` moving, coloured
    by `labels`, the view zooming out from the modes to noise. The mode circles (and shells) on the
    right fade out as the points leave data space."""
    hall = (labels < 0).numpy()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 6.0 + 0.17 * suptitle.count("\n")))
    lp, ll, left_title = left
    style(axl, left_title, box)
    mode_regions(axl, means, R99, shell)
    scatter(axl, lp, ll)
    legend(axl, entries)

    style(axr, right_title)
    view = views(path, box)
    fading = mode_regions(axr, means, R99, shell, alpha=0.7)
    body, halls = scatter(axr, path[0][1], labels)
    stamp = axr.text(0.02, 0.02, "", transform=axr.transAxes, fontsize=9, color="0.2",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    stack = np.stack([p.numpy() for _, p in path])            # (frames, n, 2)
    lead, hold = fps, fps * 2                                 # rest 1 s on the first frame, 2 s on the last
    n_frames = lead + len(path) + hold

    def draw(f):
        f = min(max(f - lead, 0), len(path) - 1)
        s, P = path[f][0], stack[f]
        body.set_offsets(P[~hall]); halls.set_offsets(P[hall])
        stamp.set_text(stamp_fmt.format(s=s))
        set_view(axr, *view[f])
        for p in fading:
            p.set_alpha(p._base_alpha * (1 - s / (T - 1)))
        return [body, halls, stamp, *fading]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=fps), dpi=90)
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for f in range(len(path)):
            draw(f + lead); fig.savefig(os.path.join(frames_dir, f"{f:04d}.png"), dpi=110)
    plt.close(fig)
    print(f"[anim] wrote {out}.gif ({len(path)} frames)")


def best_budget(name, variant, d, K, T):
    """Anchor points per mode of the best row (overall accuracy, any predictor) for (d, K) in
    output/abc123/<name>/<variant>/T<T>/results.json, or None if there is no result yet."""
    f = os.path.join(ROOT, "output", "abc123", name, variant, f"T{T}", "results.json")
    if not os.path.exists(f):
        return None
    import json
    rows = [r for r in json.load(open(f))["aggregate"] if r["d"] == d and r["K"] == K]
    return max(rows, key=lambda r: r["full_acc"]) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", nargs="+", default=["ddim", "flow", "heun", "rk45", "dpmpp2m"])
    ap.add_argument("--variants", nargs="+", default=["unweighted", "weighted"])
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="which trained model (0, 100, 200)")
    ap.add_argument("--pool", type=int, default=80000, help="seeds sampled (hallucinations are ~1%% of them)")
    ap.add_argument("--max-hall", type=int, default=1000, help="hallucinations shown at most")
    ap.add_argument("--show", type=int, default=4000, help="other (mode) samples shown")
    ap.add_argument("--anchors", type=int, default=None,
                    help="anchor points per mode inside its 99%% circle (+ half as many in its shell); "
                         "default: the best budget in output/abc123 for this (d, K), else 20000")
    ap.add_argument("--show-anchors", type=int, default=400,
                    help="anchor points shown per mode inside its circle (+ half as many in its shell)")
    ap.add_argument("--every", type=int, default=5, help="record a frame every this many steps")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frames", action="store_true", help="also save every frame as a PNG")
    ap.add_argument("--forward", action="store_true", help="seed animations noise -> data instead of data -> noise")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    assert args.d == 2, "these are 2-D pictures"
    torch.manual_seed(0)

    for name in args.procs:
        for variant in args.variants:
            render(args, name, variant)


def render(args, name, variant):
    """One sampler and variant, into out/<sampler>/<variant>/: the learned-score animation, and for
    ddim and flow also the true-score and anchor-point ones (heun / rk45 / dpmpp2m integrate DDIM's
    network, and their true score and anchor points are DDIM's, so those two would repeat DDIM's)."""
    model, ck, cfg, proc = load(name, variant, args.d, args.K, args.seed)
    means, R99, sigma, T = ck["means"], ck["R99"], ck["variance"] ** 0.5, proc.T
    shell_sigma = float(cfg.anchors.shell_sigma)
    shell = shell_sigma * sigma
    box = data_box(means, R99, shell)
    folder = os.path.join(args.out, name, variant)
    tag = f"{name}_{variant}_K{args.K}_d{args.d}"
    pretty, step = PRETTY[name], STEP[name]
    frames = lambda kind: os.path.join(folder, "frames", kind) if args.frames else None
    seed_entries = [("dot", "generated sample inside a mode's 99% circle\n(colour = that mode)"),
                    ("x", "hallucination: outside every 99% circle"),
                    ("circle", "99% circle of a mode (radius $R_{99}$)")]

    # 1) learned score, 2) true score: a pool of seeds sampled to x_0; every hallucination and a random
    #    subset of the rest are shown, played back from the generated samples to their seeds
    X = proc.seeds(args.pool, args.d, args.seed + 1)
    kinds = [("learned", lambda: learned_path(proc, model, X, args.every), "learned score")]
    if name in ("ddim", "flow"):
        kinds.append(("true", lambda: true_path(proc, X, args.every), "true score"))
    for kind, run, score in kinds:
        path = run()
        fate = core.label_fate(path[-1][1], means, R99)
        idx = pick(fate, args.show, args.max_hall, args.seed)
        path, lab = [(s, P[idx]) for s, P in path], fate[idx]
        n_h, n_hs = int((fate < 0).sum()), int((lab < 0).sum())
        suptitle = (f"{pretty}, {score} ({variant}, K = {args.K}, d = {args.d}): "
                    f"{n_h:,} of {args.pool:,} generated samples ({n_h / args.pool:.1%}) are hallucinations\n"
                    f"shown: {'all ' if n_hs == n_h else ''}{n_hs:,} hallucinations (red ×) "
                    f"and {len(lab) - n_hs:,} of the other samples, at random")
        if args.forward:
            right = f"{score}: noise $x_T$ → generated sample $x_0$"
        else:
            right, path = f"{score}: generated sample $x_0$ → back to its noise $x_T$", path[::-1]
        animate(path, lab, (path[0 if not args.forward else -1][1], lab, "generated samples $x_0$ (data space)"),
                right, suptitle,
                f"{step} {{s}} / {T - 1}\n0 = generated sample $x_0$, {T - 1} = noise $x_T$",
                T, means, R99, box, os.path.join(folder, f"{tag}_{kind}"), args.fps, frames(kind),
                entries=seed_entries)
    if name not in ("ddim", "flow"):
        return

    # 3) anchor points: the best budget's set (uniform in each 99% circle + the shell outside it, as
    #    evaluate.py places it), of which a random subset is shown, pulled back with the true score
    #    from data space to noise (each point's path is its own, so the subset's paths are exact)
    best = best_budget(name, variant, args.d, args.K, T)
    budget = args.anchors or (best["n_per_mode"] if best else 20000)
    P, y = core.ball_anchors(means, R99, budget, float(cfg.anchors.shell_frac), shell_sigma, sigma,
                             seed=int(cfg.anchors.seed))
    idx = pick(y, args.show_anchors * args.K, args.show_anchors * args.K // 2, args.seed)
    P, y = P[idx], y[idx]
    n_h = int((y < 0).sum())
    model_name = {"knn": "kNN", "altered_knn": "altered kNN", "polar3": "polar"}
    why = (f" (the best budget for d = {args.d}, K = {args.K}: {model_name.get(best['model'], best['model'])} "
           f"reaches {100 * best['full_acc']:.1f}% overall accuracy with it)" if best and not args.anchors else "")
    suptitle = (f"{pretty} ({variant}, K = {args.K}, d = {args.d}): {ANCHOR} are labelled points placed around each mode "
                f"in data space,\npulled back to noise $x_T$ with the true score; a new seed is classified by the "
                f"{ANCHOR} near it.\n{budget:,} {ANCHOR} per mode{why}.\nShown at random: "
                f"{len(y) - n_h:,} inside the 99% circles and {n_h:,} in the hallucination shells (red ×).")
    animate(pullback_path(proc, P, args.every), y, (P, y, f"{ANCHOR} where they are placed (data space)"),
            f"true score: {ANCHOR} pulled back to noise $x_T$", suptitle,
            f"{step} {{s}} / {T - 1}\n0 = {ANCHOR} in data space, {T - 1} = noise $x_T$",
            T, means, R99, box, os.path.join(folder, f"{tag}_anchors"), args.fps, frames("anchors"),
            shell=shell,
            entries=[("dot", f"{ANCHOR} inside a mode's 99% circle\n→ labelled with that mode"),
                     ("x", f"{ANCHOR} in the shell just outside\n($R_{{99}}$ to $R_{{99}} + {shell_sigma:g}\\sigma$) → labelled hallucination"),
                     ("circle", "99% circle of a mode (radius $R_{99}$)"),
                     ("shell", "hallucination shell")])


if __name__ == "__main__":
    main()
