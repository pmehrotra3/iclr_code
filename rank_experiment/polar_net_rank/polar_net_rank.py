#!/usr/bin/env python
"""polar_net_rank.py -- rank of the polar network's r projections on r points of one sphere.

The polar classifier (code/fate.py: FateNet, arch "polar") first takes r = hidden = 256 linear
projections of a seed x in noise space,

    z_j(x) = (w_j . [x / |x| sqrt(mu), |x| - mu] + b_j) / sqrt(mu + 1),    mu = sqrt(d),

and only then applies its nonlinearity, the powers (z, z^2, z^3). For each process (DDIM, flow
matching) and every (d, K) cell of the grid in config.yaml:

  1. Build the Gaussian mixture of the main experiments (first run, seed 0) and its polar
     network with code/'s own FateNet: at initialisation, and trained as in the main
     experiments (ensemble member 0, on b anchors per mode pulled back by the exact field).
  2. Take the sphere of radius rho R99 around one mode in data space, draw r distinct points
     on it and pull them back to noise space with the exact map (the same map that pulls back
     the anchors): r distinct points, all on ONE of the pulled-back concentric spheres around
     that mode.
  3. Send the SAME r points through all r projections: the r x r matrix H[i, j] = z_j(x_i),
     before the nonlinearity, and its rank. The r columns are affine in d + 1 inputs
     (direction and radius), so rank H <= d + 2, below r whenever d + 2 < r. For comparison,
     the rank of [H, H^2, H^3] (r x 3r), after the nonlinearity.

Outputs (under run.out_dir), one folder per process:

    ddim/  rank.csv  rank.tex  rank.png
    flow/  (the same)

    python rank_experiment/polar_net_rank/polar_net_rank.py --config rank_experiment/polar_net_rank/config.yaml
                                                           [--quick] [--tables-only] [--process ddim|flow]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                  # rank_sweep.py: config, device, headings
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))    # the repo's mixture, maps and polar network
import core  # noqa: E402
import fate  # noqa: E402
from processes.factory import make_process  # noqa: E402
from rank_sweep import load_config, pick_device, plt, stack  # noqa: E402

PROCESSES = {"ddim": ("DDIM", "the exact-score DDIM map"),
             "flow": ("flow matching", "the exact-velocity flow-matching map")}


class _Attr(dict):
    """A dict read with attributes: the part of a Hydra config that code/processes reads."""
    __getattr__ = dict.get


# ------------------------------------------------------------------ one (d, K) cell -------
def polar_nets(A, y, K, d, pn, device):
    """The polar network of the main experiments, ensemble member pn['member']: at its
    initialisation (fate.train_fate_classifier seeds torch and builds the net in the same
    way) and after training on the pulled-back anchors (A, y)."""
    torch.manual_seed(pn["member"])
    init = fate.FateNet(d, K, "polar", pn["hidden"], pn["degree"]).to(device).eval()
    trained = fate.train_fate_classifier(A, y, K, "polar", hidden=pn["hidden"], degree=pn["degree"],
                                         epochs=pn["epochs"], batch=pn["batch"], lr=pn["lr"],
                                         weight_decay=pn["weight_decay"], min_steps=pn["min_steps"],
                                         seed=pn["member"], device=device)
    return init, trained


def projections(net, X):
    """H[i, j] = z_j(x_i): the r projections of the polar network at the points X, before its
    nonlinearity, in float64 (the algebra of fate.FateNet.forward)."""
    X = X.to(torch.float64)
    mu = math.sqrt(X.shape[1])
    r = X.norm(dim=1, keepdim=True)
    F = torch.cat([X / r * math.sqrt(mu), r - mu], 1)
    W, b = net.proj.weight.detach().cpu().double(), net.proj.bias.detach().cpu().double()
    return (F @ W.T + b) / math.sqrt(mu + 1)


@torch.no_grad()
def network_projections(net, X):
    """The same z, as the network itself computes it (float32), taken at its projection layer."""
    got = {}
    hook = net.proj.register_forward_hook(lambda m, i, o: got.__setitem__("z", o))
    net(X)
    hook.remove()
    return got["z"].cpu().double() / math.sqrt(math.sqrt(X.shape[1]) + 1)


def rank_of(A):
    """Numerical rank (singular values above s_max * max(n, m) * eps, the numpy / torch default)
    and the gap at that cut: largest dropped / smallest kept singular value (0 at full rank)."""
    s = torch.linalg.svdvals(A)
    k = int((s > s[0] * max(A.shape) * torch.finfo(A.dtype).eps).sum())
    return k, (float(s[k] / s[k - 1]) if k < len(s) else 0.0)


def run_cell(d, K, process, cfg, device):
    """The four ranks of one cell under one process."""
    mix, pn, anc, pts = cfg["mixture"], cfg["polar_net"], cfg["anchors"], cfg["points"]
    sigma, T = mix["sigma"], cfg["process"]["T_true"]
    means, _ = core.sample_modes(K, d, mix["radius"] * math.sqrt(d / 2), sigma, mix["m_mult"],
                                 seed=mix["seed"], device=device)                # ModePlacementError
    R99 = core.r99(d, sigma, mix["mass_q"])
    proc = make_process(process, means, sigma ** 2, T, device, _Attr(process=_Attr(cfg["process"][process])))

    t0 = time.time()
    P, y = core.ball_anchors(means, R99, anc["budget"], anc["shell_frac"], anc["shell_sigma"], sigma,
                             anc["seed"], device)                                # as code/evaluate.py
    A = proc.true_field_backtrack(P, inplace=True)
    del P
    init, trained = polar_nets(A, y, K, d, pn, device)
    del A, y
    t_train = time.time() - t0

    n = pts["n"] or pn["hidden"]
    g = torch.Generator().manual_seed(pts["seed"] + 1000 * d + K)                # on the CPU: any device
    dirs = torch.randn(n, d, generator=g)
    S = means[pts["mode"]].cpu() + pts["radius"] * R99 * dirs / dirs.norm(dim=1, keepdim=True)
    X = proc.true_field_backtrack(S.to(device)).cpu()                            # r points, one sphere
    min_dist = float(torch.pdist(X.double()).min())
    if not min_dist > 0:
        raise RuntimeError(f"two of the {n} points coincide (d={d}, K={K})")

    out = {"K": K, "d": d, "r": pn["hidden"], "n": n, "bound": min(n, pn["hidden"], d + 2),
           "b": anc["budget"], "min_dist": min_dist, "train_sec": t_train}
    for tag, net in (("init", init), ("trained", trained)):
        H = projections(net, X)
        Hn = network_projections(net, X.to(device))
        err = float((H - Hn).abs().max() / H.abs().max())
        if err > 1e-4:
            raise RuntimeError(f"float64 projections differ from the network's by {err:.1e} (d={d}, K={K})")
        out[f"lin_{tag}"], out[f"gap_lin_{tag}"] = rank_of(H)
        out[f"pow_{tag}"], out[f"gap_pow_{tag}"] = rank_of(
            torch.cat([H ** q for q in range(1, pn["degree"] + 1)], 1))
    return out


# ------------------------------------------------------------------ the table -------------
# (internal key, csv heading, LaTeX heading or None when the column is only in the csv)
COLUMNS = [("K", "K", "$K$"), ("d", "d", "$d$"),
           ("r", "number of projections r", stack("projections", "$r$")),
           ("n", "number of points", stack("number", "of points")),
           ("bound", "upper bound min(r, d + 2)", stack("upper bound", "$\\min(r, d + 2)$")),
           ("lin_init", "rank before the nonlinearity: at initialisation", stack("at initial-", "isation")),
           ("lin_trained", "rank before the nonlinearity: trained", "trained"),
           ("pow_init", "rank after the nonlinearity: at initialisation", stack("at initial-", "isation")),
           ("pow_trained", "rank after the nonlinearity: trained", "trained"),
           ("b", "anchors per mode (training)", None),
           ("min_dist", "smallest distance between two points (noise space)", None),
           ("gap_lin_init", "singular-value gap at the cut, before, at initialisation", None),
           ("gap_lin_trained", "singular-value gap at the cut, before, trained", None),
           ("gap_pow_init", "singular-value gap at the cut, after, at initialisation", None),
           ("gap_pow_trained", "singular-value gap at the cut, after, trained", None),
           ("train_sec", "seconds to build and train the network", None)]
SHOWN = [c for c in COLUMNS if c[2] is not None]
FLOATS = {"min_dist", "gap_lin_init", "gap_lin_trained", "gap_pow_init", "gap_pow_trained", "train_sec"}


def read_table(path):
    """Rows of a saved rank.csv, keyed by the internal names."""
    if not os.path.exists(path):
        return []
    head = {h: k for k, h, _ in COLUMNS}
    with open(path) as f:
        return [{head[h]: (float(v) if head[h] in FLOATS else int(v)) for h, v in r.items()}
                for r in csv.DictReader(f)]


def write_table(rows, folder, process, cfg):
    """rank.csv, rank.tex and rank.png in folder: one row per (K, d)."""
    rows = sorted(rows, key=lambda r: (r["K"], r["d"]))
    proc, the_map = PROCESSES[process]
    pts = cfg["points"]
    base = os.path.join(folder, "rank")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h, _ in COLUMNS])
        for r in rows:
            w.writerow([(f"{r[k]:.4g}" if k in FLOATS else r[k]) for k, _, _ in COLUMNS])

    rho = f"{pts['radius']:g}\\,R_{{99}}" if pts["radius"] != 1 else "R_{99}"
    tex = ["% Auto-generated by rank_experiment/polar_net_rank/polar_net_rank.py -- needs booktabs and graphicx",
           "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           f"\\caption{{\\textbf{{Projection rank.}} {proc[0].upper() + proc[1:]}. For each $(K, d)$: $r$ points drawn on the "
           f"sphere of radius ${rho}$ around one mode, pulled back to noise space by {the_map}, all distinct "
           "and all on the same pulled-back sphere, go through the $r$ projections "
           "$z_j(x) = (w_j^\\top [x/\\|x\\|\\,\\sqrt{\\mu},\\ \\|x\\| - \\mu] + b_j)/\\sqrt{\\mu + 1}$, "
           "$\\mu = \\sqrt{d}$, of the first linear layer of the polar network: the same points for every "
           "projection. The table gives the rank of the $r \\times r$ matrix $z_j(x_i)$ before the "
           "nonlinearity, and of $[z, z^2, z^3]$ after it, for the network at its initialisation and "
           f"after training as in the main experiments ({cfg['anchors']['budget']:,} anchors per mode). "
           "Before the nonlinearity each column is affine in the direction and the radius, so the rank is "
           "at most $\\min(r, d + 2)$.}",
           f"\\label{{tab:polar-net-rank-{process}}}",
           "% shrinks to the text width only when it is wider",
           "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{rrrrr cc cc}", "\\toprule",
           " & & & & & \\multicolumn{2}{c}{rank before the nonlinearity} "
           "& \\multicolumn{2}{c}{rank after the nonlinearity} \\\\",
           "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
           " & ".join(t for _, _, t in SHOWN) + " \\\\", "\\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["K"] != prev:
            tex.append("\\midrule")
        prev = r["K"]
        tex.append(" & ".join(str(r[k]) for k, _, _ in SHOWN) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(base + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    table_png(rows, proc, cfg, base + ".png")


def table_png(rows, proc, cfg, path):
    """The same table as an image: grouped headings, one band per K."""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]})
    widths = [0.45, 0.55, 0.95, 0.85, 1.25, 1.0, 1.0, 1.0, 1.0]
    row_h, head_h = 0.22, 0.84
    W, H = sum(widths) + 0.2, head_h + row_h * len(rows) + 1.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x0 = [0.1 + sum(widths[:i]) for i in range(len(widths))]
    xc = [x + w / 2 for x, w in zip(x0, widths)]
    ink, rule, band = "#1f1f1f", "#5a5a5a", "#eef3fb"
    ax.text(W / 2, 0.2, "Projection rank", ha="center", va="center", fontsize=12, fontweight="bold",
            color=ink)
    ax.text(W / 2, 0.41, proc[0].upper() + proc[1:], ha="center", va="center", fontsize=9.5, color="#4a4a4a")
    top = 0.55
    ax.plot([0.1, W - 0.1], [top, top], color=rule, lw=1.0)
    for label, lo, hi in (("rank before the nonlinearity", 5, 6), ("rank after the nonlinearity", 7, 8)):
        ax.text((x0[lo] + x0[hi] + widths[hi]) / 2, top + 0.14, label, ha="center", va="center",
                fontsize=9.5, color=ink)
        ax.plot([x0[lo] + 0.06, x0[hi] + widths[hi] - 0.06], [top + 0.27, top + 0.27], color=rule, lw=0.6)
    heads = ["K", "d", "projections\nr", "number\nof points", "upper bound\nmin(r, d + 2)",
             "at initial-\nisation", "trained", "at initial-\nisation", "trained"]
    for x, h in zip(xc, heads):
        ax.text(x, top + head_h - 0.26, h, ha="center", va="center", fontsize=9.5, color=ink,
                style="italic" if h in ("K", "d") else "normal")
    y = top + head_h + 0.04
    ax.plot([0.1, W - 0.1], [y, y], color=rule, lw=0.6)
    Ks = sorted({r["K"] for r in rows})
    for i, r in enumerate(rows):
        yy = y + i * row_h
        if Ks.index(r["K"]) % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, yy), W - 0.2, row_h, color=band, lw=0, zorder=0))
        for x, (k, _, _) in zip(xc, SHOWN):
            ax.text(x, yy + row_h / 2, str(r[k]), ha="center", va="center", fontsize=9, color=ink)
    yb = y + len(rows) * row_h
    ax.plot([0.1, W - 0.1], [yb, yb], color=rule, lw=1.0)
    rho = cfg["points"]["radius"]
    ax.text(0.12, yb + 0.12,
            f"points: r distinct points on the sphere of radius {'' if rho == 1 else f'{rho:g} '}R99 around "
            "one mode, pulled back to noise space: all on the same pulled-back sphere.\n"
            "The same r points go through every projection z_j(x) = (w_j . [x/|x| sqrt(mu), |x| - mu] + b_j) "
            "/ sqrt(mu + 1) of the polar network's first linear layer.\n"
            "before: rank of the r x r matrix z_j(x_i).   after: rank of [z, z^2, z^3].   trained: as in the "
            f"main experiments, {cfg['anchors']['budget']:,} anchors per mode.",
            ha="left", va="top", fontsize=8, color="#4a4a4a", linespacing=1.4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="the small grid of the config's `quick` block")
    ap.add_argument("--tables-only", action="store_true", help="rebuild the .tex and .png from the CSVs")
    ap.add_argument("--process", choices=sorted(PROCESSES), help="run one process (default: those of the config)")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu (overrides the config)")
    ap.add_argument("--d", type=int, nargs="+", help="dimensions to run (overrides the config's grid)")
    ap.add_argument("--K", type=int, nargs="+", help="numbers of modes to run (overrides the config's grid)")
    ap.add_argument("--out-dir", help="results folder (overrides run.out_dir), e.g. one per SLURM task")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    cfg["grid"]["d"] = args.d or cfg["grid"]["d"]
    cfg["grid"]["K"] = args.K or cfg["grid"]["K"]
    cfg["run"]["out_dir"] = args.out_dir or cfg["run"]["out_dir"]
    processes = [args.process] if args.process else list(cfg["processes"])
    grid = [(d, K) for d in cfg["grid"]["d"] for K in cfg["grid"]["K"]]
    b = cfg["anchors"]["budget"]
    folders, results, done = {}, {}, {}
    for p in processes:
        folders[p] = os.path.join(cfg["run"]["out_dir"], p)
        os.makedirs(folders[p], exist_ok=True)
        rows = read_table(os.path.join(folders[p], "rank.csv"))
        if args.tables_only:
            if rows:
                write_table(rows, folders[p], p, cfg)
            print(f"[polar rank] {p}: {len(rows)} rows -> rank.csv, rank.tex, rank.png")
            continue
        # a cell is done once it has a row trained with this run's anchor budget: the full run
        # redoes the quick cells and drops their rows, so a table never mixes the two budgets
        rows = rows if cfg["run"]["resume"] else []
        done[p] = {(r["d"], r["K"]) for r in rows if r["b"] == b}
        results[p] = [r for r in rows if (r["d"], r["K"]) in done[p] or (r["d"], r["K"]) not in set(grid)]
    if args.tables_only:
        return

    device = pick_device(args.device or cfg["run"]["device"])
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("=" * 100)
    print(f"polar-network rank: {len(grid)} cells for each of {', '.join(processes)} "
          f"({', '.join(f'{p}: {len(done[p] & set(grid))} done' for p in processes)}), device={device}, "
          f"{b:,} anchors per mode, results -> {cfg['run']['out_dir']}/<process>")
    print("=" * 100)
    print(f"{'process':>7}{'d':>6}{'K':>4}{'r':>6}{'bound':>7} | {'before: init':>12}{'trained':>9} | "
          f"{'after: init':>12}{'trained':>9} | {'min dist':>9}{'sec':>7}")
    t_all = time.time()
    for d, K in grid:
        for p in processes:
            if (d, K) in done[p]:
                continue
            try:
                row = run_cell(d, K, p, cfg, device)
            except RuntimeError as e:                    # a K that does not fit at this d
                print(f"{p:>7}{d:>6}{K:>4}  skipped: {e}", flush=True)
                continue
            results[p] = [r for r in results[p] if (r["d"], r["K"]) != (d, K)] + [row]
            write_table(results[p], folders[p], p, cfg)
            print(f"{p:>7}{d:>6}{K:>4}{row['r']:>6}{row['bound']:>7} | {row['lin_init']:>12}"
                  f"{row['lin_trained']:>9} | {row['pow_init']:>12}{row['pow_trained']:>9} | "
                  f"{row['min_dist']:>9.2e}{row['train_sec']:>7.0f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\ndone in {time.time() - t_all:.0f}s -> {cfg['run']['out_dir']}/<process>")


if __name__ == "__main__":
    main()
