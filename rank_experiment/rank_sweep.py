#!/usr/bin/env python
"""rank_sweep.py -- do hallucinating points occupy fewer dimensions than Gaussian noise?

For each process (DDIM, flow matching) and every (d, K) cell of the grid in config.yaml:

  1. Build the Gaussian mixture of the main experiments: K centres on a sphere of radius
     2 sqrt(d/2), at least 3 R99 apart, each component N(mu_k, sigma^2 I).
  2. Push fresh seeds z ~ N(0, I) through the process's EXACT map (no network) -- the
     exact-score DDIM map or the exact-velocity flow-matching map, as in code/ -- and keep the
     seeds whose sample lands outside every R99 ball: the hallucinating seeds.
  3. Compare three sets with the same number of points:
        points in noise space that would hallucinate   the hallucinating seeds
        points that hallucinate in data space          the samples they produce
        points sampled from Gaussian noise             fresh N(0, I) points (the reference: no structure)
     in two coordinate systems, Euclidean and polar (hyperspherical), by
        effective rank   exp(entropy of the normalised singular values of the centred cloud)
        dimensions for 99% of the variance: principal components holding 99% of it
     Unstructured points give values close to d; a structured set gives far less.

Both processes start from the same seeds. The hallucinating seeds of a cell are collected once
and analysed in both coordinate systems. Outputs (under run.out_dir, rank_experiment/abc), one
folder per process and one table per coordinate system in three formats:

    ddim/  euclidean.csv  euclidean.tex  euclidean.png  polar.csv  polar.tex  polar.png
    flow/  (the same)

    python rank_experiment/rank_sweep.py --config rank_experiment/config.yaml
                                         [--quick] [--tables-only] [--process ddim|flow]
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from statistics import NormalDist

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COORDS = {"euclidean": "Euclidean coordinates", "polar": "polar coordinates"}
PROCESSES = {"ddim": ("DDIM", "the exact-score DDIM map"),
             "flow": ("flow matching", "the exact-velocity flow-matching map")}


# ------------------------------------------------------------------ configuration ---------
def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, quick):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if quick:
        cfg = merge(cfg, cfg.get("quick", {}))
    return cfg


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ the mixture -----------
def chi2_ppf(q, dof):
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(q, dof))
    except Exception:                                    # Wilson-Hilferty, stdlib only
        z = NormalDist().inv_cdf(q)
        t = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
        return float(dof * t ** 3)


def r99(d, sigma, q):
    return float(math.sqrt(chi2_ppf(q, d)) * sigma)


def sample_modes(K, d, R, R99, min_sep_mult, seed):
    """K centres on the sphere of radius R, placed one at a time at least min_sep_mult * 2 R99
    apart (the same rule and RNG as code/core.py:sample_modes)."""
    min_sep = min_sep_mult * 2 * R99
    rng = np.random.RandomState(seed)
    M = np.empty((K, d))
    n, tries = 0, 0
    while n < K and tries < 10_000 * K:
        tries += 1
        v = rng.randn(d)
        v *= R / np.linalg.norm(v)
        if n and np.linalg.norm(M[:n] - v, axis=1).min() < min_sep:
            continue
        M[n] = v
        n += 1
    if n < K:
        raise RuntimeError(f"only {n} of {K} centres fit at separation {min_sep:.3f}")
    return torch.tensor(M, dtype=torch.float32), float(min_sep)


# ------------------------------------------------------------------ the exact maps --------
# The same maps as code/ (core.forward_true for DDIM, processes/flow.py for flow matching),
# with the algebra arranged so that a step reads each (batch, d) array only a few times: the
# responsibilities of the modes need only x . mu_k (|x|^2 is the same for every mode and
# cancels in the softmax), and every update has the form x' = alpha x + W M. The step's scalars
# go into the small (batch, K) and (d, K) matrices, never into addmm's alpha / beta: on an Apple
# GPU a new alpha / beta rebuilds the kernel, about 1000 times slower.
def _resp(X, M, m2, c, v):
    """Responsibilities of the modes for x ~ sum_k N(c mu_k, v I), equal weights:
    softmax_k (c x . mu_k - c^2 |mu_k|^2 / 2) / v."""
    return torch.softmax(X @ (M.T * (c / v)) + m2 * (-c * c / (2 * v)), 1)


def make_schedule(T, bmin, bmax):
    """abar_i = prod_{j <= i} (1 - beta_j), linear betas, in float32 as in code/core.py."""
    return torch.tensor(np.cumprod(1.0 - np.linspace(bmin, bmax, T)), dtype=torch.float32)


@torch.no_grad()
def ddim_map(X, M, abar, var, heun):
    """Seeds -> samples along the exact-score DDIM map, from the noisiest level abar[T-1] to the
    least noisy abar[0]. At level ab, with v = ab var + 1 - ab, the exact noise prediction is
    eps(x) = s (x - sqrt(ab) R(x) M), s = sqrt(1 - ab) / v, and the DDIM step to ab' is
    x' = a x + b eps(x), a = sqrt(ab'/ab), b = sqrt(1 - ab') - sqrt(ab' (1 - ab) / ab).
    Heun averages eps at x and at the end of that step."""
    m2 = (M * M).sum(1)[None, :]
    ab = abar.tolist()
    for i in reversed(range(1, len(ab))):
        a0, a1 = ab[i], ab[i - 1]
        a = math.sqrt(a1 / a0)
        b = math.sqrt(1 - a1) - math.sqrt(a1 * (1 - a0) / a0)
        v0 = a0 * var + 1 - a0
        s0, c0 = math.sqrt(1 - a0) / v0, math.sqrt(a0)
        R0 = _resp(X, M, m2, c0, v0)
        alpha = a + b * s0                                                       # Euler step
        Xe = torch.addmm(X, R0 * (-b * s0 * c0 / alpha), M).mul_(alpha)
        if not heun:
            X = Xe
            continue
        v1 = a1 * var + 1 - a1
        s1, c1 = math.sqrt(1 - a1) / v1, math.sqrt(a1)
        R1 = _resp(Xe, M, m2, c1, v1)
        W = R0 * (-0.5 * b * s0 * c0) + R1 * (-0.5 * b * s1 * c1)
        X = (Xe * (0.5 * b * s1)).add_(X, alpha=a + 0.5 * b * s0).addmm_(W, M)
    return X


@torch.no_grad()
def flow_map(X, M, T, var, sigma_min, heun):
    """Seeds -> samples along the exact-velocity flow-matching map (optimal-transport path),
    t: 0 (noise) -> 1 (data) on the grid linspace(0, 1, T). With s = 1 - (1 - sigma_min) t,
    v = t^2 var + s^2 and g = t var / v, the exact velocity of the mixture is
    u(x, t) = p x + q R(x) M, p = (g - 1 + sigma_min) / s, q = (1 - t g) / s.
    Heun averages u at x and at the end of the Euler step."""
    m2 = (M * M).sum(1)[None, :]
    ts = torch.linspace(0.0, 1.0, T).tolist()
    dt, oms = 1.0 / (T - 1), 1.0 - sigma_min

    def coef(t):
        s = max(1.0 - oms * t, 1e-6)
        v = t * t * var + s * s
        g = t * var / v
        return (g - oms) / s, (1.0 - t * g) / s, v

    for i in range(T - 1):
        t = ts[i]
        p0, q0, v0 = coef(t)
        R0 = _resp(X, M, m2, t, v0)
        alpha = 1 + dt * p0                                                      # Euler step
        Xe = torch.addmm(X, R0 * (dt * q0 / alpha), M).mul_(alpha)
        if not heun:
            X = Xe
            continue
        p1, q1, v1 = coef(t + dt)
        R1 = _resp(Xe, M, m2, t + dt, v1)
        W = R0 * (0.5 * dt * q0) + R1 * (0.5 * dt * q1)
        X = (Xe * (0.5 * dt * p1)).add_(X, alpha=1 + 0.5 * dt * p0).addmm_(W, M)
    return X


def make_map(process, cfg, M):
    """Seeds -> samples of one process, with the settings of config.yaml."""
    smp, var = cfg["sampler"], cfg["mixture"]["sigma"] ** 2
    heun = smp["order"] == "heun"
    if process == "ddim":
        abar = make_schedule(smp["T"], smp["ddim"]["beta_min"], smp["ddim"]["beta_max"])
        return lambda X: ddim_map(X, M, abar, var, heun)
    if process == "flow":
        return lambda X: flow_map(X, M, smp["T"], var, smp["flow"]["sigma_min"], heun)
    raise ValueError(f"unknown process {process!r} (ddim | flow)")


# ------------------------------------------------------------------ polar coordinates -----
def to_polar(X, shift=None):
    """Cartesian (n, d) -> hyperspherical (n, d): [radius, polar angles phi_1..phi_{d-2}, azimuth].
    The azimuth is rotated by `shift` (default: its circular mean) so its +-pi cut sits away
    from the data. Returns (P, shift)."""
    X = X.to(torch.float64)
    n, d = X.shape
    r = X.norm(dim=1)
    if d == 1:
        return r[:, None], 0.0
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(X ** 2, [1]), 1), [1]))   # ||x_{k:}||
    polar = torch.atan2(tail[:, 1:d - 1], X[:, :d - 2])                           # in [0, pi]
    az = torch.atan2(X[:, d - 1], X[:, d - 2])                                    # in (-pi, pi]
    if shift is None:
        shift = float(torch.atan2(torch.sin(az).mean(), torch.cos(az).mean()))
    az = torch.remainder(az - shift + math.pi, 2 * math.pi) - math.pi
    return torch.cat([r[:, None], polar, az[:, None]], 1), shift


def in_coords(A, coords, standardize):
    """A cloud (n, d) in Euclidean or polar coordinates."""
    A = A.to(torch.float64)
    if coords == "euclidean":
        return A
    P, _ = to_polar(A)
    return P / P.std(0).clamp_min(1e-12) if standardize else P


# ------------------------------------------------------------------ rank measures ---------
def rank_report(A):
    """Effective rank and n95 / n99 of the mean-centred cloud (its PCA); naive rank for reference."""
    s_raw = torch.linalg.svdvals(A)
    naive = int((s_raw > s_raw.max() * max(A.shape) * torch.finfo(s_raw.dtype).eps).sum())
    s = torch.linalg.svdvals(A - A.mean(0, keepdim=True))
    s = s[s > 0]
    p = s / s.sum()
    cum = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return {"naive": naive, "effrank": float(torch.exp(-(p * torch.log(p)).sum())),
            "n95": int((cum < 0.95).sum()) + 1, "n99": int((cum < 0.99).sum()) + 1}


# ------------------------------------------------------------------ one (d, K) cell -------
def collect(d, K, cfg, device, process):
    """The hallucinating seeds of one cell under one process, and their samples."""
    mix, col = cfg["mixture"], cfg["collection"]
    R99 = r99(d, mix["sigma"], mix["mass_q"])
    M, _ = sample_modes(K, d, mix["radius"] * math.sqrt(d / 2), R99, mix["min_sep_mult"], mix["mode_seed"])
    Md = M.to(device)
    forward_map = make_map(process, cfg, Md)
    bs = int(max(col["batch_min"], min(col["batch_max"], col["elem_budget"] // d)))
    g = torch.Generator().manual_seed(col["seed"] + 1000 * d + K)      # on the CPU: same seeds on any device
    keep_z, keep_x = [], []
    got = seen = 0
    t0 = time.time()
    while got < col["n_hall"] and seen < col["max_seeds"] and time.time() - t0 < col["max_cell_sec"]:
        b = min(bs, col["max_seeds"] - seen)
        z = torch.randn(b, d, generator=g)
        x = forward_map(z.to(device))
        hall = (torch.cdist(x, Md).min(1).values > R99).cpu()      # outside every R99 ball
        if hall.any():
            keep_z.append(z[hall]); keep_x.append(x.cpu()[hall])
            got += int(hall.sum())
        seen += b
    n = min(got, col["n_hall"])
    info = {"seeds_drawn": seen, "hallucination_rate": got / max(seen, 1), "n_hallucinating": n,
            "collect_sec": time.time() - t0}
    if n < 2:
        return info, None
    return info, (torch.cat(keep_z)[:n], torch.cat(keep_x)[:n])


def analyse(clouds, coords, standardize, seed):
    """Effective rank and PCs for 99% variance of the three clouds in one coordinate system."""
    Z, X = clouds
    G = torch.randn(Z.shape, generator=torch.Generator().manual_seed(seed + 999))
    out = {}
    for key, A in (("seeds", Z), ("samples", X), ("gaussian", G)):
        rep = rank_report(in_coords(A, coords, standardize))
        out[f"eff_{key}"] = rep["effrank"]
        out[f"n99_{key}"] = rep["n99"]
    return out


# ------------------------------------------------------------------ the table -------------
def stack(*lines):
    """A LaTeX heading on several lines, bottom-aligned with the other headings."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


SETS = [("seeds", "points in noise space that would hallucinate", ("points in", "noise space", "that would", "hallucinate")),
        ("samples", "points that hallucinate in data space", ("points that", "hallucinate", "in data", "space")),
        ("gaussian", "points sampled from Gaussian noise", ("points", "sampled from", "Gaussian", "noise"))]
# (internal key, csv heading, LaTeX heading)
COLUMNS = ([("K", "K", "$K$"), ("d", "d", "$d$"),
            ("rate", "hallucination rate (%)", stack("hallucination", "rate (\\%)")),
            ("n", "number of points", stack("number", "of points"))]
           + [(f"{pre}_{key}", f"{group}: {name}", stack(*lines))
              for pre, group in (("eff", "effective rank"), ("n99", "dimensions for 99% of the variance"))
              for key, name, lines in SETS])


def fmt(key, v):
    if key == "rate":
        return f"{v:.2f}"
    if key.startswith("eff_"):
        return f"{v:.1f}"
    return f"{int(v):,}" if key == "n" else str(int(v))


def read_table(path):
    """Rows of a saved <coords>.csv, keyed by the internal names."""
    if not os.path.exists(path):
        return []
    head = {h: k for k, h, _ in COLUMNS}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{head[h]: (float(v) if head[h] in ("rate", "eff_seeds", "eff_samples", "eff_gaussian")
                      else int(v.replace(",", ""))) for h, v in r.items()} for r in rows]


def write_table(rows, coords, folder, process):
    """<coords>.csv, <coords>.tex and <coords>.png in folder: one row per (K, d)."""
    rows = sorted(rows, key=lambda r: (r["K"], r["d"]))
    proc, the_map = PROCESSES[process]
    name = f"{proc}, {COORDS[coords]}"
    base = os.path.join(folder, coords)
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h, _ in COLUMNS])
        for r in rows:
            w.writerow([f"{r['rate']:.4f}" if k == "rate" else (f"{r[k]:.3f}" if k.startswith("eff_") else r[k])
                        for k, _, _ in COLUMNS])

    tex = ["% Auto-generated by rank_experiment/rank_sweep.py -- needs booktabs and graphicx",
           "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           f"\\caption{{\\textbf{{Hallucination rank.}} {name[0].upper() + name[1:]}. For each $(K, d)$: the "
           f"hallucination rate of {the_map}, the number of points in each set, and two "
           "measures of how many dimensions a set of points uses, for three sets: the points in noise "
           "space that would hallucinate (the seeds whose samples land outside every $R_{99}$ ball around "
           "a mode centre), the points that hallucinate in data space (those samples), and as many points "
           "sampled from Gaussian noise (the reference, which has no structure). The effective rank is the "
           "exponential of the entropy of the normalised singular values of the mean-centred points; the "
           "second measure is the number of principal components that hold 99\\% of the variance. Points "
           "without structure give values close to $d$.}",
           f"\\label{{tab:rank-{process}-{coords}}}",
           "% shrinks to the text width only when it is wider",
           "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{rrrr ccc ccc}", "\\toprule",
           " & & & & \\multicolumn{3}{c}{effective rank} & \\multicolumn{3}{c}{dimensions for 99\\% of the variance} \\\\",
           "\\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
           " & ".join(t for _, _, t in COLUMNS) + " \\\\", "\\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["K"] != prev:
            tex.append("\\midrule")
        prev = r["K"]
        tex.append(" & ".join(fmt(k, r[k]).replace(",", "{,}") for k, _, _ in COLUMNS) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(base + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    table_png(rows, name, base + ".png")


def table_png(rows, name, path):
    """The same table as an image: grouped headings, one band per K."""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]})
    widths = [0.45, 0.6, 1.05, 0.85, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]
    row_h, head_h = 0.22, 1.04
    W, H = sum(widths) + 0.2, head_h + row_h * len(rows) + 1.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x0 = [0.1 + sum(widths[:i]) for i in range(len(widths))]
    xc = [x + w / 2 for x, w in zip(x0, widths)]
    ink, rule, band = "#1f1f1f", "#5a5a5a", "#eef3fb"
    ax.text(W / 2, 0.2, "Hallucination rank", ha="center", va="center",
            fontsize=12, fontweight="bold", color=ink)
    ax.text(W / 2, 0.41, name[0].upper() + name[1:], ha="center", va="center", fontsize=9.5, color="#4a4a4a")
    top = 0.55
    ax.plot([0.1, W - 0.1], [top, top], color=rule, lw=1.0)
    for label, lo, hi in (("effective rank", 4, 6), ("dimensions for 99% of the variance", 7, 9)):
        ax.text((x0[lo] + x0[hi] + widths[hi]) / 2, top + 0.14, label, ha="center", va="center",
                fontsize=9.5, color=ink)
        ax.plot([x0[lo] + 0.06, x0[hi] + widths[hi] - 0.06], [top + 0.27, top + 0.27], color=rule, lw=0.6)
    subs = ["\n".join(lines) for _, _, lines in SETS]
    heads = ["K", "d", "hallucination\nrate (%)", "number\nof points"] + subs + subs
    for x, h in zip(xc, heads):
        ax.text(x, top + head_h - 0.33, h, ha="center", va="center", fontsize=9.5, color=ink,
                style="italic" if h in ("K", "d") else "normal")
    y = top + head_h + 0.04
    ax.plot([0.1, W - 0.1], [y, y], color=rule, lw=0.6)
    Ks = sorted({r["K"] for r in rows})
    for i, r in enumerate(rows):
        yy = y + i * row_h
        if Ks.index(r["K"]) % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, yy), W - 0.2, row_h, color=band, lw=0, zorder=0))
        for x, (k, _, _) in zip(xc, COLUMNS):
            ax.text(x, yy + row_h / 2, fmt(k, r[k]), ha="center", va="center", fontsize=9, color=ink)
    yb = y + len(rows) * row_h
    ax.plot([0.1, W - 0.1], [yb, yb], color=rule, lw=1.0)
    ax.text(0.12, yb + 0.12,
            "points in noise space that would hallucinate: the seeds whose samples land outside every R99 ball.   "
            "points that hallucinate in data space: those samples.\n"
            "points sampled from Gaussian noise: as many fresh N(0, I) points, with no structure; "
            "points without structure give about d for both measures.\n"
            "effective rank: exp(entropy of the normalised singular values of the centred points).   "
            "dimensions for 99% of the variance: principal components needed to hold 99% of it.",
            ha="left", va="top", fontsize=8, color="#4a4a4a", linespacing=1.4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ main ------------------
def rebuild(cfg, processes):
    for process in processes:
        folder = os.path.join(cfg["run"]["out_dir"], process)
        for coords in cfg["analysis"]["coords"]:
            rows = read_table(os.path.join(folder, coords + ".csv"))
            if rows:
                write_table(rows, coords, folder, process)
                print(f"[rank] {process}/{coords}: {len(rows)} rows -> {coords}.csv, {coords}.tex, {coords}.png")
            else:
                print(f"[rank] no results for {coords} under {folder}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="the small grid of the config's `quick` block")
    ap.add_argument("--tables-only", action="store_true", help="rebuild the .tex and .png from the CSVs")
    ap.add_argument("--process", choices=sorted(PROCESSES), help="run one process (default: those of the config)")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu (overrides the config)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    processes = [args.process] if args.process else list(cfg["processes"])
    if args.tables_only:
        rebuild(cfg, processes)
        return
    device = pick_device(args.device or cfg["run"]["device"])
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    coords_list = cfg["analysis"]["coords"]
    grid = [(d, K) for d in cfg["grid"]["d"] for K in cfg["grid"]["K"]]
    # a cell is done once it has the n_hall points this run asks for: the full run redoes the
    # cells of a quick run (fewer points), and a quick run leaves the full results alone. The
    # cells of this grid that are not done are redone: their old rows are dropped now, so the
    # tables never mix the rows of a quick run with those of the full run
    want = cfg["collection"]["n_hall"]
    folders, results, done = {}, {}, {}
    for p in processes:
        folders[p] = os.path.join(cfg["run"]["out_dir"], p)
        os.makedirs(folders[p], exist_ok=True)
        res = {c: (read_table(os.path.join(folders[p], c + ".csv")) if cfg["run"]["resume"] else [])
               for c in coords_list}
        done[p] = set.intersection(*[{(r["d"], r["K"]) for r in res[c] if r["n"] >= want} for c in coords_list])
        results[p] = {c: [r for r in res[c] if (r["d"], r["K"]) in done[p] or (r["d"], r["K"]) not in set(grid)]
                      for c in coords_list}
    print("=" * 108)
    print(f"rank experiment: {len(grid)} cells for each of {', '.join(processes)} "
          f"({', '.join(f'{p}: {len(done[p] & set(grid))} done' for p in processes)}), device={device}, "
          f"results -> {cfg['run']['out_dir']}/<process>")
    print("=" * 108)
    print("effective rank of: noise = points in noise space that would hallucinate, data = points that "
          "hallucinate in data space, Gauss. = points sampled from Gaussian noise")
    print(f"{'process':>7}{'d':>6}{'K':>4}{'hall. %':>9}{'points':>8} | "
          f"{'  '.join(c[:3] + ' noise   data  Gauss.' for c in coords_list)} | {'sec':>6}")
    t_all = time.time()
    for d, K in grid:
        for p in processes:
            if (d, K) in done[p]:
                continue
            try:
                info, clouds = collect(d, K, cfg, device, p)
            except RuntimeError as e:
                print(f"{p:>7}{d:>6}{K:>4}  skipped: {e}", flush=True)
                continue
            if clouds is None:
                print(f"{p:>7}{d:>6}{K:>4}  fewer than 2 hallucinating seeds in {info['seeds_drawn']:,} draws",
                      flush=True)
                continue
            line = []
            for coords in coords_list:
                out = analyse(clouds, coords, cfg["analysis"]["polar_standardize"], cfg["collection"]["seed"])
                results[p][coords] = [r for r in results[p][coords] if (r["d"], r["K"]) != (d, K)] + [
                    {"K": K, "d": d, "rate": 100 * info["hallucination_rate"], "n": info["n_hallucinating"], **out}]
                write_table(results[p][coords], coords, folders[p], p)
                line.append(f"{coords[:3]} {out['eff_seeds']:6.1f} {out['eff_samples']:6.1f} "
                            f"{out['eff_gaussian']:6.1f}")
            print(f"{p:>7}{d:>6}{K:>4}{100 * info['hallucination_rate']:>9.3f}{info['n_hallucinating']:>8,} | "
                  f"{'  '.join(line)} | {info['collect_sec']:>6.0f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\ndone in {time.time() - t_all:.0f}s -> {cfg['run']['out_dir']}/<process>")


if __name__ == "__main__":
    main()
