"""diagrams.py — the figures of the preview pages, as inline SVG (colours come from style.css,
so they follow light / dark mode).

fate_map()      COMPUTED with the repository's own code (core.py): a 2-D, 3-mode mixture, every
                seed of a grid coloured by where the exact DDIM sampler takes it, next to the
                data space where a few hundred N(0, I) seeds actually land.
anchors()       schematic: where evaluate.py plants its labelled anchors around one mode.
"""
from __future__ import annotations
import math
import os
import random
import sys

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")


def _rle_rows(lab, x0, y0, cell, classes):
    """One <rect> per horizontal run of equal labels (keeps the SVG small)."""
    out = []
    n_rows, n_cols = len(lab), len(lab[0])
    for i in range(n_rows):
        j = 0
        while j < n_cols:
            v, s = lab[i][j], j
            while j < n_cols and lab[i][j] == v:
                j += 1
            out.append(f'<rect x="{x0 + s * cell:.2f}" y="{y0 + i * cell:.2f}" width="{(j - s) * cell + .4:.2f}" '
                       f'height="{cell + .4:.2f}" class="{classes[v]}"/>')
    return "".join(out)


def fate_map(T=200, n_grid=150, n_seeds=600):
    """Returns (svg, facts). Uses core.sample_modes / make_schedule / forward_true / label_fate."""
    sys.path.insert(0, CODE)
    import torch
    import core

    d, K, sigma = 2, 3, 0.1
    M, _ = core.sample_modes(K, d, 2.0, sigma, 1.5, seed=0)          # data.radius = 2 at d = 2
    R99 = core.r99(d, sigma)
    ab = core.make_schedule(T, 1e-4, 0.02)
    L = 3.0                                                          # seed grid covers [-L, L]^2
    xs = torch.linspace(-L, L, n_grid)
    grid = torch.stack(torch.meshgrid(xs, xs.flip(0), indexing="xy"), -1).reshape(-1, 2)
    lab = core.label_fate(core.forward_true(grid, M, ab, T, sigma ** 2), M, R99).reshape(n_grid, n_grid).tolist()
    seeds = torch.randn(n_seeds, 2, generator=torch.Generator().manual_seed(1))
    ends = core.forward_true(seeds, M, ab, T, sigma ** 2)
    fates = core.label_fate(ends, M, R99)
    big = torch.randn(200_000, 2, generator=torch.Generator().manual_seed(2))
    hall_rate = float((core.label_fate(core.forward_true(big, M, ab, T, sigma ** 2), M, R99) == -1).float().mean())

    W, P, top = 360, 40, 46                          # panel size, gap, title band
    cls_fill = {-1: "dg-hall", 0: "dg-m1", 1: "dg-m2", 2: "dg-m3"}
    s = [f'<svg viewBox="0 0 {2 * W + P + 20} {W + top + 58}" role="img" '
         f'aria-label="Seed-fate map computed with core.py: seed space coloured by fate, and data space with the modes">']
    # ---- left: seed space ----
    x0, y0 = 10, top
    s.append(f'<text x="{x0}" y="20" class="t-title dg-ink">Seed space: every starting point coloured by its fate</text>')
    s.append(f'<text x="{x0}" y="37" class="t-small dg-faint">{n_grid}×{n_grid} seeds in [−3, 3]², exact sampler, T = {T}</text>')
    s.append('<g opacity=".62">' + _rle_rows(lab, x0, y0, W / n_grid, {k: v for k, v in cls_fill.items()}) + "</g>")
    c, u = (x0 + W / 2, y0 + W / 2), W / (2 * L)
    for r, name in ((1, "|x| = 1"), (2, "|x| = 2")):
        s.append(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r * u:.1f}" class="dg-line" stroke-width="1.1" stroke-dasharray="3 3"/>')
        s.append(f'<text x="{c[0] + r * u * .72 + 4:.1f}" y="{c[1] + r * u * .72 + 12:.1f}" class="t-small dg-ink">{name}</text>')
    s.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{W}" class="dg-rule" stroke-width="1"/>')
    # ---- right: data space ----
    x1 = x0 + W + P
    s.append(f'<text x="{x1}" y="20" class="t-title dg-ink">Data space: where {n_seeds} random seeds land</text>')
    s.append(f'<text x="{x1}" y="37" class="t-small dg-faint">dashed circles = each mode\'s R99 ball (radius {R99:.2f})</text>')
    cen = M.mean(0)
    half = float((M - cen).abs().max()) + R99 + 0.3              # every mode fully inside the panel
    uu = W / (2 * half)
    to = lambda p: (x1 + (float(p[0]) - float(cen[0]) + half) * uu, y0 + (float(cen[1]) - float(p[1]) + half) * uu)
    s.append(f'<clipPath id="dsclip"><rect x="{x1}" y="{y0}" width="{W}" height="{W}"/></clipPath>')
    s.append(f'<g clip-path="url(#dsclip)">')
    for k in range(K):
        mx, my = to(M[k])
        s.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{R99 * uu:.1f}" class="dg-rule dg-s-m{k + 1}" '
                 f'stroke-width="1.6" stroke-dasharray="5 4" fill="none"/>')
    for p, f in zip(ends, fates.tolist()):
        if f >= 0:
            px, py = to(p)
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.9" class="{cls_fill[f]}" opacity=".75"/>')
    for p, f in zip(ends, fates.tolist()):
        if f == -1:
            px, py = to(p)
            s.append(f'<g transform="translate({px:.1f},{py:.1f})" class="dg-s-hall" stroke-width="2.2">'
                     f'<line x1="-4" y1="-4" x2="4" y2="4"/><line x1="-4" y1="4" x2="4" y2="-4"/></g>')
    for k in range(K):
        mx, my = to(M[k])
        s.append(f'<text x="{mx + R99 * uu + 5:.1f}" y="{my - R99 * uu * .6:.1f}" class="t-small dg-ink">mode {k}</text>')
    s.append("</g>")
    s.append(f'<rect x="{x1}" y="{y0}" width="{W}" height="{W}" class="dg-rule" stroke-width="1"/>')
    # ---- legend ----
    ly = y0 + W + 26
    items = [("dg-m1", "→ mode 0"), ("dg-m2", "→ mode 1"), ("dg-m3", "→ mode 2"), ("dg-hall", "→ no ball: hallucination (−1)")]
    lx = x0
    for cl, name in items:
        s.append(f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" class="{cl}" opacity=".75"/>'
                 f'<text x="{lx + 20}" y="{ly}" class="dg-ink">{name}</text>')
        lx += 20 + 8.2 * len(name) + 22
    s.append("</svg>")
    n_hall_shown = int((fates == -1).sum())
    return "".join(s), {"T": T, "R99": R99, "hall_rate": hall_rate, "n_hall_shown": n_hall_shown,
                        "n_seeds": n_seeds, "means": [[round(float(v), 2) for v in m] for m in M]}


def anchors():
    """Schematic of core.ball_anchors (left) and core.altered_knn_anchors (right), in 2-D."""
    rnd = random.Random(0)
    W, H = 820, 330
    R, band = 78.0, 50.0                      # R99 and the shell_sigma * sigma band, in px
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Where the anchors are planted around one mode">']

    # left: ball + shell
    cx, cy = 190, 175
    s.append('<text x="20" y="22" class="t-title dg-ink">ball_anchors — used by knn, quadratic, polar3</text>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R + band}" class="dg-rule" stroke-width="1" stroke-dasharray="2 3"/>')
    for _ in range(210):                      # uniform in the ball: radius R * u^(1/d), d = 2
        r, a = R * math.sqrt(rnd.random()), rnd.uniform(0, 2 * math.pi)
        s.append(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" r="2.2" class="dg-m1" opacity=".8"/>')
    for _ in range(105):                      # the band just outside: radius uniform in [R, R + band]
        r, a = R + band * rnd.random(), rnd.uniform(0, 2 * math.pi)
        s.append(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" r="2.2" class="dg-hall" opacity=".85"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" class="dg-line" stroke-width="1.8"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" class="dg-ink"/>')
    s.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + R}" y2="{cy}" class="dg-line" stroke-width="1.2"/>')
    s.append(f'<text x="{cx + 14}" y="{cy - 6}" class="t-small dg-ink">R99</text>')
    s.append(f'<text x="{cx - 12}" y="{cy + 18}" class="t-small dg-ink">μₖ</text>')
    s.append(f'<text x="{cx + R + band + 8}" y="{cy - 60}" class="t-small dg-faint">R99 + shell_sigma·σ</text>')
    s.append(f'<circle cx="40" cy="{H - 22}" r="4" class="dg-m1"/><text x="50" y="{H - 18}" class="t-small dg-ink">b points, label k</text>')
    s.append(f'<circle cx="170" cy="{H - 22}" r="4" class="dg-hall"/><text x="180" y="{H - 18}" class="t-small dg-ink">b × shell_frac points, label −1</text>')

    # right: rings
    cx = 600
    s.append('<text x="430" y="22" class="t-title dg-ink">altered_knn_anchors — used by altered_knn</text>')
    n_rings, r_max, w_min = 5, 1.5, 0.2
    for j in range(1, n_rings + 1):
        r = R * r_max * j / n_rings
        w = 1 - (1 - w_min) * (j / n_rings)
        for i in range(34):
            a = 2 * math.pi * i / 34 + j * .31
            s.append(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" r="{1.4 + 2.2 * w:.1f}" '
                     f'class="dg-m1" opacity="{.25 + .7 * w:.2f}"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" class="dg-line" stroke-width="1.8" stroke-dasharray="6 4"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" class="dg-ink"/>')
    s.append(f'<text x="{cx + R * .72 + 2}" y="{cy - R * .72 - 4}" class="t-small dg-ink">R99</text>')
    s.append(f'<text x="{cx - 18}" y="{cy + 18}" class="t-small dg-ink">μₖ</text>')
    s.append(f'<text x="430" y="{H - 18}" class="t-small dg-ink">5 rings at 0.3 … 1.5 × R99, all labelled k;'
             f' bigger, darker dot = higher weight</text>')
    s.append("</svg>")
    return "".join(s)
