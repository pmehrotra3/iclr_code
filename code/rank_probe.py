#!/usr/bin/env python
# ==========================================================================================
# rank_probe — SELF-CONTAINED: is the DDIM hallucination set low-rank?
#
# Depends on NOTHING in the repo. Needs numpy + torch (scipy optional). Runs three ways:
#     !python rank_probe.py          %run rank_probe.py          (or paste the whole file in a cell)
# Edit the CONFIG block below; everything else is self-contained.
#
# A seed is a point in NOISE space (x ~ N(0,I)); the DDIM map carries it to DATA space. A seed
# "hallucinates" if its endpoint lands in no mode ball. We gather ~n_hall hallucinators and
# measure how many dimensions their coordinates occupy in:
#   (1) HALL @ noise : the seeds x0        (2) HALL @ data : their endpoints
#   (3) GAUSS base   : fresh N(0,I) draws (the unstructured control)
#
# Naive matrix rank is trivially d for all three (n >> d, continuous data) -- reported so you
# can see it saturate, but the signal is the SVD/effective-rank block on the mean-centred
# (PCA) matrix: structured set -> effrank << d, n99 << d ; Gaussian noise -> ~= d.
# ==========================================================================================
import math
import time
import numpy as np
import torch
from statistics import NormalDist

# -------------------------------------- CONFIG --------------------------------------------
d          = 32           # ambient dimension
K          = 8            # number of GMM modes
sigma      = 0.1          # within-mode std dev
radius     = 2.0          # base sphere radius (scaled by sqrt(d/2))
m_mult     = 1.5          # min mode-separation multiplier
mass_q     = 0.99         # core mass fraction -> R99
T          = 200          # DDIM steps
beta_min   = 1e-4
beta_max   = 0.02
order      = "heun"       # "heun" (2nd order) | "euler"
n_hall     = 100_000      # target number of hallucinating points
max_seeds  = 40_000_000   # cap on seeds sampled
batch      = 200_000      # seeds per batch
seed       = 12345
mode_seed  = 0
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_npz   = None         # e.g. "out.npz" to dump coords + spectra
# ------------------------------------------------------------------------------------------


def chi2_ppf(q, dof):
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(q, dof))
    except Exception:                                   # Wilson-Hilferty, stdlib only
        z = NormalDist().inv_cdf(q)
        t = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
        return float(dof * t ** 3)


def r99(d, sigma, q=0.99):
    return float(math.sqrt(chi2_ppf(q, d)) * sigma)


def make_schedule(T, bmin, bmax, dev):
    abar = np.cumprod(1.0 - np.linspace(bmin, bmax, T))
    return torch.tensor(abar, dtype=torch.float32, device=dev)


def sample_modes(K, d, R, sigma, m_mult, seed, dev):
    min_sep = m_mult * 2 * r99(d, sigma)
    rng = np.random.RandomState(seed)
    modes = np.empty((K, d))
    n, tries, mx = 0, 0, 10_000 * K
    while n < K and tries < mx:
        tries += 1
        v = rng.randn(d)
        v *= R / np.linalg.norm(v)
        if n and np.linalg.norm(modes[:n] - v, axis=1).min() < min_sep:
            continue
        modes[n] = v
        n += 1
    if n < K:
        raise RuntimeError(f"placed only {n}/{K} modes (min_sep={min_sep:.3f}); "
                           f"raise radius or lower sigma/K")
    return torch.tensor(modes.astype(np.float32), device=dev), float(min_sep)


@torch.no_grad()
def true_score(X, Mt, v):
    w = torch.softmax(-(torch.cdist(X, Mt) ** 2) / (2 * v), 1)
    return (w @ Mt - X) / v


@torch.no_grad()
def _eps(x, M, ab, var):
    v = ab * var + (1 - ab)
    return -torch.sqrt(1 - ab) * true_score(x, torch.sqrt(ab) * M, v)


@torch.no_grad()
def _step(x, eps, ab, abp):
    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps


@torch.no_grad()
def forward_map(X0, M, abar, T, var, order="heun", chunk=100_000):
    """Carry seeds X0 (noise) forward to data space under the exact GMM score."""
    heun = str(order).lower() == "heun"
    levels = [(abar[i], abar[i - 1]) for i in reversed(range(1, T))]
    X = X0.clone()
    for ab, abp in levels:
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            e1 = _eps(xs, M, ab, var)
            if heun:
                e1 = 0.5 * (e1 + _eps(_step(xs, e1, ab, abp), M, abp, var))
            X[s:s + chunk] = _step(xs, e1, ab, abp)
    return X


@torch.no_grad()
def label_fate(Xf, M, R99, chunk=100_000):
    lab = torch.empty(Xf.shape[0], dtype=torch.long, device=Xf.device)
    for s in range(0, Xf.shape[0], chunk):
        dmin, arg = torch.cdist(Xf[s:s + chunk], M).min(1)
        lab[s:s + chunk] = torch.where(dmin <= R99, arg, torch.full_like(arg, -1))
    return lab


def rank_report(A, name, dev):
    A = A.to(device=dev, dtype=torch.float64)
    n, dd = A.shape
    s_raw = torch.linalg.svdvals(A)
    tol = s_raw.max() * max(n, dd) * torch.finfo(s_raw.dtype).eps
    naive = int((s_raw > tol).sum())
    s = torch.linalg.svdvals(A - A.mean(0, keepdim=True))
    s = s[s > 0]
    s2 = s ** 2
    p = s / s.sum()
    effrank = float(torch.exp(-(p * torch.log(p)).sum()))
    stable = float(s2.sum() / s2[0])
    pr = float((s2.sum() ** 2) / (s2 ** 2).sum())
    cum = torch.cumsum(s2, 0) / s2.sum()
    n90 = int((cum < 0.90).sum()) + 1
    n95 = int((cum < 0.95).sum()) + 1
    n99 = int((cum < 0.99).sum()) + 1
    return {"name": name, "n": n, "d": dd, "naive_rank": naive, "effrank": effrank,
            "stable_rank": stable, "participation_ratio": pr, "n90": n90, "n95": n95,
            "n99": n99, "svals": s.detach().cpu().numpy()}


def mode_subspace_energy(A, M, dev):
    A = A.to(device=dev, dtype=torch.float64)
    M = M.to(device=dev, dtype=torch.float64)
    Ac = A - A.mean(0, keepdim=True)
    Q = torch.linalg.svd((M - M.mean(0, keepdim=True)).T, full_matrices=False).U
    tot = (Ac ** 2).sum()
    if tot <= 0:
        return float("nan")
    return float(((Ac @ Q) ** 2).sum() / tot)


def fmt(r):
    return (f"{r['name']:<16} n={r['n']:>8,} d={r['d']:>3} | naive_rank={r['naive_rank']:>3} | "
            f"effrank={r['effrank']:>6.2f} stable={r['stable_rank']:>6.2f} "
            f"PR={r['participation_ratio']:>6.2f} | n90={r['n90']:>3} n95={r['n95']:>3} "
            f"n99={r['n99']:>3}")


# ======================================================================================
# run
# ======================================================================================
variance = sigma ** 2
R = radius * (d / 2.0) ** 0.5                            # keep modes resolvable as d grows
R99 = r99(d, sigma, mass_q)
M, min_sep = sample_modes(K, d, R, sigma, m_mult, mode_seed, device)
abar = make_schedule(T, beta_min, beta_max, device)

print("=" * 90)
print(f"rank probe (exact-score DDIM)  d={d}  K={K}  T={T}  order={order}")
print(f"  sigma={sigma}  radius={R:.3f}  R99={R99:.4f}  min_sep={min_sep:.3f}  device={device}")
print(f"  target n_hall={n_hall:,}  max_seeds={max_seeds:,}  batch={batch:,}")
print("=" * 90)

g = torch.Generator(device=device).manual_seed(seed)
X0k, Xfk = [], []
got, seen, t0 = 0, 0, time.time()
print("[collect] running exact-score DDIM (noise -> data) and keeping hallucinators ...")
while got < n_hall and seen < max_seeds:
    b = min(batch, max_seeds - seen)
    X0 = torch.randn(b, d, generator=g, device=device)
    Xf = forward_map(X0, M, abar, T, variance, order=order)
    m = label_fate(Xf, M, R99) == -1
    if m.any():
        X0k.append(X0[m].cpu())
        Xfk.append(Xf[m].cpu())
        got += int(m.sum())
    seen += b
    print(f"  ... seeds={seen:>13,}  halls={got:>10,}  rate={got / max(seen, 1):.4f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

assert got > 0, "collected 0 hallucinations — raise max_seeds/K or sigma"
X0h = torch.cat(X0k)[:n_hall]
Xfh = torch.cat(Xfk)[:n_hall]
n = X0h.shape[0]
print(f"[collect] gathered {n:,} hallucinators from {seen:,} seeds (rate {got / max(seen, 1):.4f})")
if n < n_hall:
    print(f"[collect] NOTE: fewer than requested; hit max_seeds. Measures valid on {n:,} rows.")

G = torch.randn(n, d, generator=torch.Generator(device=device).manual_seed(seed + 999), device=device)

print("\n" + "-" * 90)
print("RANK / EFFECTIVE-RANK  (effective measures on the mean-centred matrix = PCA)")
print("-" * 90)
r_noise = rank_report(X0h, "HALL @ noise", device)
r_data = rank_report(Xfh, "HALL @ data", device)
r_gauss = rank_report(G, "GAUSS baseline", device)
for r in (r_noise, r_data, r_gauss):
    print(fmt(r))

r_modes = int(torch.linalg.matrix_rank((M - M.mean(0, keepdim=True)).to(torch.float64)))
print("-" * 90)
print(f"mode centres: K={K}, span dim (rank of centred means) = {r_modes}  (<= K-1 = {K - 1})")
print(f"variance in mode span:  HALL@noise={mode_subspace_energy(X0h, M, device):.3f}   "
      f"HALL@data={mode_subspace_energy(Xfh, M, device):.3f}   "
      f"GAUSS={mode_subspace_energy(G, M, device):.3f}")

print("-" * 90)
print("top centred singular values:")
kk = min(12, d)
print("  HALL@noise : " + "  ".join(f"{v:.3g}" for v in r_noise["svals"][:kk]))
print("  HALL@data  : " + "  ".join(f"{v:.3g}" for v in r_data["svals"][:kk]))
print("  GAUSS      : " + "  ".join(f"{v:.3g}" for v in r_gauss["svals"][:kk]))
print("=" * 90)

print("READ:")
naive_line = "  * naive rank = {}/{} (hall) vs {}/{} (gauss): naive rank cannot separate them.".format(
    r_noise["naive_rank"], d, r_gauss["naive_rank"], d)
print(naive_line)
print("  * effective rank: HALL@noise={:.2f}  HALL@data={:.2f}  GAUSS={:.2f}  (ceiling ~{}).".format(
    r_noise["effrank"], r_data["effrank"], r_gauss["effrank"], d))
if r_gauss["effrank"] > 0:
    print("    -> halls occupy {:.2f}x the Gaussian's effective dimension; n99 = {} vs {} PCs.".format(
        r_noise["effrank"] / r_gauss["effrank"], r_noise["n99"], r_gauss["n99"]))
close = abs(r_noise["effrank"] - r_data["effrank"]) < 0.25 * max(r_noise["effrank"], 1)
print("  * noise vs data effrank are {} ({:.2f} vs {:.2f}): structure {}.".format(
    "CLOSE" if close else "DIFFERENT", r_noise["effrank"], r_data["effrank"],
    "is already present in noise space" if close else "differs between spaces"))

if save_npz:
    np.savez_compressed(save_npz, X0_hall=X0h.numpy(), Xf_hall=Xfh.numpy(),
                        sv_noise=r_noise["svals"], sv_data=r_data["svals"],
                        sv_gauss=r_gauss["svals"], means=M.cpu().numpy())
    print(f"[save] wrote {save_npz}")
