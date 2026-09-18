"""common/gmm.py — the Gaussian-mixture reference distribution.

K isotropic modes (std sigma) on a sphere of radius R in R^d. The labelling rule L: a
data-space point is "in mode k" if mu_k is the nearest centre and lies within R99 (the radius
holding fraction q of a mode's mass), and is a *hallucination* (label -1) if it is within R99
of no mode. The same rule colours the ground truth and the planted anchors.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import chi2


def r99(d: int, sigma: float, q: float = 0.99) -> float:
    """Radius of the ball that holds fraction q of an isotropic Gaussian mode."""
    return float(np.sqrt(chi2.ppf(q, d)) * sigma)


def sample_modes(K: int, d: int, R: float, sigma: float, m_mult: float = 1.0,
                 seed: int = 0, device: torch.device | None = None):
    """Place K mode centers on a sphere of radius R with a minimum separation."""
    mult = 5.0 if d > 2 else 3.0
    min_sep2 = (mult * m_mult * sigma) ** 2
    rng = np.random.RandomState(seed)
    modes: list[np.ndarray] = []
    tries, max_tries = 0, 500 * K + 2_000_000
    while len(modes) < K and tries < max_tries:
        v = rng.randn(d)
        v = R * v / np.linalg.norm(v)
        if all(np.dot(v - m, v - m) > min_sep2 for m in modes):
            modes.append(v)
        tries += 1
    if len(modes) < K and d == 2:
        # d=2 packs K modes onto a circle, where rejection sampling stalls well before the
        # geometric limit (K=32 needs 9.6 of the 12.57 circumference). Equal spacing is the
        # only arrangement up to rotation, so place them deterministically instead.
        ang = 2 * np.pi * (np.arange(K) + rng.rand()) / K
        M = np.stack([R * np.cos(ang), R * np.sin(ang)], 1).astype(np.float32)
        if np.min(np.sum((M[:, None] - M[None]) ** 2, -1) + np.eye(K) * 1e9) <= min_sep2:
            raise RuntimeError(f"d=2, K={K}: even spacing still violates the {mult}-sigma separation")
        return torch.tensor(M, device=device), float(mult)
    if len(modes) < K:
        raise RuntimeError(f"could only place {len(modes)}/{K} modes for d={d}, K={K}")
    M = np.asarray(modes, dtype=np.float32)
    return torch.tensor(M, device=device), float(mult)


def sample_data(means_t: torch.Tensor, sigma: float, n: int, device: torch.device):
    """n samples from the mixture (uniform component weights)."""
    K, d = means_t.shape
    k = torch.randint(0, K, (n,), device=device)
    return means_t[k] + sigma * torch.randn(n, d, device=device)


@torch.no_grad()
def label_fate(Xf: torch.Tensor, means_t: torch.Tensor, R99: float, chunk: int = 50000):
    """Fate label of data-space endpoints: mode index, or -1 for hallucination."""
    N = Xf.shape[0]
    lab = torch.empty(N, dtype=torch.long, device=Xf.device)
    for s in range(0, N, chunk):
        D = torch.cdist(Xf[s:s + chunk], means_t)
        dmin, arg = D.min(1)
        lab[s:s + chunk] = torch.where(dmin <= R99, arg, torch.full_like(arg, -1))
    return lab


@torch.no_grad()
def ball_anchors(means_t, R99, n_per_mode, shell_frac=0.5, shell_sigma=2.0, sigma=None,
                 seed=0, device=None):
    """Step 1 of the atlas: labelled anchors planted in DATA space, no model involved.

    Around every mode centre: n_per_mode points uniform in the R99 ball, plus shell_frac *
    n_per_mode points uniform in radius over the band R99 .. R99 + shell_sigma * sigma just
    outside it -- the planted hypothesis that hallucinations live in a thin band around each
    mode. Every anchor is then coloured with the SAME rule L as the ground truth (label_fate:
    nearest mode if within R99 of it, else -1). Colouring by the planting mode instead breaks
    whenever two balls overlap (mode centres closer than 2 R99, the rule at d = 2): a point in
    mode k's ball can be nearer to mode j, and a point in k's band can lie inside j's ball.
    Returns P (n, d) data-space positions and y (n,) labels.
    """
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(seed)
    n_shell = max(1, int(round(shell_frac * n_per_mode)))
    width = shell_sigma * sigma
    P = []
    for k in range(K):
        u = torch.rand(n_per_mode, generator=g, device=device) ** (1.0 / d)
        dirs = torch.randn(n_per_mode, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        P.append(means_t[k] + (R99 * u)[:, None] * dirs)
        r = R99 + width * torch.rand(n_shell, generator=g, device=device)
        dirs = torch.randn(n_shell, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        P.append(means_t[k] + r[:, None] * dirs)
    P = torch.cat(P, 0)
    return P, label_fate(P, means_t, R99)


@torch.no_grad()
def altered_knn_anchors(means_t, R99, n_per_mode, n_rings=5, r_max=1.5, weight="linear",
                        w_min=0.2, sigma=None, seed=0, device=None):
    """Mode-labelled anchors with confidence weights for altered_knn (no hallucination class).

    Per mode: n_rings concentric spheres of radius r_j = R99 * r_max * j / n_rings (j = 1..n_rings),
    n_per_mode / n_rings points uniform on each, every point labelled with its own mode. The
    weight decays with the radius:
      linear   : w = 1 - (1 - w_min) * r / (r_max R99)
      gaussian : w = exp(-r^2 / (2 sigma^2))  (the mode's own density ratio; needs sigma)
    Returns P (n, d), y (n,), w (n,).
    """
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(seed)
    per_ring = max(1, n_per_mode // n_rings)
    P, y, w = [], [], []
    for k in range(K):
        for j in range(1, n_rings + 1):
            r = R99 * r_max * j / n_rings
            dirs = torch.randn(per_ring, d, generator=g, device=device)
            dirs /= dirs.norm(dim=1, keepdim=True)
            P.append(means_t[k] + r * dirs)
            y += [k] * per_ring
            if weight == "gaussian":
                wj = float(np.exp(-r ** 2 / (2 * sigma ** 2)))
            else:
                wj = 1.0 - (1.0 - w_min) * (r / (R99 * r_max))
            w += [wj] * per_ring
    return torch.cat(P, 0), torch.tensor(y, device=device), torch.tensor(w, device=device, dtype=torch.float32)


@torch.no_grad()
def true_score(X: torch.Tensor, means_t: torch.Tensor, v: float):
    """Score of the isotropic mixture N(mu_k, v I) at X (uniform weights)."""
    d2 = torch.cdist(X, means_t) ** 2
    w = torch.softmax(-d2 / (2 * v), 1)
    return (w @ means_t - X) / v
