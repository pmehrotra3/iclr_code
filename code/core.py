"""
core.py — shared building blocks for the diffusion-atlas experiments.

Everything that more than one stage needs lives here: the score network, the VP
schedule, GMM mode sampling, the learned + exact-score samplers, the exact-field
backtrack / forward, and the labelled data-space anchors. The stage files
(train / evaluate / visualize) import from here so the numerics are defined once.
"""


# --------------------------------------------------------------------------------------
# Importing necessary modules
# --------------------------------------------------------------------------------------
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2


# --------------------------------------------------------------------------------------
# device
# Defaults to "auto": picks the GPU if one is visible, else CPU.
# "cpu" and "cuda" are explicit overrides; "cuda" errors here if CUDA is missing.
# --------------------------------------------------------------------------------------

def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# noising schedule
# Linear betas over T steps. Returns abar[i] = prod(1 - beta), the signal left at step i:
# x_t = sqrt(abar)*x0 + sqrt(1-abar)*noise.
# Defaults assume T=1000, beta_min = 0.001, beta_max =  0.02
# --------------------------------------------------------------------------------------

def make_schedule(T: int = 1000, beta_min: float = 0.001, beta_max: float = 0.02,
                  device: torch.device | None = None) -> torch.Tensor:
    betas = np.linspace(beta_min, beta_max, T)
    abar = np.cumprod(1.0 - betas)
    return torch.tensor(abar, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------------------
# Radius of the ball holding fraction q of an isotropic Gaussian mode.
# Chi-squared with d dof, so it grows like sqrt(d)*sigma. This is the
# mode/hallucination boundary used downstream.
# --------------------------------------------------------------------------------------
def r99(d: int, sigma: float, q: float = 0.99) -> float:
    return float(np.sqrt(chi2.ppf(q, d)) * sigma)



# --------------------------------------------------------------------------------------
# GMM modes
# Rejection-sample K centers on the sphere of radius R, keeping every pair at least
# m_mult * 2*R99 apart so the 99% balls don't overlap (adaptive in d via R99).
# Returns the centers and the separation actually enforced.
# --------------------------------------------------------------------------------------
def sample_modes(K: int, d: int, R: float, sigma: float, m_mult: float = 1.5,
                 seed: int = 0, device: torch.device | None = None):
    min_sep = m_mult * 2 * r99(d, sigma)
    rng = np.random.RandomState(seed)

    modes = np.empty((K, d), dtype=np.float64)
    n, tries, max_tries = 0, 0, 10_000 * K
    while n < K and tries < max_tries:
        tries += 1
        v = rng.randn(d)
        v *= R / np.linalg.norm(v)
        if n and np.linalg.norm(modes[:n] - v, axis=1).min() < min_sep:
            continue
        modes[n] = v
        n += 1

    if n < K:
        raise RuntimeError(
            f"placed only {n}/{K} modes for d={d}, K={K} "
            f"(R={R}, sigma={sigma}, min_sep={min_sep:.3f}); increase R or lower sigma/K"
        )
    return torch.tensor(modes.astype(np.float32), device=device), float(min_sep)


# --------------------------------------------------------------------------------------
# score network
# Residual MLP conditioned on t via a sinusoidal embedding.
# Predicts the noise eps added at step t (not the score itself).
# --------------------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        a = t[:, None].float() * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class MLPBlock(nn.Module):
    def __init__(self, h, act):
        super().__init__()
        self.n = nn.LayerNorm(h)
        self.a = act
        self.f1 = nn.Linear(h, h)
        self.f2 = nn.Linear(h, h)

    def forward(self, x):
        z = self.n(x); z = self.a(z); z = self.f1(z); z = self.a(z); z = self.f2(z)
        return x + z


class ScoreNet(nn.Module):
    H, NB, TD = 256, 4, 128

    def __init__(self, d, h=H, nb=NB, td=TD):
        super().__init__()
        act = nn.LeakyReLU(0.2)
        self.inp = nn.Linear(d, h)
        self.te = SinusoidalPosEmb(td)
        self.tm = nn.Sequential(nn.Linear(td, h), act)
        self.bl = nn.ModuleList([MLPBlock(h, act) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), act, nn.Linear(h, d))

    def forward(self, x, t):
        z = self.inp(x) + self.tm(self.te(t))
        for b in self.bl:
            z = b(z)
        return self.out(z)


# --------------------------------------------------------------------------------------
# learned model: train + sample
# --------------------------------------------------------------------------------------
def train_learned(means_t, d, K, abar, T, variance, n_steps,
                  lr=1e-3, batch=512, seed=0, device=None):
    torch.manual_seed(seed)
    sa = torch.sqrt(abar); soma = torch.sqrt(1 - abar)
    m = ScoreNet(d).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sigma = math.sqrt(variance)
    for _ in range(n_steps):
        k = torch.randint(0, K, (batch,), device=device)
        x0 = means_t[k] + sigma * torch.randn(batch, d, device=device)
        ti = torch.randint(0, T, (batch,), device=device)
        noise = torch.randn_like(x0)
        xt = sa[ti][:, None] * x0 + soma[ti][:, None] * noise
        loss = ((m(xt, ti) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def sample_learned(model, abar, T, N, d, seed, device, chunk=50000):
    """Deterministic DDIM reverse pass. Returns (initial seeds X0, endpoints Xf)."""
    model.eval()
    torch.manual_seed(seed)
    X = torch.randn(N, d, device=device)
    X0 = X.clone()
    for i in reversed(range(1, T)):
        ab, abp = abar[i], abar[i - 1]
        for s in range(0, N, chunk):
            xs = X[s:s + chunk]
            ti = torch.full((xs.shape[0],), i, dtype=torch.long, device=device)
            eps = model(xs, ti)
            x0 = (xs - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            X[s:s + chunk] = torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps
    return X0, X


@torch.no_grad()
def label_fate(Xf, means_t, R99, chunk=50000, device=None):
    N = Xf.shape[0]
    lab = torch.empty(N, dtype=torch.long, device=Xf.device)
    for s in range(0, N, chunk):
        D = torch.cdist(Xf[s:s + chunk], means_t)
        dmin, arg = D.min(1)
        lab[s:s + chunk] = torch.where(dmin <= R99, arg, torch.full_like(arg, -1))
    return lab


# --------------------------------------------------------------------------------------
# true-score sampler + backtrack (the analytic reference process)
#
# The exact score of the GMM, in closed form: no network, no training. Used to build the
# atlas, so its anchors are grounded in the true geometry rather than in whatever the
# learned model happened to fit.
#
# true_score      : s(x) = sum_k w_k (mu_k - x)/v, with w = softmax(-||x-mu_k||^2 / 2v).
#                   Callers pass the noised mixture at level t: means sqrt(ab)*mu,
#                   variance v = ab*variance + (1-ab).
# backtrack_true  : runs the DDIM update forwards in t (noise increasing), driving it with
#                   the true score instead of a network, so data-space points are carried
#                   back to the seeds that would have produced them.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def true_score(X, Mt, v):
    d2 = torch.cdist(X, Mt) ** 2
    w = torch.softmax(-d2 / (2 * v), 1)
    return (w @ Mt - X) / v


@torch.no_grad()
def backtrack_true(Xd, means_t, abar_t, T, variance, chunk=50000):
    X = Xd.clone()
    for i in range(1, T):
        ab, abp = abar_t[i - 1], abar_t[i]
        v = ab * variance + (1 - ab)
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            sc = true_score(xs, torch.sqrt(ab) * means_t, v)
            x0 = (xs + (1 - ab) * sc) / torch.sqrt(ab)
            eps = -torch.sqrt(1 - ab) * sc
            X[s:s + chunk] = torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps
    return X


@torch.no_grad()
def forward_true(X0, means_t, abar_t, T, variance, chunk=50000):
    """Exact-score DDIM FORWARD: seeds (noise) -> data. The inverse of backtrack_true,
    driven by the closed-form GMM score instead of a network. Used to label seeds by the
    exact score (the analytic reference), e.g. altered_knn's calibration set."""
    X = X0.clone()
    for i in reversed(range(1, T)):
        ab, abp = abar_t[i], abar_t[i - 1]
        v = ab * variance + (1 - ab)
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            sc = true_score(xs, torch.sqrt(ab) * means_t, v)
            eps = -torch.sqrt(1 - ab) * sc
            x0 = (xs - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            X[s:s + chunk] = torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps
    return X


# --------------------------------------------------------------------------------------
# labelled anchors planted in DATA space (no model, no backtrack).
#   ball_anchors        : mode balls + a thin band just outside -> labels {k, -1}, for the
#                         knn / polar predictors.
#   altered_knn_anchors : mode-only concentric spheres with radius-decaying weights (no
#                         hallucination class), for the altered_knn predictor.
# Callers backtrack these to seed space with the exact field before fitting.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def ball_anchors(means_t, R99, n_per_mode, shell_frac=0.5, shell_sigma=2.0, sigma=None,
                 seed=0, device=None):
    """Per mode: n_per_mode points uniform in the R99 ball, plus shell_frac * n_per_mode
    points uniform in radius over R99 .. R99 + shell_sigma * sigma. Every anchor is coloured
    with the ground-truth rule L (nearest mode within R99, else -1)."""
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
    """Per mode: n_rings concentric spheres of radius r_j = R99 * r_max * j / n_rings, each
    labelled with its mode, weighted by a radius-decaying confidence (linear or gaussian).
    Returns P (n, d), y (n,), w (n,); no hallucination class."""
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
    return (torch.cat(P, 0), torch.tensor(y, device=device),
            torch.tensor(w, device=device, dtype=torch.float32))


