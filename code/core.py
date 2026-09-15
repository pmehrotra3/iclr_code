"""
core.py — shared building blocks for the diffusion-atlas experiments.

Everything that more than one stage needs lives here: the score network, the VP
schedule, GMM mode sampling, the learned + true-score samplers, the true-score
atlas builder, the all-anchor Gaussian vote, and the analytic-responsibility
predictor. The stage files (train / evaluate / visualize) import from here so the
numerics are defined exactly once.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2


# --------------------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------------------
def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# VP schedule
# --------------------------------------------------------------------------------------
def make_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 0.02,
                  device: torch.device | None = None) -> torch.Tensor:
    betas = np.linspace(beta_min, beta_max, T)
    abar = np.cumprod(1.0 - betas)
    return torch.tensor(abar, dtype=torch.float32, device=device)


def r99(d: int, sigma: float, q: float = 0.99) -> float:
    """Radius of the ball that holds fraction q of an isotropic Gaussian mode."""
    return float(np.sqrt(chi2.ppf(q, d)) * sigma)


# --------------------------------------------------------------------------------------
# GMM modes
# --------------------------------------------------------------------------------------
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
    if len(modes) < K:
        raise RuntimeError(f"could only place {len(modes)}/{K} modes for d={d}, K={K}")
    M = np.asarray(modes, dtype=np.float32)
    return torch.tensor(M, device=device), float(mult)


# --------------------------------------------------------------------------------------
# score network
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
    def __init__(self, d, h=256, nb=4, td=128):
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
# --------------------------------------------------------------------------------------
@torch.no_grad()
def true_score(X, Mt, v):
    d2 = torch.cdist(X, Mt) ** 2
    w = torch.softmax(-d2 / (2 * v), 1)
    return (w @ Mt - X) / v


@torch.no_grad()
def backtrack_true(Xd, means_t, abar_t, T, variance, chunk=50000):
    """Carry data-space points back to the initial noise via the TRUE-score reverse ODE."""
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


# --------------------------------------------------------------------------------------
# true-score atlas: filled disk (mode) + shell (hallucination), backtracked
# --------------------------------------------------------------------------------------
@torch.no_grad()
def build_atlas(means_t, R99, K, d, n_disk, abar_true, T_true, variance,
                shell_w=1.0, device=None):
    n_shell = max(1, n_disk // 2)
    seeds_all, labs_all = [], []
    for k in range(K):
        u = torch.rand(n_disk, device=device) ** (1.0 / d)
        r_in = R99 * u
        dirs = torch.randn(n_disk, d, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        disk = means_t[k] + r_in[:, None] * dirs
        seeds_all.append(backtrack_true(disk, means_t, abar_true, T_true, variance))
        labs_all += [k] * n_disk

        r_sh = R99 * (1 + shell_w * torch.rand(n_shell, device=device))
        d2 = torch.randn(n_shell, d, device=device)
        d2 /= d2.norm(dim=1, keepdim=True)
        shell = means_t[k] + r_sh[:, None] * d2
        seeds_all.append(backtrack_true(shell, means_t, abar_true, T_true, variance))
        labs_all += [-1] * n_shell
    return torch.cat(seeds_all, 0), torch.tensor(labs_all, device=device)


@torch.no_grad()
def gauss_vote_all(Xq, anchors, alabels, K, R99, d=None, h_frac=0.3,
                   q_chunk=2048, a_chunk=100000, device=None):
    """ALL anchors vote, weighted by exp(-dist^2 / 2h^2). No k, no topk."""
    na = anchors.shape[0]
    if d is None:
        d = anchors.shape[1]
    h = max(h_frac * R99, 1e-4)
    inv2h2 = 1.0 / (2 * h * h)
    if na > 500000:
        q_chunk = 256
    elif na > 200000:
        q_chunk = 512
    elif na > 50000:
        q_chunk = 1024
    Nq = Xq.shape[0]
    pred = torch.empty(Nq, dtype=torch.long, device=Xq.device)
    ls = alabels + 1
    nc = K + 1
    for qs in range(0, Nq, q_chunk):
        q = Xq[qs:qs + q_chunk]
        B = q.shape[0]
        votes = torch.zeros(B, nc, device=Xq.device)
        for a0 in range(0, na, a_chunk):
            A = anchors[a0:a0 + a_chunk]
            dm = torch.cdist(q, A)
            w = torch.exp(-dm * dm * inv2h2)
            lab = ls[a0:a0 + a_chunk][None, :].expand(B, -1)
            votes.scatter_add_(1, lab, w)
            del dm, w
        pred[qs:qs + q_chunk] = votes.argmax(1) - 1
    return pred, na, float(h)


# --------------------------------------------------------------------------------------
# analytic responsibility predictor (no anchors) — the dimension-robust baseline
# --------------------------------------------------------------------------------------
@torch.no_grad()
def responsibilities(X0, means_t, ab, variance):
    v = float(ab * variance + (1 - ab))
    sM = torch.sqrt(ab) * means_t
    d2 = torch.cdist(X0, sM) ** 2
    return torch.softmax(-d2 / (2 * v), 1)


@torch.no_grad()
def predict_responsibility(X0, means_t, abar, variance, delta=0.1):
    """Label seeds from the initial noise alone via source-scale responsibility."""
    ab = abar[-1]
    w = responsibilities(X0, means_t, ab, variance)
    pmax, arg = w.max(1)
    return torch.where(pmax >= 1 - delta, arg, torch.full_like(arg, -1)), w
