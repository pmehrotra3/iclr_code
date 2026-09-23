"""core.py — the numerics shared by every stage.

    device, repeat seeds              get_device, seed_list
    the GMM                           r99, sample_modes, mode_weights, gmm_train_set, label_fate
    the learned model                 ScoreNet, net_arch, learned_closure (DDIM eps-loss)
    the training recipe               run_optimizers (Adam + cosine lr + clip + EMA, CUDA graph)
    the exact (closed-form) sampler   true_score, backtrack_true (data -> seed), forward_true
    labelled data-space anchors       ball_anchors, altered_knn_anchors

A seed x ~ N(0, I) has a FATE: the mode k whose R99 ball its sample lands in, or -1
(hallucination) if it lands in none.
"""
from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2


# --------------------------------------------------------------------------------------
# device and repeat seeds
# --------------------------------------------------------------------------------------
SEED_STRIDE = 100      # repeats use seed, seed+100, ...: per-purpose offsets (+1, +2, +7) never collide


def get_device(pref: str = "auto") -> torch.device:
    """'cpu', 'cuda' (error without CUDA) or 'auto' (the GPU if one is visible)."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested but CUDA is not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_list(cfg) -> list[int]:
    """The cfg.n_seeds independent repeats; each re-draws modes, model init and eval seeds."""
    return [int(cfg.seed) + SEED_STRIDE * i for i in range(int(cfg.n_seeds or 1))]


# --------------------------------------------------------------------------------------
# the Gaussian mixture: K isotropic modes N(mu_k, sigma^2 I) on a sphere of radius R
# --------------------------------------------------------------------------------------
def make_schedule(T: int = 1000, beta_min: float = 0.001, beta_max: float = 0.02,
                  device=None) -> torch.Tensor:
    """abar[i] = prod_{j<=i} (1 - beta_j) for linear betas: x_i = sqrt(abar) x0 + sqrt(1-abar) eps."""
    abar = np.cumprod(1.0 - np.linspace(beta_min, beta_max, T))
    return torch.tensor(abar, dtype=torch.float32, device=device)


def r99(d: int, sigma: float, q: float = 0.99) -> float:
    """Radius of the ball holding mass q of N(0, sigma^2 I_d) (chi-square): the mode boundary."""
    return float(np.sqrt(chi2.ppf(q, d)) * sigma)


def sample_modes(K, d, R, sigma, m_mult=1.5, seed=0, device=None):
    """K means on the sphere of radius R, rejection-sampled so every pair is >= m_mult * 2 R99
    apart (the 99% balls never overlap). Returns (means (K, d), min_sep)."""
    min_sep = m_mult * 2 * r99(d, sigma)
    rng = np.random.RandomState(seed)
    modes = np.empty((K, d), dtype=np.float64)
    n, tries = 0, 0
    while n < K and tries < 10_000 * K:
        tries += 1
        v = rng.randn(d)
        v *= R / np.linalg.norm(v)
        if n and np.linalg.norm(modes[:n] - v, axis=1).min() < min_sep:
            continue
        modes[n] = v
        n += 1
    if n < K:
        raise RuntimeError(f"placed only {n}/{K} modes for d={d}, K={K} (R={R}, sigma={sigma}, "
                           f"min_sep={min_sep:.3f}); increase R or lower sigma/K")
    return torch.tensor(modes.astype(np.float32), device=device), float(min_sep)


WEIGHT_LO, WEIGHT_HI = 0.3, 0.8   # raw mixing weights are drawn from U[WEIGHT_LO, WEIGHT_HI]


def mode_weights(K, seed, weighted, device=None, base_seed=0):
    """Mixing weights (K,), uniform unless `weighted`. Weighted runs draw a single weight
    profile per (K, base_seed) and let each repeat `seed` hand it out to the modes in its own
    random order, so the spread over seeds comes from where the weights land, not from new
    weights. Drawn on the CPU so the device does not matter."""
    if not weighted:
        return torch.full((K,), 1.0 / K, device=device)
    g = torch.Generator().manual_seed(int(base_seed) + 13)
    w = WEIGHT_LO + (WEIGHT_HI - WEIGHT_LO) * torch.rand(K, generator=g)
    w = w / w.sum()
    g = torch.Generator().manual_seed(int(seed) + 17)
    return w[torch.randperm(K, generator=g)].to(device)


def _draw_modes(K, n, weights, g, device):
    """n mode indices, uniform (weights None) or multinomial under `weights`."""
    if weights is None:
        return torch.randint(0, K, (n,), generator=g, device=device)
    return torch.multinomial(weights.to(device), n, replacement=True, generator=g)


def gmm_train_set(means_t, n, variance, seed, device=None, weights=None):
    """A fixed training set of n GMM draws, reproducible per seed. Returns (X, mode index)."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(int(seed) + 11)
    k = _draw_modes(K, n, weights, g, device)
    X = means_t[k] + math.sqrt(variance) * torch.randn(n, d, generator=g, device=device)
    return X, k


def _minibatch(means_t, variance, batch, n_train, seed, device, weights=None, g=None):
    """Returns draw(), which gives one clean minibatch (batch, d): resampled from a fixed
    training set of n_train points, or fresh mixture samples every step if n_train is None.
    All randomness comes from the generator `g`."""
    K, d = means_t.shape
    if n_train:
        X, _ = gmm_train_set(means_t, int(n_train), variance, seed, device, weights)

        def draw():
            return X.index_select(0, torch.randint(0, X.shape[0], (batch,), generator=g, device=device))
    else:
        sigma = math.sqrt(variance)
        w = None if weights is None else weights.to(device)

        def draw():
            k = _draw_modes(K, batch, w, g, device)
            return means_t[k] + sigma * torch.randn(batch, d, generator=g, device=device)
    return draw


@torch.no_grad()
def label_fate(Xf, means_t, R99, chunk=50000):
    """Fate of each endpoint: its nearest mode if within R99 of it, else -1."""
    lab = torch.empty(Xf.shape[0], dtype=torch.long, device=Xf.device)
    for s in range(0, Xf.shape[0], chunk):
        dmin, arg = torch.cdist(Xf[s:s + chunk], means_t).min(1)
        lab[s:s + chunk] = torch.where(dmin <= R99, arg, torch.full_like(arg, -1))
    return lab


# --------------------------------------------------------------------------------------
# the learned model: a residual MLP on (x, t) that predicts the noise (DDIM) or velocity (flow)
# (submodule names are the checkpoint state_dict keys: keep them)
# --------------------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        a = t[:, None].float() * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class MLPBlock(nn.Module):
    """x + Linear(act(Linear(act(LayerNorm(x)))))."""

    def __init__(self, h, act):
        super().__init__()
        self.n, self.a = nn.LayerNorm(h), act
        self.f1, self.f2 = nn.Linear(h, h), nn.Linear(h, h)

    def forward(self, x):
        return x + self.f2(self.a(self.f1(self.a(self.n(x)))))


class ScoreNet(nn.Module):
    H, NB, TD = 256, 4, 128          # defaults: hidden width, residual blocks, time-embedding dim

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


def net_size(d, cfg):
    """'small' for d <= train.small_max_d, else 'large': one fixed width cannot resolve the
    mode ball in high d (the R99 margin shrinks like 1/sqrt(d), the model's error does not)."""
    return "small" if d <= int(cfg.train.small_max_d) else "large"


def net_arch(d, cfg):
    """ScoreNet kwargs for dimension d: width train.width[net_size(d)], default depth."""
    return {"h": int(cfg.train.width[net_size(d, cfg)]), "nb": ScoreNet.NB, "td": ScoreNet.TD}


def step_generator(seed, device):
    """The private RNG of one training run (attached as `closure.generator`). Every random draw
    of a step comes from it, never from the global RNG, so a seed trains to the same model
    alone or grouped with other seeds (eager loop or one CUDA graph)."""
    return torch.Generator(device=device).manual_seed(int(seed) + 29)


def learned_closure(means_t, d, K, abar, T, variance, batch=512, seed=0, device=None,
                    n_train=None, weights=None, arch=None):
    """A fresh ScoreNet (init seeded by `seed`) and its DDIM loss closure: MSE between the
    network's noise prediction and the noise added to a minibatch at uniform random steps."""
    torch.manual_seed(seed)
    sa, soma = torch.sqrt(abar), torch.sqrt(1 - abar)
    m = ScoreNet(d, **(arch or {})).to(device)
    g = step_generator(seed, device)
    draw = _minibatch(means_t, variance, batch, n_train, seed, device, weights, g)

    def step():
        x0 = draw()
        ti = torch.randint(0, T, (batch,), generator=g, device=device)
        noise = torch.randn(x0.shape, generator=g, device=device)
        xt = sa[ti][:, None] * x0 + soma[ti][:, None] * noise
        return ((m(xt, ti) - noise) ** 2).mean()

    step.generator = g
    return m, step


# --------------------------------------------------------------------------------------
# the training recipe: Adam + cosine lr decay (lr -> lr_min) + grad-norm clip + an EMA of the
# weights that replaces them at the end. Any piece is off when its setting is None / 0.
# --------------------------------------------------------------------------------------
def _ema_decay(step: int, decay: float, warmup: int) -> float:
    """EMA decay, ramped up from ~0 over the first `warmup` steps."""
    if warmup and step < warmup:
        return min(decay, (1.0 + step) / (1.0 + warmup))
    return decay


def _cosine_lr(step: int, n_steps: int, lr: float, lr_min) -> float:
    """The lr at `step` of torch's CosineAnnealingLR(T_max=n_steps, eta_min=lr_min)."""
    if lr_min is None or n_steps <= 1:
        return lr
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * step / n_steps))


def run_optimizers(models, loss_closures, n_steps, lr=1e-3, lr_min=None, grad_clip=None,
                   ema_decay=None, ema_warmup=0, cuda_graph=True, branch_streams=True):
    """Train several independent models for `n_steps` steps each, model i on loss_closures[i]().
    Returns the models (with their EMA weights when ema_decay is set).

    On a GPU, one training step of all the models is recorded once as a CUDA graph (each model
    on its own stream) and then replayed. The networks are small, so a plain Python loop spends
    most of its time launching kernels; the graph does the same maths about 15x faster. The CPU / cuda_graph=False path is the
    plain eager loop, the reference. branch_streams=False runs the branches one after another:
    required when several such processes share a GPU (multi-stream graphs from concurrent
    processes crash with 'illegal memory access' on driver 565 / torch 2.5)."""
    assert len(models) == len(loss_closures) and models
    if cuda_graph and next(models[0].parameters()).device.type == "cuda":
        return _run_optimizers_graph(models, loss_closures, n_steps, lr, lr_min, grad_clip,
                                     ema_decay, ema_warmup, branch_streams)
    return [_run_optimizer_eager(m, c, n_steps, lr, lr_min, grad_clip, ema_decay, ema_warmup)
            for m, c in zip(models, loss_closures)]


def _run_optimizer_eager(model, loss_closure, n_steps, lr, lr_min, grad_clip, ema_decay, ema_warmup):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    if lr_min is not None and n_steps > 1:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr_min)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema_decay else None
    for step in range(n_steps):
        loss = loss_closure()
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if ema is not None:
            dcy = _ema_decay(step, ema_decay, ema_warmup)
            live = model.state_dict()
            with torch.no_grad():
                for k, v in ema.items():
                    if v.dtype.is_floating_point:
                        v.mul_(dcy).add_(live[k].detach(), alpha=1 - dcy)
                    else:
                        v.copy_(live[k])
    if ema is not None:
        model.load_state_dict(ema)
    return model


class _GraphBranch:
    """One model's step inside the graph: capturable Adam, clip, foreach EMA. The lr / EMA decay
    of the current step are read from shared device tables (nothing syncs with the host)."""

    def __init__(self, model, loss_closure, lr_tab, dcy_tab, grad_clip):
        self.model, self.closure, self.grad_clip = model, loss_closure, grad_clip
        self.params = list(model.parameters())
        self.lr_tab, self.dcy_tab = lr_tab, dcy_tab
        self.lr_t = lr_tab[0].clone()
        self.opt = torch.optim.Adam(self.params, lr=self.lr_t, foreach=True, capturable=True)
        self.ema_f = None
        if dcy_tab is not None:
            self.dcy_t = dcy_tab[0].clone()
            self.omd_t = 1.0 - self.dcy_t
            sd = model.state_dict()                              # views of the live weights
            self.ema_keys = list(sd.keys())
            self.live_f = [v for v in sd.values() if v.dtype.is_floating_point]
            self.ema_f = [v.detach().clone() for v in self.live_f]
            self.ema_o = {k: (v, v.detach().clone()) for k, v in sd.items()
                          if not v.dtype.is_floating_point}

    def step(self, step_t):
        self.lr_t.copy_(self.lr_tab.index_select(0, step_t).squeeze(0))
        self.closure().backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip, foreach=True)
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)                    # the graph owns the grad buffers
        if self.ema_f is not None:
            self.dcy_t.copy_(self.dcy_tab.index_select(0, step_t).squeeze(0))
            self.omd_t.copy_(1.0 - self.dcy_t)
            with torch.no_grad():
                torch._foreach_mul_(self.ema_f, self.dcy_t)
                torch._foreach_add_(self.ema_f, torch._foreach_mul(self.live_f, self.omd_t))
                for v, e in self.ema_o.values():
                    e.copy_(v)

    def finish(self):
        if self.ema_f is not None:
            float_w = iter(self.ema_f)
            sd = self.model.state_dict()
            self.model.load_state_dict({k: next(float_w) if sd[k].dtype.is_floating_point
                                        else self.ema_o[k][1] for k in self.ema_keys})
        return self.model


def _run_optimizers_graph(models, loss_closures, n_steps, lr, lr_min, grad_clip,
                          ema_decay, ema_warmup, branch_streams=True, n_warmup=3):
    """CUDA-graph form of the eager loop (same per-step maths)."""
    device = next(models[0].parameters()).device
    n_warmup = min(n_warmup, n_steps)
    lr_tab = torch.tensor([_cosine_lr(s, n_steps, lr, lr_min) for s in range(n_steps)],
                          dtype=torch.float32, device=device)
    dcy_tab = None
    if ema_decay:
        dcy_tab = torch.tensor([_ema_decay(s, ema_decay, ema_warmup) for s in range(n_steps)],
                               dtype=torch.float32, device=device)
    step_t = torch.zeros((1,), dtype=torch.long, device=device)   # on-device step counter
    branches = [_GraphBranch(m, c, lr_tab, dcy_tab, grad_clip) for m, c in zip(models, loss_closures)]
    streams = [torch.cuda.Stream() if branch_streams else None for _ in branches]

    def one_round(parent):
        # fork every branch off the parent stream, join them back, advance the counter
        for b, s in zip(branches, streams):
            if s is None:
                b.step(step_t)
                continue
            s.wait_stream(parent)
            with torch.cuda.stream(s):
                b.step(step_t)
        for s in streams:
            if s is not None:
                parent.wait_stream(s)
        step_t.add_(1)

    for _ in range(n_warmup):                  # real steps that allocate grads / Adam state
        one_round(torch.cuda.current_stream())
    if n_steps > n_warmup:
        g = torch.cuda.CUDAGraph()
        for b in branches:                     # replays must advance each run's private RNG
            if getattr(b.closure, "generator", None) is not None:
                g.register_generator_state(b.closure.generator)
        with torch.cuda.graph(g):
            one_round(torch.cuda.current_stream())
        for _ in range(n_steps - n_warmup):
            g.replay()
    torch.cuda.synchronize()
    return [b.finish() for b in branches]


# --------------------------------------------------------------------------------------
# the exact sampler: the GMM score in closed form (no network), used to build the atlas
#   s(x) = sum_k r_k (mu_k - x) / v,  r = softmax_k(log pi_k - |x - mu_k|^2 / 2v)
# at noise level abar the noised mixture has means sqrt(abar) mu_k, variance abar sigma^2 + 1 - abar
# --------------------------------------------------------------------------------------
@torch.no_grad()
def true_score(X, Mt, v, logw=None):
    logits = -torch.cdist(X, Mt) ** 2 / (2 * v)
    if logw is not None:
        logits = logits + logw
    return (torch.softmax(logits, 1) @ Mt - X) / v


def log_weights(weights, device=None):
    """log pi (1, K) for true_score / the flow velocity, or None when weights is None."""
    if weights is None:
        return None
    w = weights.to(device) if device is not None else weights
    return torch.log(w.clamp_min(1e-30))[None, :]


@torch.no_grad()
def _exact_eps(x, means_t, ab, variance, logw=None):
    """The exact score at noise level `ab`, as an eps-prediction."""
    v = ab * variance + (1 - ab)
    return -torch.sqrt(1 - ab) * true_score(x, torch.sqrt(ab) * means_t, v, logw)


@torch.no_grad()
def _ddim_step(x, eps, ab, abp):
    """One deterministic DDIM update from noise level `ab` to `abp`."""
    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps


@torch.no_grad()
def _ddim_transport(Xd, means_t, order, levels, variance, chunk=50000, logw=None, inplace=False):
    """Carry points along the exact-score DDIM map over `levels` [(ab, abp), ...].
    'euler' uses eps at the start; 'heun' averages it with eps at the predicted end point
    (second order: data -> seed -> data round-trips to the same label). inplace=True reuses
    Xd's memory (multi-GB anchor sets)."""
    heun = str(order).lower() == "heun"
    X = Xd if inplace else Xd.clone()
    for ab, abp in levels:
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            eps = _exact_eps(xs, means_t, ab, variance, logw)
            if heun:
                eps = 0.5 * (eps + _exact_eps(_ddim_step(xs, eps, ab, abp), means_t, abp, variance, logw))
            X[s:s + chunk] = _ddim_step(xs, eps, ab, abp)
    return X


@torch.no_grad()
def backtrack_true(Xd, means_t, abar_t, T, variance, chunk=50000, order="heun", weights=None,
                   inplace=False):
    """Data -> seed under the exact score (noise increasing)."""
    levels = [(abar_t[i - 1], abar_t[i]) for i in range(1, T)]
    return _ddim_transport(Xd, means_t, order, levels, variance, chunk,
                           log_weights(weights, Xd.device), inplace)


@torch.no_grad()
def forward_true(X0, means_t, abar_t, T, variance, chunk=50000, order="heun", weights=None):
    """Seed -> data under the exact score: the inverse of backtrack_true."""
    levels = [(abar_t[i], abar_t[i - 1]) for i in reversed(range(1, T))]
    return _ddim_transport(X0, means_t, order, levels, variance, chunk, log_weights(weights, X0.device))


# --------------------------------------------------------------------------------------
# labelled anchors, placed around the modes in data space (callers carry them back to seeds)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def ball_anchors(means_t, R99, n_per_mode, shell_frac=0.5, shell_sigma=2.0, sigma=None,
                 seed=0, device=None):
    """Per mode: n_per_mode points uniform in the R99 ball plus shell_frac * n_per_mode points in
    the band R99 .. R99 + shell_sigma * sigma, labelled by label_fate (so {k, -1}). One tensor
    filled in place (at d=512 it can be ~15 GB)."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(seed)
    n_shell = max(1, int(round(shell_frac * n_per_mode)))
    per_mode = n_per_mode + n_shell
    P = torch.empty(K * per_mode, d, device=device)
    for k in range(K):
        s0 = k * per_mode
        u = torch.rand(n_per_mode, generator=g, device=device) ** (1.0 / d)     # uniform in the ball
        dirs = torch.randn(n_per_mode, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        torch.addcmul(means_t[k], (R99 * u)[:, None], dirs, out=P[s0:s0 + n_per_mode])
        r = R99 + shell_sigma * sigma * torch.rand(n_shell, generator=g, device=device)
        dirs = torch.randn(n_shell, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        torch.addcmul(means_t[k], r[:, None], dirs, out=P[s0 + n_per_mode:s0 + per_mode])
    return P, label_fate(P, means_t, R99)


@torch.no_grad()
def altered_knn_anchors(means_t, R99, n_per_mode, n_rings=5, r_max=1.5, weight="linear",
                        w_min=0.2, sigma=None, seed=0, device=None):
    """Per mode: n_rings spheres of radius R99 * r_max * j / n_rings, all labelled with the mode
    (no hallucination class) and weighted by a radius-decaying confidence (linear | gaussian).
    Returns P (n, d), y (n,), w (n,)."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(seed)
    per_ring = max(1, n_per_mode // n_rings)
    per_mode = n_rings * per_ring
    radii = [R99 * r_max * j / n_rings for j in range(1, n_rings + 1)]
    if weight == "gaussian":
        w_ring = [float(np.exp(-r ** 2 / (2 * sigma ** 2))) for r in radii]
    else:
        w_ring = [1.0 - (1.0 - w_min) * (r / (R99 * r_max)) for r in radii]
    P = torch.empty(K * per_mode, d, device=device)
    y = torch.arange(K, device=device).repeat_interleave(per_mode)
    w = torch.tensor(w_ring, device=device, dtype=torch.float32).repeat_interleave(per_ring).repeat(K)
    for k in range(K):
        for j, r in enumerate(radii):
            dirs = torch.randn(per_ring, d, generator=g, device=device)
            dirs /= dirs.norm(dim=1, keepdim=True)
            s0 = k * per_mode + j * per_ring
            blk = P[s0:s0 + per_ring]
            torch.mul(dirs, r, out=blk); blk += means_t[k]      # == means_t[k] + r * dirs, bit for bit
    return P, y, w
