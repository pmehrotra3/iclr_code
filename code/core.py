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
# repeat seeds
# The sweep is repeated over cfg.n_seeds independent seeds, spaced 100 apart from cfg.seed so
# the per-purpose offsets (+1 eval, +2 calibration, +7 probe) of different repeats never
# coincide. Each repeat re-samples the mode placement, the model init and the eval seeds.
# --------------------------------------------------------------------------------------
SEED_STRIDE = 100


def seed_list(cfg) -> list[int]:
    n = int(getattr(cfg, "n_seeds", 1) or 1)
    return [int(cfg.seed) + SEED_STRIDE * i for i in range(n)]


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
# optimisation helper: Adam + cosine lr decay + grad-norm clip + EMA weights
#
# The three additions from the improved recipe (Step 1) live here so every process shares
# them: the learned DDIM score and the flow velocity both call run_optimizer with their own
# per-step loss closure. Cosine-decays lr from `lr` to `lr_min`, clips the gradient norm to
# `grad_clip`, and keeps an exponential moving average of the weights (decay `ema_decay`,
# warmed up over `ema_warmup` steps) that REPLACES the raw weights at the end. Any of the
# three is disabled by passing None / 0, which recovers the old plain-Adam loop.
# --------------------------------------------------------------------------------------
def _ema_decay(step: int, decay: float, warmup: int) -> float:
    """EMA decay with a warm-up ramp: rises from ~0 to `decay` over the first `warmup` steps."""
    if warmup and step < warmup:
        return min(decay, (1.0 + step) / (1.0 + warmup))
    return decay


def _cosine_lr(step: int, n_steps: int, lr: float, lr_min: float | None) -> float:
    """Per-step lr of torch's CosineAnnealingLR(T_max=n_steps, eta_min=lr_min): the value in
    force at `step` when the scheduler is stepped once after every optimizer step."""
    if lr_min is None or n_steps <= 1:
        return lr
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * step / n_steps))


def run_optimizer(model, loss_closure, n_steps, lr=1e-3, lr_min=None, grad_clip=None,
                  ema_decay=None, ema_warmup=0, cuda_graph=True, branch_streams=True):
    """Optimise `model` for `n_steps` steps; loss_closure() returns the scalar loss each step.

    Returns the model with EMA weights loaded (when ema_decay is set). One-model wrapper of
    run_optimizers (see there for the CUDA-graph fast path)."""
    return run_optimizers([model], [loss_closure], n_steps, lr, lr_min, grad_clip,
                          ema_decay, ema_warmup, cuda_graph, branch_streams)[0]


def run_optimizers(models, loss_closures, n_steps, lr=1e-3, lr_min=None, grad_clip=None,
                   ema_decay=None, ema_warmup=0, cuda_graph=True, branch_streams=True):
    """Optimise several independent models for the same `n_steps` steps, each with its own
    loss_closure() (Adam + cosine lr decay + grad-norm clip + EMA, identically per model).
    Returns the models with EMA weights loaded (when ema_decay is set).

    On CUDA the whole step of EVERY model (closure + backward + clip + Adam + EMA) is captured
    into ONE CUDA graph, each model on its own stream as a parallel branch, and that graph is
    replayed n_steps times. The networks here are small MLPs at batch ~512: the eager loop is
    bound by kernel-launch overhead (~6 ms of CPU per ~0.5 ms of GPU work) and one model's
    kernels are too small to fill the GPU, so the graph removes the per-step Python and the
    parallel branches overlap the models' kernels (~6x from the graph, ~2x more from 3-4
    branches; same maths). The lr and EMA-decay schedules are precomputed into device tables
    read inside the graph through an on-device step counter, so nothing syncs. A single graph
    is used on purpose: separately-replayed graphs racing on multiple streams are not safe
    (shared RNG state), one graph with branches is. `cuda_graph=False` (or a CPU device)
    runs the plain eager loop, model after model, which is the reference for what the graph
    computes.

    branch_streams=False puts every model on the capture stream instead (branches run one
    after another inside the graph; ~2x slower for 3+ models but with no stream concurrency).
    Use it whenever MORE THAN ONE such training process shares a GPU: with two or more
    concurrent processes each replaying a multi-stream graph on the same GPU (driver 565,
    torch 2.5) a fraction of them die with "illegal memory access", while one process per GPU
    (alongside unrelated work) and the single-stream form are both stable -- see
    scripts/main.sh, which sets this from JOBS_PER_GPU."""
    assert len(models) == len(loss_closures) and models
    device = next(models[0].parameters()).device
    if cuda_graph and device.type == "cuda":
        return _run_optimizers_graph(models, loss_closures, n_steps, lr, lr_min, grad_clip,
                                     ema_decay, ema_warmup, branch_streams)
    return [_run_optimizer_eager(m, c, n_steps, lr, lr_min, grad_clip, ema_decay, ema_warmup)
            for m, c in zip(models, loss_closures)]


def _run_optimizer_eager(model, loss_closure, n_steps, lr, lr_min, grad_clip, ema_decay, ema_warmup):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    if lr_min is not None and n_steps > 1:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr_min)
    ema = None
    if ema_decay:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
            msd = model.state_dict()
            with torch.no_grad():
                for k, v in ema.items():
                    if v.dtype.is_floating_point:
                        v.mul_(dcy).add_(msd[k].detach(), alpha=1 - dcy)
                    else:
                        v.copy_(msd[k])
    if ema is not None:
        model.load_state_dict(ema)
    return model


class _GraphBranch:
    """One model's per-step work for _run_optimizers_graph: capturable Adam, grad clip and a
    foreach EMA, with the lr / decay of the current step read from shared device tables."""

    def __init__(self, model, loss_closure, lr_tab, dcy_tab, grad_clip):
        self.model, self.closure, self.grad_clip = model, loss_closure, grad_clip
        self.params = list(model.parameters())
        self.lr_tab, self.dcy_tab = lr_tab, dcy_tab
        self.lr_t = lr_tab[0].clone()
        # capturable Adam takes the lr as a device tensor and never syncs on its step count
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
        loss = self.closure()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip, foreach=True)
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)                    # keep the grad buffers the graph owns
        if self.ema_f is not None:
            self.dcy_t.copy_(self.dcy_tab.index_select(0, step_t).squeeze(0))
            self.omd_t.copy_(1.0 - self.dcy_t)
            with torch.no_grad():
                torch._foreach_mul_(self.ema_f, self.dcy_t)
                torch._foreach_add_(self.ema_f, torch._foreach_mul(self.live_f, self.omd_t))
                for v, e in self.ema_o.values():
                    e.copy_(v)

    def finish(self):
        if self.ema_f is None:
            return self.model
        sd = self.model.state_dict()
        it = iter(self.ema_f)
        out = {k: (next(it) if sd[k].dtype.is_floating_point else self.ema_o[k][1])
               for k in self.ema_keys}
        self.model.load_state_dict(out)
        return self.model


def _run_optimizers_graph(models, loss_closures, n_steps, lr, lr_min, grad_clip,
                          ema_decay, ema_warmup, branch_streams=True, n_warmup=3):
    """CUDA-graph version of the loop (see run_optimizers). Same per-step semantics as the
    eager loop: lr(step) = cosine schedule, EMA decay(step) = _ema_decay, clip before Adam."""
    device = next(models[0].parameters()).device
    n_warmup = min(n_warmup, n_steps)

    # schedules as device tables, shared by every branch; `step_t` indexes them from inside
    # the graph
    lr_tab = torch.tensor([_cosine_lr(s, n_steps, lr, lr_min) for s in range(n_steps)],
                          dtype=torch.float32, device=device)
    dcy_tab = None
    if ema_decay:
        dcy_tab = torch.tensor([_ema_decay(s, ema_decay, ema_warmup) for s in range(n_steps)],
                               dtype=torch.float32, device=device)
    step_t = torch.zeros((1,), dtype=torch.long, device=device)
    branches = [_GraphBranch(m, c, lr_tab, dcy_tab, grad_clip) for m, c in zip(models, loss_closures)]
    streams = [torch.cuda.Stream() if branch_streams else None for _ in branches]

    def one_round(parent):
        # fork: every branch waits for the parent stream, runs its step on its own stream;
        # join: the parent waits for all of them, then the shared counter advances
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

    # warm-up rounds allocate grads / Adam state before capture (these are real steps)
    main = torch.cuda.current_stream()
    for _ in range(n_warmup):
        one_round(main)

    if n_steps > n_warmup:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):                                # records the round; does not run it
            one_round(torch.cuda.current_stream())
        for _ in range(n_steps - n_warmup):
            g.replay()
    torch.cuda.synchronize()
    return [b.finish() for b in branches]


# --------------------------------------------------------------------------------------
# learned model: train + sample
# --------------------------------------------------------------------------------------
def gmm_train_set(means_t, n, variance, seed, device=None):
    """A fixed training set: n i.i.d. draws from the GMM (uniform mode, isotropic variance),
    seeded so it is reproducible per (d, K, seed). Returns (X (n, d), mode index (n,))."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(int(seed) + 11)
    k = torch.randint(0, K, (n,), generator=g, device=device)
    X = means_t[k] + math.sqrt(variance) * torch.randn(n, d, generator=g, device=device)
    return X, k


def _minibatch(means_t, variance, batch, n_train, seed, device):
    """Return a sampler of clean minibatches x0 (batch, d): from a fixed training set of
    n_train GMM draws (minibatches re-drawn with replacement, so each step is a random subset),
    or fresh i.i.d. draws every step when n_train is None (the infinite-data regime)."""
    K, d = means_t.shape
    sigma = math.sqrt(variance)
    if n_train:
        X, _ = gmm_train_set(means_t, int(n_train), variance, seed, device)

        def draw():
            idx = torch.randint(0, X.shape[0], (batch,), device=device)
            return X.index_select(0, idx)
    else:
        def draw():
            k = torch.randint(0, K, (batch,), device=device)
            return means_t[k] + sigma * torch.randn(batch, d, device=device)
    return draw


def learned_closure(means_t, d, K, abar, T, variance, batch=512, seed=0, device=None,
                    n_train=None):
    """A fresh ScoreNet (init seeded by `seed`) and its eps-prediction loss closure: one
    minibatch of noised GMM samples at random steps, MSE against the noise. The clean samples
    come from a fixed n_train-sample training set (or fresh draws when n_train is None)."""
    torch.manual_seed(seed)
    sa = torch.sqrt(abar); soma = torch.sqrt(1 - abar)
    m = ScoreNet(d).to(device)
    draw = _minibatch(means_t, variance, batch, n_train, seed, device)

    def step():
        x0 = draw()
        ti = torch.randint(0, T, (batch,), device=device)
        noise = torch.randn_like(x0)
        xt = sa[ti][:, None] * x0 + soma[ti][:, None] * noise
        return ((m(xt, ti) - noise) ** 2).mean()

    return m, step


def train_learned(means_t, d, K, abar, T, variance, n_steps,
                  lr=1e-3, batch=512, seed=0, device=None,
                  lr_min=None, grad_clip=None, ema_decay=None, ema_warmup=0, cuda_graph=True,
                  branch_streams=True):
    m, step = learned_closure(means_t, d, K, abar, T, variance, batch, seed, device)
    return run_optimizer(m, step, n_steps, lr, lr_min, grad_clip, ema_decay, ema_warmup,
                         cuda_graph=cuda_graph, branch_streams=branch_streams)


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


# --- exact-score DDIM building blocks (shared by the Euler and Heun integrators) ---
@torch.no_grad()
def _exact_eps(x, means_t, ab, variance):
    """Closed-form eps-prediction of the exact GMM score at signal level `ab`."""
    v = ab * variance + (1 - ab)
    sc = true_score(x, torch.sqrt(ab) * means_t, v)
    return -torch.sqrt(1 - ab) * sc


@torch.no_grad()
def _ddim_step(x, eps, ab, abp):
    """One deterministic DDIM update from signal level `ab` to `abp`, given eps."""
    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps


@torch.no_grad()
def _ddim_transport(Xd, means_t, abar_t, order, levels, variance, chunk=50000):
    """Integrate the exact-score DDIM map along the sequence of (ab, abp) `levels`.

    order='euler': one exact-eps evaluation per step (first order).
    order='heun' : predictor-corrector -- eps1 at x, predict x~ = DDIM(x, eps1), eps2 at x~
                   (at the TARGET level), then x' = DDIM(x, (eps1+eps2)/2). Second order,
                   and (data -> seed -> data) round-trips to L exactly at the same T.
    """
    heun = str(order).lower() == "heun"
    X = Xd.clone()
    for ab, abp in levels:
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            eps1 = _exact_eps(xs, means_t, ab, variance)
            if heun:
                xtil = _ddim_step(xs, eps1, ab, abp)
                eps2 = _exact_eps(xtil, means_t, abp, variance)
                eps1 = 0.5 * (eps1 + eps2)
            X[s:s + chunk] = _ddim_step(xs, eps1, ab, abp)
    return X


@torch.no_grad()
def backtrack_true(Xd, means_t, abar_t, T, variance, chunk=50000, order="heun"):
    """Data -> seed (noise) under the exact GMM score. `order` in {euler, heun}."""
    levels = [(abar_t[i - 1], abar_t[i]) for i in range(1, T)]
    return _ddim_transport(Xd, means_t, abar_t, order, levels, variance, chunk)


@torch.no_grad()
def forward_true(X0, means_t, abar_t, T, variance, chunk=50000, order="heun"):
    """Seeds (noise) -> data under the exact GMM score, the inverse of backtrack_true. Used to
    label seeds by the exact score (the analytic reference), e.g. altered_knn's calibration set."""
    levels = [(abar_t[i], abar_t[i - 1]) for i in reversed(range(1, T))]
    return _ddim_transport(X0, means_t, abar_t, order, levels, variance, chunk)


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


