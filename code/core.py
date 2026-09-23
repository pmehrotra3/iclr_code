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
    H, NB, TD = 256, 4, 128          # defaults: hidden width, residual blocks, time-embed dim

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


def net_size(d, cfg=None):
    """'small' for d <= train.small_max_d, else 'large'. Two model sizes for the whole sweep:
    small covers d <= 64, large covers d = 128 .. 512. (A single fixed width cannot resolve
    the sigma-ball in high d: the R99 margin shrinks like 1/sqrt(d) while the model's
    residual error does not.)"""
    t = getattr(cfg, "train", None) if cfg is not None else None
    small_max = int(getattr(t, "small_max_d", 64)) if t is not None else 64
    return "small" if d <= small_max else "large"


def net_arch(d, cfg=None):
    """ScoreNet constructor kwargs for ambient dimension d: the hidden width of the size class
    net_size(d) from train.width (a {small, large} mapping); without a config every d uses the
    ScoreNet default. Depth and time-embedding dim stay at the ScoreNet defaults."""
    t = getattr(cfg, "train", None) if cfg is not None else None
    widths = getattr(t, "width", None) if t is not None else None
    h = int(widths[net_size(d, cfg)]) if widths else ScoreNet.H
    return {"h": h, "nb": ScoreNet.NB, "td": ScoreNet.TD}


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
WEIGHT_LO, WEIGHT_HI = 0.3, 0.8   # raw mixing weights are drawn from this closed interval


def mode_weights(K, seed, weighted, device=None, base_seed=0):
    """Mixing weights (K,) of the GMM. Uniform 1/K when not weighted. Otherwise ONE imbalance
    profile per (K, base_seed) -- w_k ~ U[WEIGHT_LO, WEIGHT_HI] i.i.d., normalised to sum 1
    (so no mode is more than WEIGHT_HI/WEIGHT_LO x another) -- shared by every repeat, and
    each repeat `seed` assigns that profile to the modes by its own random permutation
    (sampling the weights without replacement). So the mean +- std over seeds averages over
    placements of the SAME weights, not over different weight draws."""
    if not weighted:
        return torch.full((K,), 1.0 / K, device=device)
    # drawn on CPU so the profile and its placement do not depend on the device
    g = torch.Generator().manual_seed(int(base_seed) + 13)
    w = WEIGHT_LO + (WEIGHT_HI - WEIGHT_LO) * torch.rand(K, generator=g)
    w = w / w.sum()
    g = torch.Generator().manual_seed(int(seed) + 17)
    return w[torch.randperm(K, generator=g)].to(device)


def _draw_modes(K, n, weights, g, device):
    """n mode indices: uniform when weights is None, else multinomial under weights."""
    if weights is None:
        return torch.randint(0, K, (n,), generator=g, device=device)
    return torch.multinomial(weights.to(device), n, replacement=True, generator=g)


def gmm_train_set(means_t, n, variance, seed, device=None, weights=None):
    """A fixed training set: n i.i.d. draws from the GMM (mode ~ weights, uniform when None;
    isotropic variance), seeded so it is reproducible per (d, K, seed). Returns (X (n, d),
    mode index (n,))."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(int(seed) + 11)
    k = _draw_modes(K, n, weights, g, device)
    X = means_t[k] + math.sqrt(variance) * torch.randn(n, d, generator=g, device=device)
    return X, k


def _minibatch(means_t, variance, batch, n_train, seed, device, weights=None):
    """Return a sampler of clean minibatches x0 (batch, d): from a fixed training set of
    n_train GMM draws (minibatches re-drawn with replacement, so each step is a random subset),
    or fresh i.i.d. draws every step when n_train is None (the infinite-data regime).
    Modes are drawn under `weights` (uniform when None)."""
    K, d = means_t.shape
    sigma = math.sqrt(variance)
    if n_train:
        X, _ = gmm_train_set(means_t, int(n_train), variance, seed, device, weights)

        def draw():
            idx = torch.randint(0, X.shape[0], (batch,), device=device)
            return X.index_select(0, idx)
    else:
        w = None if weights is None else weights.to(device)

        def draw():
            k = _draw_modes(K, batch, w, None, device)
            return means_t[k] + sigma * torch.randn(batch, d, device=device)
    return draw


def learned_closure(means_t, d, K, abar, T, variance, batch=512, seed=0, device=None,
                    n_train=None, weights=None, arch=None):
    """A fresh ScoreNet (init seeded by `seed`; size from `arch`, see net_arch) and its
    eps-prediction loss closure: one minibatch of noised GMM samples at random steps, MSE
    against the noise. The clean samples come from a fixed n_train-sample training set (or
    fresh draws when n_train is None)."""
    torch.manual_seed(seed)
    sa = torch.sqrt(abar); soma = torch.sqrt(1 - abar)
    m = ScoreNet(d, **(arch or {})).to(device)
    draw = _minibatch(means_t, variance, batch, n_train, seed, device, weights)

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
# true_score      : s(x) = sum_k r_k (mu_k - x)/v, with r = softmax(log pi_k - ||x-mu_k||^2 / 2v)
#                   the posterior responsibilities under mixing weights pi (uniform when
#                   logw is None). Callers pass the noised mixture at level t: means
#                   sqrt(ab)*mu, variance v = ab*variance + (1-ab).
# backtrack_true  : runs the DDIM update forwards in t (noise increasing), driving it with
#                   the true score instead of a network, so data-space points are carried
#                   back to the seeds that would have produced them.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def true_score(X, Mt, v, logw=None):
    logits = -torch.cdist(X, Mt) ** 2 / (2 * v)
    if logw is not None:
        logits = logits + logw
    w = torch.softmax(logits, 1)
    return (w @ Mt - X) / v


def log_weights(weights, device=None):
    """log mixing weights (1, K) for true_score / the flow velocity, or None when uniform."""
    if weights is None:
        return None
    w = weights.to(device) if device is not None else weights
    return torch.log(w.clamp_min(1e-30))[None, :]


# --- exact-score DDIM building blocks (shared by the Euler and Heun integrators) ---
@torch.no_grad()
def _exact_eps(x, means_t, ab, variance, logw=None):
    """Closed-form eps-prediction of the exact GMM score at signal level `ab`."""
    v = ab * variance + (1 - ab)
    sc = true_score(x, torch.sqrt(ab) * means_t, v, logw)
    return -torch.sqrt(1 - ab) * sc


@torch.no_grad()
def _ddim_step(x, eps, ab, abp):
    """One deterministic DDIM update from signal level `ab` to `abp`, given eps."""
    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps


@torch.no_grad()
def _ddim_transport(Xd, means_t, abar_t, order, levels, variance, chunk=50000, logw=None,
                    inplace=False):
    """Integrate the exact-score DDIM map along the sequence of (ab, abp) `levels`.

    order='euler': one exact-eps evaluation per step (first order).
    order='heun' : predictor-corrector -- eps1 at x, predict x~ = DDIM(x, eps1), eps2 at x~
                   (at the TARGET level), then x' = DDIM(x, (eps1+eps2)/2). Second order,
                   and (data -> seed -> data) round-trips to L exactly at the same T.
    inplace=True overwrites Xd instead of cloning it (the caller no longer needs the input):
    halves the standing memory of a multi-GB anchor backtrack.
    """
    heun = str(order).lower() == "heun"
    X = Xd if inplace else Xd.clone()
    for ab, abp in levels:
        for s in range(0, X.shape[0], chunk):
            xs = X[s:s + chunk]
            eps1 = _exact_eps(xs, means_t, ab, variance, logw)
            if heun:
                xtil = _ddim_step(xs, eps1, ab, abp)
                eps2 = _exact_eps(xtil, means_t, abp, variance, logw)
                eps1 = 0.5 * (eps1 + eps2)
            X[s:s + chunk] = _ddim_step(xs, eps1, ab, abp)
    return X


@torch.no_grad()
def backtrack_true(Xd, means_t, abar_t, T, variance, chunk=50000, order="heun", weights=None,
                   inplace=False):
    """Data -> seed (noise) under the exact GMM score. `order` in {euler, heun}. inplace=True
    overwrites Xd (see _ddim_transport)."""
    levels = [(abar_t[i - 1], abar_t[i]) for i in range(1, T)]
    return _ddim_transport(Xd, means_t, abar_t, order, levels, variance, chunk,
                           log_weights(weights, Xd.device), inplace=inplace)


@torch.no_grad()
def forward_true(X0, means_t, abar_t, T, variance, chunk=50000, order="heun", weights=None):
    """Seeds (noise) -> data under the exact GMM score, the inverse of backtrack_true. Used to
    label seeds by the exact score (the analytic reference), e.g. altered_knn's calibration set."""
    levels = [(abar_t[i], abar_t[i - 1]) for i in reversed(range(1, T))]
    return _ddim_transport(X0, means_t, abar_t, order, levels, variance, chunk,
                           log_weights(weights, X0.device))


# --------------------------------------------------------------------------------------
# labelled anchors planted in DATA space (no model, no backtrack).
#   ball_anchors        : mode balls + a thin band just outside -> labels {k, -1}, for the
#                         knn / polar predictors.
#   altered_knn_anchors : mode-only concentric spheres with radius-decaying weights (no
#                         hallucination class), for the altered_knn predictor.
#   polar_weighted_anchors : per mode, n_spheres concentric spheres at uniformly spaced radii
#                         (inner ones labelled k, those past R99 -> -1), directions by farthest-
#                         point sampling, each anchor weighted by the Gaussian's log-density
#                         decay rate at its radius. Replaces ball_anchors when
#                         anchors.strategy = polar_weighted.
# Callers backtrack these to seed space with the exact field before fitting.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def ball_anchors(means_t, R99, n_per_mode, shell_frac=0.5, shell_sigma=2.0, sigma=None,
                 seed=0, device=None):
    """Per mode: n_per_mode points uniform in the R99 ball, plus shell_frac * n_per_mode
    points uniform in radius over R99 .. R99 + shell_sigma * sigma. Every anchor is coloured
    with the ground-truth rule L (nearest mode within R99, else -1).

    The (n, d) tensor is allocated once and filled per mode: at d=512 with 300k anchors per
    mode it is ~15 GB, so building a list of pieces and torch.cat-ing them would need twice
    that for a moment."""
    K, d = means_t.shape
    g = torch.Generator(device=device).manual_seed(seed)
    n_shell = max(1, int(round(shell_frac * n_per_mode)))
    width = shell_sigma * sigma
    per_mode = n_shell + n_per_mode
    P = torch.empty(K * per_mode, d, device=device)
    for k in range(K):
        s0 = k * per_mode
        u = torch.rand(n_per_mode, generator=g, device=device) ** (1.0 / d)
        dirs = torch.randn(n_per_mode, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        torch.addcmul(means_t[k], (R99 * u)[:, None], dirs, out=P[s0:s0 + n_per_mode])
        r = R99 + width * torch.rand(n_shell, generator=g, device=device)
        dirs = torch.randn(n_shell, d, generator=g, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        torch.addcmul(means_t[k], r[:, None], dirs, out=P[s0 + n_per_mode:s0 + per_mode])
    return P, label_fate(P, means_t, R99)
@torch.no_grad()
def fps_directions(n, d, pool_factor=4, generator=None, device=None, min_pool=4096):
    """n unit vectors in R^d spread as far apart as possible: greedy farthest-point sampling
    over max(pool_factor * n, min_pool) random directions (each pick maximises its smallest
    angle to the ones already chosen). On the circle this gives evenly spaced angles."""
    m = max(n, int(pool_factor * n), int(min_pool))
    C = torch.randn(m, d, generator=generator, device=device)
    C /= C.norm(dim=1, keepdim=True)
    chosen = torch.empty(n, dtype=torch.long, device=device)
    chosen[0] = 0
    maxcos = C @ C[0]                               # cosine to the nearest chosen direction
    for i in range(1, n):
        j = torch.argmin(maxcos)
        chosen[i] = j
        maxcos = torch.maximum(maxcos, C @ C[j])
    return C[chosen]


@torch.no_grad()
def random_rotation(d, generator=None, device=None):
    """Haar-random orthogonal d x d matrix (QR of a Gaussian matrix, signs fixed)."""
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=generator, device=device))
    return Q * torch.sign(torch.diagonal(R))[None, :]


def polar_weight(r, R99, sigma, kind="log_decay", w_min=0.1):
    """Anchor weight at distance r from its mode centre (large near the boundary, small at
    the centre):
      log_decay   : the Gaussian's log-density decay rate -d log p / dr = r / sigma^2,
                    normalised to 1 at R99  ->  w = r / R99
      inv_density : inverse Gaussian density, normalised to 1 at R99
                    ->  w = exp((r^2 - R99^2) / (2 sigma^2))
      uniform     : w = 1
    floored at w_min."""
    if kind == "uniform":
        return 1.0
    if kind == "inv_density":
        w = math.exp((r ** 2 - R99 ** 2) / (2 * sigma ** 2))
    elif kind == "log_decay":
        w = r / R99
    else:
        raise ValueError(f"unknown polar_weighted weight {kind!r}: log_decay | inv_density | uniform")
    return max(float(w_min), float(w))


@torch.no_grad()
def polar_weighted_anchors(means_t, R99, n_per_mode, n_spheres=10, n_per_sphere=None, r_max=1.2,
                           fps_pool=4, weight="log_decay", w_min=0.1, sigma=None, seed=0,
                           device=None):
    """Per mode k: n_spheres concentric spheres at radii r_j = r_max * R99 * j / n_spheres
    (j = 1..n_spheres, uniform in radius), n_per_sphere points on each (default n_per_mode //
    n_spheres). The directions are one farthest-point set, turned by a fresh random rotation
    per (mode, sphere) so spheres do not share directions. Labels by the usual rule L (spheres
    past R99 -> -1); weight polar_weight(r_j). Returns P (n, d), y (n,), w (n,)."""
    K, d = means_t.shape
    per_sphere = int(n_per_sphere) if n_per_sphere else max(1, int(n_per_mode) // int(n_spheres))
    g = torch.Generator(device=device).manual_seed(seed)
    D = fps_directions(per_sphere, d, fps_pool, g, device)
    radii = [float(r_max) * R99 * j / n_spheres for j in range(1, n_spheres + 1)]
    per_mode = n_spheres * per_sphere
    P = torch.empty(K * per_mode, d, device=device)
    w = torch.empty(K * per_mode, device=device)
    for k in range(K):
        for j, r in enumerate(radii):
            s0 = k * per_mode + j * per_sphere
            blk = P[s0:s0 + per_sphere]
            torch.matmul(D, random_rotation(d, g, device), out=blk)
            blk.mul_(r).add_(means_t[k])
            w[s0:s0 + per_sphere] = polar_weight(r, R99, sigma, weight, w_min)
    return P, label_fate(P, means_t, R99), w


@torch.no_grad()
def altered_knn_anchors(means_t, R99, n_per_mode, n_rings=5, r_max=1.5, weight="linear",
                        w_min=0.2, sigma=None, seed=0, device=None):
    """Per mode: n_rings concentric spheres of radius r_j = R99 * r_max * j / n_rings, each
    labelled with its mode, weighted by a radius-decaying confidence (linear or gaussian).
    Returns P (n, d), y (n,), w (n,); no hallucination class. P is allocated once and filled
    ring by ring (see ball_anchors); y and w are built as tensors, not Python lists."""
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
            s0 = k * per_mode + j * per_ring
            dirs = torch.randn(per_ring, d, generator=g, device=device)
            dirs /= dirs.norm(dim=1, keepdim=True)
            blk = P[s0:s0 + per_ring]
            torch.mul(dirs, r, out=blk); blk += means_t[k]        # == means_t[k] + r * dirs, bit for bit
    return P, y, w
