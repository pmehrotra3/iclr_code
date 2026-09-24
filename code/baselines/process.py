"""baselines/process.py — RODS and IQ as first-class processes.

Design note
-----------
RODS (Tian et al., NeurIPS 2025) and IQ (Sobieski et al., arXiv 2605.05026) are
sampling-time interventions on a *trained diffusion model*: same network, same
training objective, different reverse recursion. In this repo's vocabulary that
makes them processes that share everything with `ddim` except `sample()`.

So each subclasses DDIMProcess and overrides `sample()` only. Everything else --
training, the analytic field (`true_field_forward` / `true_field_backtrack`), the
atlas, the seed -> fate predictors, the tables -- is inherited untouched. The
question the repo asks ("from the seed alone, can you predict the fate?") is then
asked of each intervention for free, which is the comparison Abhinav wants: a
method that respects the class boundary should leave the seed -> fate map about
as predictable as the baseline's, while one that scrambles classes should make it
harder to predict.

`true_field_forward` is deliberately NOT overridden. The analytic field is the
reference sampler for the GMM and does not depend on which intervention is being
studied; leaving it shared keeps the `eval.labels=true` control meaningful.

Reusing the ddim checkpoints
---------------------------
train_closure is inherited and seeded identically, so training here reproduces the
ddim weights exactly -- but it would pay for them again. code/baselines/main.py
links checkpoints/<name>/<variant>/checkpoints to ddim's instead
(link_ddim_checkpoints); where the repo has a checkpoint check
(artifacts.validate_checkpoint), it compares them against the ddim settings they were
trained with. Only the checkpoints
are shared: gt_cache/ stays separate, because each process's ground truth is its own
sampler's fates.

Knobs
-----
Each process reads its knobs from its own config, conf/process/<name>.yaml (so they
are part of the run's recorded settings, like any process setting). process_cfg()
builds that config for a process other than the selected one (stage `basin`).

Thresholds
----------
Both H (RODS) and LID (IQ) move by orders of magnitude across the trajectory --
score norms go like 1/sigma_t -- so a single scalar cutoff can only ever fire in
the last few steps, where a rho-sized kick knocks an already-converged sample off
its mode and manufactures hallucinations. Both use a per-timestep quantile
calibrated on a reference batch of seeds, which is how the LID paper's own
filtering works. `q_pct` is PER STEP: 99 fires on ~1% of steps, so ~T/100
corrections per trajectory. Lower values over-correct badly.

CAVEAT: the DSM -> LID reduction in `_lid` is inferred from algorithm 1, not
copied from the paper. Verify before quoting IQ numbers.
"""
from __future__ import annotations

import numpy as np
import torch

import os

from omegaconf import OmegaConf

from processes import factory
from processes.ddim import DDIMProcess
from train import sampler_dir, variant_of

NAMES = ("rods_cas", "rods_sas", "iq")
DEFAULTS = dict(rho_frac=0.15, q_pct=99.0, n_mc=8, lam=0.1, n_calibrate=2000, chunk=20000)
_PROCESS_CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "conf", "process")


def register(name):
    """Add a process class to processes.factory under `name` (make_process looks it up there)."""
    def deco(cls):
        cls.name = name
        factory._REGISTRY[name] = cls
        return cls
    return deco


def _process_yaml(name):
    """conf/process/<name>.yaml with its in-group defaults (e.g. ddim, base) merged in."""
    node = OmegaConf.load(os.path.join(_PROCESS_CONF, f"{name}.yaml"))
    parents = [x for x in node.pop("defaults", []) if x != "_self_"]
    return OmegaConf.merge(*[_process_yaml(str(x)) for x in parents], node) if parents else node


def checkpoint_config(ck, cfg):
    """cfg on the checkpoint's own noise schedule: the process settings stored in the checkpoint
    (beta_min, beta_max, ...) replace the current ones; name, T and the exact-field solver stay."""
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    stored = (ck.get("config") or {}).get("process", {})
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key not in ("name", "T_true", "T_train", "true_order", "true_solver"):
                out.process[key] = value
    return out


def process_cfg(cfg, ck, name):
    """cfg with process `name` selected, on the checkpoint's schedule. The selected process
    keeps its command-line overrides; any other one gets its conf/process/<name>.yaml."""
    if str(cfg.process.name) != name:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.process = _process_yaml(name)
    return checkpoint_config(ck, cfg)


def as_ddim(cfg):
    """cfg with a baseline's process block turned back into the ddim block it extends."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for k in DEFAULTS:
        cfg.process.pop(k, None)
    cfg.process.name = "ddim"
    return cfg


def _patch_checkpoint_check():
    """Where the repo has artifacts.validate_checkpoint, it refuses a checkpoint whose sampler or
    process settings differ from the run's, which a baseline reusing the ddim models always
    would. Run that check as if the process were ddim instead, so data / training / schedule
    mismatches are still caught. Versions without the check need nothing."""
    try:
        import artifacts
    except ImportError:
        return
    orig = getattr(artifacts, "validate_checkpoint", None)
    if orig is None or getattr(orig, "_baselines", False):
        return

    def validate_checkpoint(ck, cfg, d, K, seed, path):
        if str(cfg.process.name) in NAMES:
            cfg = as_ddim(cfg)
        return orig(ck, cfg, d, K, seed, path)

    validate_checkpoint._baselines = True
    artifacts.validate_checkpoint = validate_checkpoint


def link_ddim_checkpoints(cfg):
    """checkpoints/<name>/<variant>/checkpoints -> ../../ddim/<variant>/checkpoints for a
    baseline process, so it evaluates the ddim models instead of training copies."""
    name, variant = str(cfg.process.name), variant_of(cfg)
    if name not in NAMES:
        return
    src = os.path.join(sampler_dir(cfg.paths.checkpoints, "ddim", variant), "checkpoints")
    parent = sampler_dir(cfg.paths.checkpoints, name, variant)
    dst = os.path.join(parent, "checkpoints")
    if os.path.lexists(dst):
        return
    if not os.path.isdir(src):
        print(f"[{name}] no ddim checkpoints at {src} yet: run stages=[train] with process=ddim first")
        return
    os.makedirs(parent, exist_ok=True)
    os.symlink(os.path.relpath(src, parent), dst)
    print(f"[{name}] linked {dst} -> {src}")


def _unit(g, eps=1e-12):
    n = g.flatten(1).norm(dim=1).view(-1, *([1] * (g.ndim - 1)))
    return g / (n + eps)


class _Corrected(DDIMProcess):
    """Shared plumbing: calibrate a per-step threshold, then sample with it."""

    variant: str = "cas"

    def _ddim_step(self, x, i, eps):
        """One DDIM update from level i to i-1 (written out: core._ddim_step runs under no_grad)."""
        ab, abp = self.abar[i], self.abar[i - 1]
        x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps

    def _knobs(self):
        """This process's knobs from cfg.process (conf/process/<name>.yaml), else DEFAULTS."""
        c = self.cfg.process if str(self.cfg.process.get("name", "")) == self.name else {}
        g = lambda k: c.get(k, DEFAULTS[k]) if c else DEFAULTS[k]      # noqa: E731
        return (float(g("rho_frac")), float(g("q_pct")), int(g("n_mc")), float(g("lam")),
                int(g("n_calibrate")), int(g("chunk")))

    def _dmin(self):
        D = torch.cdist(self.means_t, self.means_t)
        D.fill_diagonal_(float("inf"))
        return float(D.min())

    # ---------------- RODS curvature index ---------------- #
    def _grad_score_norm(self, model, x, i):
        x = x.detach().requires_grad_(True)
        ti = torch.full((x.shape[0],), int(i), device=self.device, dtype=torch.long)
        with torch.enable_grad():
            v = -model(x, ti) / torch.sqrt(1 - self.abar[i])
            g = torch.autograd.grad(v.flatten(1).norm(dim=1).sum(), x)[0]
        return g.detach()

    def _index(self, model, x, i, rho):
        """H(x) of eq. (8), and the CAS perturbation (the same direction)."""
        g0 = self._grad_score_norm(model, x, i)
        delta = float(rho) * _unit(g0)
        g1 = self._grad_score_norm(model, x + delta, i)
        return (g1 - g0).flatten(1).norm(dim=1), delta

    # ---------------- IQ energy ---------------- #
    def _lid(self, model, x0_hat, j, n_mc, gen=None):
        """DSM-loss estimate of LID at x0_hat. INFERRED -- see module docstring."""
        sig = torch.sqrt(1 - self.abar[j])
        a_s = torch.sqrt(self.abar[j])
        ti = torch.full((x0_hat.shape[0],), int(j), device=self.device,
                        dtype=torch.long)
        acc = 0.0
        for _ in range(int(n_mc)):
            e = torch.randn(x0_hat.shape, device=self.device, generator=gen)
            xs = a_s * x0_hat + sig * e
            rec = (xs - sig * model(xs, ti)) / a_s
            acc = acc + (rec - x0_hat).flatten(1).pow(2).sum(1)
        return acc / float(n_mc) / (sig ** 2)

    # ---------------- one corrected trajectory ---------------- #
    def _run(self, model, X, rho, thr, n_mc, lam, seed, log=False):
        order = list(reversed(range(1, self.T)))
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        L = np.zeros((X.shape[0], len(order)), dtype=np.float32) if log else None
        fired = torch.zeros(X.shape[0], device=self.device)
        x = X.clone()

        for j, i in enumerate(order):
            ti = torch.full((x.shape[0],), i, device=self.device, dtype=torch.long)

            if self.variant in ("cas", "sas"):
                H, cas_delta = self._index(model, x, i, rho)
                if log:
                    L[:, j] = H.detach().cpu().numpy()
                with torch.no_grad():
                    eps = model(x, ti)
                    if thr is not None:
                        hit = H >= float(thr[j])
                        if bool(hit.any()):
                            dl = cas_delta if self.variant == "cas" \
                                else float(rho) * _unit(eps)
                            eps = torch.where(hit.view(-1, 1),
                                              model(x + dl, ti), eps)
                            fired += hit.float()
                    x = self._ddim_step(x, i, eps)

            else:                                   # iq
                sig = torch.sqrt(1 - self.abar[i])
                at = torch.sqrt(self.abar[i])
                xg = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    eps = model(xg, ti)
                    x0h = (xg - sig * eps) / at
                    lid = self._lid(model, x0h, max(i // 10, 1), n_mc, gen=gen)
                    g = torch.autograd.grad(lid.sum(), xg)[0]
                if log:
                    L[:, j] = lid.detach().cpu().numpy()
                with torch.no_grad():
                    raw = (sig ** 2) * g.detach()
                    nat = x0h.detach() - x.detach()
                    sc = (float(lam) * nat.flatten(1).norm(dim=1)
                          / (raw.flatten(1).norm(dim=1) + 1e-8)).view(-1, 1)
                    x0n = x0h.detach() - sc * raw
                    if thr is not None:
                        hit = lid.detach() >= float(thr[j])
                        x0n = torch.where(hit.view(-1, 1), x0n, x0h.detach())
                        fired += hit.float()
                    x = self._ddim_step(x, i, (x.detach() - at * x0n) / sig)

        return x, L, fired

    def sample(self, model, X0, chunk=20000):
        model.eval()
        rho_frac, q_pct, n_mc, lam, n_cal, ck = self._knobs()
        rho = rho_frac * self._dmin()
        chunk = min(chunk, ck)

        # per-timestep thresholds from a reference batch of seeds
        cal = X0[:min(n_cal, X0.shape[0])]
        _, L, _ = self._run(model, cal, rho, None, n_mc, lam, seed=0, log=True)
        thr = np.percentile(L, q_pct, axis=0)

        out = torch.empty_like(X0)
        total = 0.0
        for s in range(0, X0.shape[0], chunk):
            xb, _, fb = self._run(model, X0[s:s + chunk], rho, thr, n_mc, lam,
                                  seed=s)
            out[s:s + chunk] = xb
            total += float(fb.sum())
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[{self.name}] rho={rho:.4f}  q={q_pct}  thresholds "
              f"{thr.min():.3g}..{thr.max():.3g}  "
              f"{total / X0.shape[0]:.2f} corrections/trajectory")
        return out


@register("rods_cas")
class RODSCAS(_Corrected):
    variant = "cas"


@register("rods_sas")
class RODSSAS(_Corrected):
    variant = "sas"


@register("iq")
class IQ(_Corrected):
    variant = "iq"


_patch_checkpoint_check()
