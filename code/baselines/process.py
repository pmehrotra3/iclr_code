"""baselines/process.py — RODS and IQ as first-class processes.

Design note
-----------
RODS (Tian et al., NeurIPS 2025) and IQ (Sobieski et al., arXiv 2605.05026) are
sampling-time interventions on a *trained diffusion model*: same network, same
training objective, different reverse recursion. In this repo's vocabulary that
makes them processes that share everything with `ddim` except `sample()`.

So each subclasses DDIMProcess and overrides `sample()` only. Everything else --
training, the analytic field, `true_backward`, the ring atlas, the seed -> fate
classifiers, the seedmap panels, the tables -- is inherited untouched. The
question the repo asks ("from the seed alone, can you predict the fate?") is then
asked of each intervention for free, which is the comparison Abhinav wants: a
method that respects the class boundary should leave the seed -> fate map about
as predictable as the baseline's, while one that scrambles classes should make it
harder to predict.

`true_forward` is deliberately NOT overridden. The analytic field is the
reference sampler for the GMM and does not depend on which intervention is being
studied; leaving it shared keeps the `eval.labels=true` control meaningful.

Reusing the ddim checkpoints
---------------------------
train_model is inherited and seeded identically, so training here reproduces the
ddim weights exactly -- but it would pay for them again. Symlink instead:

    cd checkpoints && ln -s ddim rods_cas && ln -s ddim rods_sas && ln -s ddim iq

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

from common.process import register
from ddim.process import DDIMProcess


def _unit(g, eps=1e-12):
    n = g.flatten(1).norm(dim=1).view(-1, *([1] * (g.ndim - 1)))
    return g / (n + eps)


class _Corrected(DDIMProcess):
    """Shared plumbing: calibrate a per-step threshold, then sample with it."""

    variant: str = "cas"

    def _knobs(self):
        c = getattr(self.cfg, self.name, None)
        return (float(getattr(c, "rho_frac", 0.15)),
                float(getattr(c, "q_pct", 99.0)),
                int(getattr(c, "n_mc", 8)),
                float(getattr(c, "lam", 0.1)),
                int(getattr(c, "n_calibrate", 2000)),
                int(getattr(c, "chunk", 20000)))

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