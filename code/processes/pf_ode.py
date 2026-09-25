"""processes/pf_ode.py — the shared base of the solvers that sample the trained DDIM network.

DDIM's deterministic sampler is the Euler method for the probability-flow ODE of the VP
diffusion, written in y = x / sqrt(abar) and sigma = sqrt((1 - abar) / abar):

    dy / dsigma = eps_theta(x, t),    x = y / sqrt(1 + sigma^2)    (Song et al. 2021, Karras et al. 2022)

so the same eps-network can be integrated by any ODE solver: only the discretisation changes,
never the model. These processes reuse the DDIM checkpoints (checkpoints/ddim/...) untouched and
step over the same T_train grid:

    heun.py     2nd order: Euler predictor + trapezoidal corrector            2 network calls / step
    rk45.py     fixed-step Dormand-Prince, 5th-order solution                 6 network calls / step
    dpmpp2m.py  DPM-Solver++(2M) (Lu et al. 2022): 2nd-order multistep in      1 network call  / step
                log-SNR on the x0-prediction (its 1st-order step is DDIM)

This file holds what they share: PFODEProcess, DDIM with `sample` handed to a solver's `_solve`.

The exact field, the anchors and the calibration seeds are DDIM's (the same ODE with the exact
score), so only the ground truth -- where the learned sampler sends each seed -- differs; it is
cached under checkpoints/<process>/ and results go to output/<run_id>/<process>/.
Stochastic samplers (DDPM, DPM-Solver++ SDE) are left out: a seed has no single fate under them.
"""
from __future__ import annotations

import torch

from processes.ddim import DDIMProcess


class PFODEProcess(DDIMProcess):
    """DDIM's schedule, exact field and checkpoints; `sample` integrates the learned eps with
    `_solve` instead of DDIM steps. Subclasses set `name` and write `_solve`."""
    ckpt_process = "ddim"                       # train.py / evaluate.py load and never train these

    def __init__(self, means_t, variance, T, device, cfg, weights=None):
        super().__init__(means_t, variance, T, device, cfg, weights)
        ab = self.abar.double()
        self.sig = torch.sqrt((1 - ab) / ab)    # sigma at each grid index, increasing with the index

    def train_closure(self, K, d, batch, seed):
        raise RuntimeError(f"process '{self.name}' samples the DDIM network: train process=ddim")

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        model.eval()
        X = X0.clone()
        for s in range(0, X.shape[0], chunk):   # chunks outside the steps: dpmpp2m keeps history
            X[s:s + chunk] = self._solve(model, X[s:s + chunk])
        return X

    # ---- the ODE in (y, sigma) ----
    def _eps(self, model, x, idx):
        """The network's eps at grid index `idx` (fractional between grid points: the time
        embedding is continuous in its input)."""
        return model(x, torch.full((x.shape[0],), float(idx), device=x.device))

    def _index_of(self, sigma):
        """Fractional grid index of a sigma inside the grid (linear between neighbours)."""
        j = int(torch.searchsorted(self.sig, torch.tensor(float(sigma), dtype=self.sig.dtype,
                                                           device=self.sig.device)).clamp(1, self.T - 1))
        lo, hi = float(self.sig[j - 1]), float(self.sig[j])
        return j - 1 + (float(sigma) - lo) / (hi - lo)

    def _f(self, model, y, sigma, idx=None):
        """dy/dsigma = eps(x, t) at x = y / sqrt(1 + sigma^2)."""
        x = y / (1.0 + float(sigma) ** 2) ** 0.5
        return self._eps(model, x, self._index_of(sigma) if idx is None else idx)

    def _solve(self, model, X):
        raise NotImplementedError
