"""ddim/process.py — variance-preserving diffusion with a deterministic DDIM sampler.

Time convention (internal): step index i = T-1 is the source (noise), i = 0 is data.
The network predicts eps. The analytic reference field is the closed-form GMM score, and
`true_forward` is the same DDIM recursion driven by that score (the ideal sampler).

Noise schedule: continuous-time VP (Song et al. 2021) sampled at T points, so the
terminal abar_T ~ 4e-5 is independent of T and the source end is genuinely N(0, I).
(The discrete DDPM linspace(1e-4, 0.02, T) schedule only reaches that at T = 1000; at
T = 100 it leaves abar_T ~ 0.36, which silently breaks the seed -> fate map.)
"""
from __future__ import annotations
import numpy as np
import torch

from common import gmm, nets
from common.process import Process, register


def make_schedule(T: int, beta_min: float = 0.1, beta_max: float = 20.0, device=None):
    """abar(t) = exp(-(beta_min t + (beta_max - beta_min) t^2 / 2)), t in (0, 1]."""
    t = np.linspace(1.0 / T, 1.0, T)
    log_abar = -(beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)
    return torch.tensor(np.exp(log_abar), dtype=torch.float32, device=device)


@register("ddim")
class DDIMProcess(Process):
    def __init__(self, means_t, variance, T, device, cfg=None):
        super().__init__(means_t, variance, T, device, cfg)
        sch = getattr(cfg, "ddim", None)
        self.beta_min = float(getattr(sch, "beta_min", 0.1))
        self.beta_max = float(getattr(sch, "beta_max", 20.0))
        self.true_solver = str(getattr(sch, "true_solver", "euler"))   # euler | heun, exact-score passes only
        self.abar = make_schedule(T, self.beta_min, self.beta_max, device=device)

    def extra_ckpt(self):
        return {"ddim": {"beta_min": self.beta_min, "beta_max": self.beta_max}}

    def build_model(self, d):
        return nets.ScoreNet(d, **self.net_kwargs()).to(self.device)

    # ---- learned model: eps-prediction (DDPM objective) ----
    def train_model(self, K, d, n_steps, lr, batch, seed):
        torch.manual_seed(seed)
        sa, soma = torch.sqrt(self.abar), torch.sqrt(1 - self.abar)
        sigma = self.variance ** 0.5
        m = self.build_model(d)

        def loss_fn(batch):
            x0 = gmm.sample_data(self.means_t, sigma, batch, self.device)
            t = torch.randint(0, self.T, (batch,), device=self.device)
            eps = torch.randn_like(x0)
            xt = sa[t][:, None] * x0 + soma[t][:, None] * eps
            return ((m(xt, t) - eps) ** 2).mean()
        return self.fit(m, loss_fn, n_steps, lr, batch)

    # ---- one DDIM step shared by the learned and the analytic sampler ----
    def _ddim_step(self, xs, i, eps):
        ab, abp = self.abar[i], self.abar[i - 1]
        x0 = (xs - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        model.eval()
        X = X0.clone()
        for i in reversed(range(1, self.T)):
            for s in range(0, X.shape[0], chunk):
                xs = X[s:s + chunk]
                ti = torch.full((xs.shape[0],), i, dtype=torch.long, device=self.device)
                X[s:s + chunk] = self._ddim_step(xs, i, model(xs, ti))
        return X

    def _true_eps(self, xs, i):
        ab = self.abar[i]
        v = ab * self.variance + (1 - ab)
        return -torch.sqrt(1 - ab) * gmm.true_score(xs, torch.sqrt(ab) * self.means_t, v)

    def _true_step(self, xs, i, j):
        """One exact-score DDIM step from schedule index i to index j (either direction).

        euler: the plain DDIM update with eps evaluated at the start of the step.
        heun : predictor-corrector -- take the Euler step, evaluate eps at its end, redo the
               step with the average eps. Second order, so the forward and backward passes are
               inverse to O(1/T^2) instead of O(1/T); the difference shows up only on anchors
               whose seed image hugs a basin boundary.
        """
        ab, abj = self.abar[i], self.abar[j]

        def step(eps):
            x0 = (xs - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            return torch.sqrt(abj) * x0 + torch.sqrt(1 - abj) * eps
        eps = self._true_eps(xs, i)
        if self.true_solver == "euler":
            return step(eps)
        if self.true_solver == "heun":
            return step(0.5 * (eps + self._true_eps(step(eps), j)))
        raise ValueError(f"unknown ddim.true_solver {self.true_solver!r}")

    @torch.no_grad()
    def true_backward(self, Xd, chunk=50000):
        """Reverse DDIM with the exact score: data (index 0) -> noise (index T-1)."""
        X = Xd.clone()
        for i in range(1, self.T):
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._true_step(X[s:s + chunk], i - 1, i)
        return X

    @torch.no_grad()
    def true_forward(self, X0, chunk=50000):
        """DDIM driven by the exact score of the noised mixture N(sqrt(ab) mu_k, (ab s^2 + 1 - ab) I)."""
        X = X0.clone()
        for i in reversed(range(1, self.T)):
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._true_step(X[s:s + chunk], i, i - 1)
        return X
