"""processes/dpmpp2m.py — the DDIM network integrated with DPM-Solver++(2M) (see pf_ode.py)."""
from __future__ import annotations

import math

import torch

from processes.pf_ode import PFODEProcess


class DPMpp2MProcess(PFODEProcess):
    """DPM-Solver++(2M): with alpha = sqrt(abar), s = sqrt(1 - abar), lambda = log(alpha / s) and
    the network's data prediction x0 = (x - s eps) / alpha, one step s -> t is
        x_t = (s_t / s_s) x_s - alpha_t (exp(-h) - 1) D,     h = lambda_t - lambda_s
    with D = x0_s on the first step (exactly a DDIM step) and afterwards the 2nd-order blend
        D = (1 + 1/(2r)) x0_s - (1/(2r)) x0_prev,             r = h_prev / h."""
    name = "dpmpp2m"

    def _solve(self, model, X):
        ab = self.abar.double()
        alpha, s = torch.sqrt(ab), torch.sqrt(1 - ab)
        lam = torch.log(alpha / s)
        x0_prev, h_prev = None, None
        for i in reversed(range(1, self.T)):
            eps = self._eps(model, X, i)
            x0 = (X - float(s[i]) * eps) / float(alpha[i])
            h = float(lam[i - 1] - lam[i])
            if x0_prev is None:
                D = x0
            else:
                r = h_prev / h
                D = (1 + 0.5 / r) * x0 - (0.5 / r) * x0_prev
            X = float(s[i - 1] / s[i]) * X - float(alpha[i - 1]) * (math.exp(-h) - 1) * D
            x0_prev, h_prev = x0, h
        return X
