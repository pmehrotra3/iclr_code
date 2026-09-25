"""processes/heun.py — the DDIM network integrated with Heun's method (see pf_ode.py)."""
from __future__ import annotations

import core
from processes.pf_ode import PFODEProcess


class HeunProcess(PFODEProcess):
    """Heun: an Euler (= DDIM) step to predict the end point, then the trapezoid of the slopes
    at both ends. Both slopes sit on grid points, so the network is only asked about times it
    was trained on. Written with core._ddim_step, it is the learned-network twin of the exact
    field's `true_order=heun` transport."""
    name = "heun"

    def _solve(self, model, X):
        for i in reversed(range(1, self.T)):
            ab, abp = self.abar[i], self.abar[i - 1]
            eps = self._eps(model, X, i)
            eps = 0.5 * (eps + self._eps(model, core._ddim_step(X, eps, ab, abp), i - 1))
            X = core._ddim_step(X, eps, ab, abp)
        return X
