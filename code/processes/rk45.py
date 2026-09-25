"""processes/rk45.py — the DDIM network integrated with fixed-step Dormand-Prince RK45 (see pf_ode.py)."""
from __future__ import annotations

import torch

from processes.pf_ode import PFODEProcess

# Dormand-Prince 5(4): nodes, stage matrix and the 5th-order weights (the 7th, FSAL stage only
# feeds the embedded error estimate, which a fixed-step solver does not use)
_DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_DP_A = ((),
         (1 / 5,),
         (3 / 40, 9 / 40),
         (44 / 45, -56 / 15, 32 / 9),
         (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
         (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656))
_DP_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)


class RK45Process(PFODEProcess):
    """Fixed-step Dormand-Prince (the RK45 of scipy / MATLAB without step-size control): six
    slopes per step, at sigma between the two grid points, combined with 5th-order weights."""
    name = "rk45"

    def _solve(self, model, X):
        y = X / torch.sqrt(self.abar[self.T - 1])
        for i in reversed(range(1, self.T)):
            s0, s1 = float(self.sig[i]), float(self.sig[i - 1])
            h = s1 - s0                                          # < 0: sigma decreases to data
            k = []
            for c, a in zip(_DP_C, _DP_A):
                yi = y + h * sum(aj * kj for aj, kj in zip(a, k)) if a else y
                idx = i if c == 0 else (i - 1 if c == 1 else None)
                k.append(self._f(model, yi, s0 + c * h, idx))
            y = y + h * sum(b * kj for b, kj in zip(_DP_B, k) if b)
        return y * torch.sqrt(self.abar[0])
