"""heun / rk45 / dpmpp2m integrate the DDIM network's ODE with other solvers. Fed the EXACT eps
of a Gaussian mixture in place of a network, every solver must follow the exact field: Heun
reproduces core.forward_true(order=heun) step for step, and all of them send seeds to the same
modes. Plus the wiring: these processes load the DDIM checkpoints and are never trained."""
import tempfile
import unittest

import torch

from unit_tests import _CODE  # noqa: F401
from unit_tests._cfg import tiny_cfg


class ExactEps(torch.nn.Module):
    """The exact eps of the GMM at a (possibly fractional) grid index, shaped like ScoreNet."""

    def __init__(self, means, abar, variance):
        super().__init__()
        self.means, self.abar, self.variance = means, abar.double(), variance

    def forward(self, x, t):
        import core
        i = float(t[0])
        lo = int(i)
        hi = min(lo + 1, self.abar.numel() - 1)
        ab = self.abar[lo] + (i - lo) * (self.abar[hi] - self.abar[lo])
        return core._exact_eps(x, self.means, ab.float(), self.variance)


class TestSamplers(unittest.TestCase):
    def setUp(self):
        import core
        from processes.factory import make_process
        torch.manual_seed(0)
        self.cfg = tiny_cfg(tempfile.mkdtemp(prefix="iclr_samplers_"))
        self.d, self.K, self.T, self.var = 2, 3, 200, 0.05 ** 2
        self.means, _ = core.sample_modes(self.K, self.d, 2.0, self.var ** 0.5, seed=0)
        self.R99 = core.r99(self.d, self.var ** 0.5)
        self.ddim = make_process("ddim", self.means, self.var, self.T, torch.device("cpu"), self.cfg)
        self.model = ExactEps(self.means, self.ddim.abar, self.var)
        self.X = torch.randn(3000, self.d)

    def test_heun_is_the_exact_fields_heun_transport(self):
        import core
        from processes.factory import make_process
        heun = make_process("heun", self.means, self.var, self.T, torch.device("cpu"), self.cfg)
        ref = core.forward_true(self.X, self.means, self.ddim.abar, self.T, self.var, order="heun")
        self.assertLess(float((heun.sample(self.model, self.X) - ref).abs().max()), 1e-4)

    def test_every_solver_follows_the_exact_field(self):
        import core
        from processes.factory import make_process
        ref = core.label_fate(core.forward_true(self.X, self.means, self.ddim.abar, self.T, self.var),
                              self.means, self.R99)
        for name in ("ddim", "heun", "rk45", "dpmpp2m"):
            p = make_process(name, self.means, self.var, self.T, torch.device("cpu"), self.cfg)
            lab = core.label_fate(p.sample(self.model, self.X), self.means, self.R99)
            self.assertGreater(float((lab == ref).float().mean()), 0.99, name)

    def test_they_use_ddim_checkpoints_and_never_train(self):
        import train
        from processes.factory import checkpoint_process
        for name in ("heun", "rk45", "dpmpp2m"):
            self.assertEqual(checkpoint_process(name), "ddim")
            cfg = tiny_cfg(tempfile.mkdtemp(prefix="iclr_samplers_"), process=name)
            self.assertEqual(cfg.process.name, name)
            self.assertEqual(cfg.process.true_order, "heun")         # every DDIM setting inherited
            train.run(cfg)                                           # prints and returns
        self.assertEqual(checkpoint_process("ddim"), "ddim")
        self.assertEqual(checkpoint_process("flow"), "flow")


if __name__ == "__main__":
    unittest.main()
