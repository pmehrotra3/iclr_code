"""A run built in pieces equals the run done in one invocation (code/runstate.py): the same
checkpoints, the same results.json; changed settings are refused. Tiny cells on the CPU, plus
a CUDA-graph check that training seeds together or apart gives the same models."""
import json
import os
import tempfile
import unittest

import torch

from unit_tests import _CODE  # noqa: F401
from unit_tests._cfg import tiny_cfg


def _run(cfg):
    import train, evaluate, runstate
    runstate.check_and_record(cfg)
    train.run(cfg)
    evaluate.run(cfg)


def _results(cfg):
    import train
    p = os.path.join(train.sweep_dir(cfg.paths.output, cfg.run_id, "ddim", cfg.process.T_true,
                                     train.variant_of(cfg)), "results.json")
    blob = json.load(open(p))
    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("secs")} for r in rows]
    return strip(blob["results"]), strip(blob["aggregate"])


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _same_weights(test, a, b, what):
    a, b = _weights(a), _weights(b)
    test.assertEqual(a.keys(), b.keys())
    for k in a:
        test.assertTrue(torch.equal(a[k], b[k]), f"{what}: {k}")


class TestResume(unittest.TestCase):
    def test_pieces_equal_one_run(self):
        import train
        # one invocation: 2 seeds, 2 anchor budgets
        one = tiny_cfg(tempfile.mkdtemp(prefix="iclr_one_"), n_seeds=2, run_id="abc123",
                       **{"sweep.anchors": "[30,50]"})
        _run(one)
        # the same run in three pieces: seed 0 at one budget, then the second budget, then seed 100
        root = tempfile.mkdtemp(prefix="iclr_parts_")
        _run(tiny_cfg(root, n_seeds=1, run_id="abc123", **{"sweep.anchors": "[50]"}))
        _run(tiny_cfg(root, n_seeds=1, run_id="abc123", **{"sweep.anchors": "[30,50]"}))
        parts = tiny_cfg(root, n_seeds=2, run_id="abc123", **{"sweep.anchors": "[30,50]"})
        _run(parts)

        self.assertEqual(_results(one), _results(parts))
        for seed in (0, 100):
            _same_weights(self, train.ckpt_path(one.paths.checkpoints, "ddim", 2, 2, seed),
                          train.ckpt_path(parts.paths.checkpoints, "ddim", 2, 2, seed), f"seed {seed}")
        # everything lives in the named run folder
        rd = os.path.join(root, "output", "abc123")
        self.assertTrue(os.path.isdir(os.path.join(rd, "checkpoints", "ddim", "unweighted", "checkpoints")))
        self.assertEqual(len(open(os.path.join(rd, "history.log")).readlines()), 3)

    def test_changed_settings_refused(self):
        import runstate
        root = tempfile.mkdtemp(prefix="iclr_lock_")
        runstate.check_and_record(tiny_cfg(root, run_id="r1"))
        # selecting other cells / seeds / budgets / variant / T_true is fine
        runstate.check_and_record(tiny_cfg(root, run_id="r1", n_seeds=3, **{
            "sweep.d": "[4]", "sweep.anchors": "[10,20]", "data.weighted": "true", "process.T_true": 30}))
        # changing what a cell computes is not
        with self.assertRaises(ValueError):
            runstate.check_and_record(tiny_cfg(root, run_id="r1", **{"train.base_steps": 151}))
        with self.assertRaises(ValueError):
            runstate.check_and_record(tiny_cfg(root, run_id="r1", **{"classifier.models.polar3.degree": 5}))
        runstate.check_and_record(tiny_cfg(root, run_id="r1", strict_run=False, **{"train.base_steps": 151}))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_graph_grouping_does_not_change_models(self):
        """Seeds trained together in one CUDA graph == each trained alone (per-seed RNGs)."""
        import train
        together = tiny_cfg(tempfile.mkdtemp(prefix="iclr_gpu_all_"), device="cuda", n_seeds=2)
        train.run(together)
        alone = tempfile.mkdtemp(prefix="iclr_gpu_one_")
        train.run(tiny_cfg(alone, device="cuda", n_seeds=1))
        train.run(tiny_cfg(alone, device="cuda", n_seeds=1, seed=100))
        a_cfg = tiny_cfg(alone, device="cuda")
        for seed in (0, 100):
            _same_weights(self, train.ckpt_path(together.paths.checkpoints, "ddim", 2, 2, seed),
                          train.ckpt_path(a_cfg.paths.checkpoints, "ddim", 2, 2, seed), f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
