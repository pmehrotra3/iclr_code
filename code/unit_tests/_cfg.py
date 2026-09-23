"""A tiny, fast config for the tests: d=2, K=2, a few hundred training steps, small anchor and
eval sets. Everything is written under `root` (output/<run_id>/...)."""
import os

from hydra import compose, initialize_config_dir

from unit_tests import _CODE

CONF = os.path.join(os.path.dirname(_CODE), "conf")


def tiny_cfg(root, device="cpu", **over):
    base = {"device": device, "paths.root": root, "sweep.d": "[2]", "sweep.K": "[2]",
            "sweep.anchors": "[30]", "n_seeds": 1, "process.T_train": 50, "process.T_true": 20,
            "train.base_steps": 150, "train.ema_warmup": 10, "train.n_train": 2000,
            "train.probe_n": 300, "train.max_attempts": 1, "eval.n_eval_per_mode": 300,
            "anchors.n_calibrate": 200, "classifier._shared.min_steps": 20,
            "classifier._shared.epochs": 2, "classifier._shared.ensemble": 1}
    base.update(over)
    with initialize_config_dir(config_dir=CONF, version_base=None):
        return compose("config", overrides=[f"{k}={v}" for k, v in base.items()])
