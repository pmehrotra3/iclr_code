"""common/main.py — the Hydra entry point every process folder wraps.

    python code/ddim/main.py                  # full pipeline for ddim
    python code/flow/main.py stages=[evaluate,visualize] classifier=ladder sweep=ladder

Each process folder's main.py calls run(<its conf dir>); the config there composes the
shared groups from common/conf (base, sweep/*, classifier/*) via hydra.searchpath.
"""
from __future__ import annotations
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# ${range:a,b,step} -> [a, a+step, ..., < b]   (used for T sweeps in the sweep configs)
OmegaConf.register_new_resolver("range", lambda a, b, s=1: list(range(int(a), int(b), int(s))))

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../code
if CODE not in sys.path:                     # make `common`, `ddim`, `flow` importable without install
    sys.path.insert(0, CODE)


def run(conf_dir: str) -> None:
    from common.stages import run_stages

    @hydra.main(version_base=None, config_path=conf_dir, config_name="config")
    def _main(cfg: DictConfig) -> None:
        run_stages(cfg)

    _main()
