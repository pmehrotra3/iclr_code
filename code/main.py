"""main.py — Hydra entry point: python code/main.py [overrides]

Stages (stages=[...]): train -> evaluate -> visualize; merge rebuilds results.json from the
cell files (scripts/main.sh evaluates per d, then merges once). Models go to checkpoints/,
results to output/<run_id>/ (run_id is a NAME, default abc123); every invocation adds to them,
skipping work already done (code/runstate.py checks the settings agree).

    python code/main.py                                    # the configured sweep
    python code/main.py sweep.d=[16,64] n_seeds=3          # add cells / seeds to the run
    python code/main.py process=flow process.T_true=500    # another process / exact-field T
    python code/main.py stages=[visualize]                 # re-plot only
"""
from __future__ import annotations
import os
import sys
# the eval stage holds a few multi-GB anchor tensors; expandable segments stop the caching
# allocator from fragmenting around them (must be set before torch is imported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# make sibling stage modules importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as train_stage      # noqa: E402
import evaluate as eval_stage    # noqa: E402
import visualize as viz_stage    # noqa: E402
import runstate                  # noqa: E402

STAGES = {
    "train": train_stage.run,
    "evaluate": eval_stage.run,
    "merge": eval_stage.merge,          # rebuild results.json from every cell file of the run
    "visualize": viz_stage.run,
}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    for stage in cfg.stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'. choose from {list(STAGES)}")
    # same run_id = same run: check the settings against output/<run_id>/run.json and log the
    # invocation (its resolved config goes to output/<run_id>/invocations/)
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        overrides = []
    rd = runstate.check_and_record(cfg, overrides)
    print(f"run '{cfg.run_id}' -> {rd}")
    for stage in cfg.stages:
        print(f"\n>>> STAGE: {stage}\n" + "-" * 40)
        STAGES[stage](cfg)
    print("\nAll requested stages complete.")


if __name__ == "__main__":
    main()
