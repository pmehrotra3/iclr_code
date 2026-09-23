"""
main.py — Hydra entry point.

A run is a NAME (run_id=abc123, the default); everything it produces -- checkpoints, gt caches,
results, figures, logs -- lives in output/<run_id>/, and every invocation with that name adds to
it and resumes where it stopped (code/runstate.py). Existing checkpoints / cell files are reused
unless train.force_retrain / eval.force.

Run the full pipeline:
    python code/main.py run_id=abc123

In pieces (same files as one invocation with sweep.d=[16,64] n_seeds=3):
    python code/main.py run_id=abc123 sweep.d=[16]
    python code/main.py run_id=abc123 sweep.d=[16,64]               # d=16 is skipped
    python code/main.py run_id=abc123 sweep.d=[16,64] n_seeds=3     # only seeds 100, 200 are new
    python code/main.py run_id=abc123 process.T_true=500            # adds another T

Run a custom sweep:
    python code/main.py sweep.d=[2,8,32] sweep.K=[8] sweep.anchors=[5000,50000,200000]

Run only some stages:
    python code/main.py stages=[evaluate,visualize]
    python code/main.py stages=[visualize]          # just re-plot from output/results.json

Other handy overrides:
    python code/main.py device=cpu train.force_retrain=true process.T_true=200
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
