"""common/stages — the pipeline. Each stage is run(cfg) -> dict and is process-agnostic."""
from __future__ import annotations
from omegaconf import DictConfig, OmegaConf

from common.stages import train, evaluate, merge, seedmap, visualize, atlas, atlas_viz, pullback_iq

STAGES = {
    "train": train.run,
    "evaluate": evaluate.run,
    "merge": merge.run,
    "seedmap": seedmap.run,
    "visualize": visualize.run,
    "atlas": atlas.run,
    "atlas_merge": atlas.merge,
    "atlas_viz": atlas_viz.run,
    "pullback": pullback_iq.run,
    "pullback_viz": pullback_iq.viz,
}


def run_stages(cfg: DictConfig) -> dict:
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    out = {}
    for stage in cfg.stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; choose from {list(STAGES)}")
        print(f"\n>>> STAGE: {stage}  [{cfg.process}]\n" + "-" * 40, flush=True)
        out[stage] = STAGES[stage](cfg)
    print("\nAll requested stages complete.")
    return out
