"""common/stages — the pipeline: train -> atlas -> atlas_viz. Each stage is run(cfg) -> dict."""
from __future__ import annotations
from omegaconf import DictConfig, OmegaConf

from common.stages import train, atlas, atlas_viz

STAGES = {
    "train": train.run,              # learned samplers, one per (d, K), cached in checkpoints/
    "atlas": atlas.run,              # plant -> backtrack -> fit -> score (the experiment)
    "atlas_merge": atlas.merge,      # combine per-d shards written by scripts/run.sh
    "atlas_viz": atlas_viz.run,      # heatmaps, curves, tables from atlas_results.json
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
