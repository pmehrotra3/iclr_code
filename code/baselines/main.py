"""code/baselines/main.py — code/main.py plus the pullback / hall_bench / basin stages and the
RODS / IQ processes. Same config (conf/config.yaml, via conf/extras.yaml), same run_id rules.

Stages, on top of train | evaluate | merge | visualize:
    pullback | pullback_viz                  jeffrey_old_stages/pullback_iq.py  (Prop. 4 normal vs IQ)
    pullback_sweep | pullback_sweep_viz      jeffrey_old_stages/pullback_iq.py  (repair over (d, K))
    hall_bench | hall_bench_viz | hall_bench_anim | hall_bench_traj   jeffrey_old_stages/hall_bench.py
    basin                                    baselines/basin.py  (every sampler vs the exact field)
Processes, on top of ddim | flow:
    rods_cas | rods_sas | iq                 baselines/process.py  (reuse the ddim checkpoints)

    python code/baselines/main.py 'stages=[pullback,pullback_viz]' 'pullback.d=[2]' 'pullback.K=[2]'
    python code/baselines/main.py 'stages=[hall_bench,hall_bench_viz]' 'bench.d=[2,4]'
    python code/baselines/main.py process=rods_cas 'stages=[evaluate,visualize]'
    python code/baselines/main.py 'stages=[basin]'

The pullback / psweep / bench / basin blocks only steer these stages, so they are left out of
the settings recorded in output/<run_id>/run.json; the run check (code/runstate.py) runs only
when a stage of code/main.py is requested.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # as code/main.py

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/

import train as train_stage                                  # noqa: E402
import evaluate as eval_stage                                # noqa: E402
import visualize as viz_stage                                # noqa: E402
import runstate                                              # noqa: E402
import baselines.process as baseline_procs                   # noqa: E402  registers rods_* / iq
from baselines import basin                                  # noqa: E402
from jeffrey_old_stages import pullback_iq, hall_bench       # noqa: E402

CORE_STAGES = {
    "train": train_stage.run,
    "evaluate": eval_stage.run,
    "merge": eval_stage.merge,
    "visualize": viz_stage.run,
}
STAGES = {
    **CORE_STAGES,
    "pullback": pullback_iq.run,
    "pullback_viz": pullback_iq.viz,
    "pullback_sweep": pullback_iq.run_sweep,
    "pullback_sweep_viz": pullback_iq.sweep_viz,
    "hall_bench": hall_bench.run,
    "hall_bench_viz": hall_bench.viz,
    "hall_bench_anim": hall_bench.anim,
    "hall_bench_traj": hall_bench.traj,
    "basin": basin.run,
}
EXTRA_BLOCKS = ("pullback", "psweep", "bench", "basin")


@hydra.main(version_base=None, config_path="../../conf", config_name="extras")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    for stage in cfg.stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'. choose from {list(STAGES)}")
    baseline_procs.link_ddim_checkpoints(cfg)
    if any(s in CORE_STAGES for s in cfg.stages):
        try:
            overrides = [o for o in HydraConfig.get().overrides.task
                         if o.lstrip("+~").split(".")[0].split("=")[0] not in EXTRA_BLOCKS]
        except Exception:
            overrides = []
        core_cfg = cfg.copy()
        with open_dict(core_cfg):
            for k in EXTRA_BLOCKS:
                core_cfg.pop(k, None)
        rd = runstate.check_and_record(core_cfg, overrides)
        print(f"run '{cfg.run_id}' -> {rd}")
    for stage in cfg.stages:
        print(f"\n>>> STAGE: {stage}\n" + "-" * 40)
        STAGES[stage](cfg)
    print("\nAll requested stages complete.")


if __name__ == "__main__":
    main()
