#!/usr/bin/env bash
# Ring-atlas T sweep for every process: learned sampler at T=150 (trained on demand, cached in
# checkpoints/ unless train.force_retrain=true), true-score backtrack for T = 50..1000 step 50,
# predictors fit on the anchors and scored on the learned sampler. Runs one process at a time,
# each split by dimension across all GPUs.
#   scripts/run_atlas.sh                                 # ddim then flow
#   scripts/run_atlas.sh ddim                            # one process
#   scripts/run_atlas.sh ddim flow anchors.n_per_mode=1000 train.force_retrain=true
# Results: output/<date>/<process>/T_<T>/results.{json,csv} + summary.csv
# Figures: visualization/<date>/<process>/T_<T>/heatmap_{full_acc,hall_f1,mode_f1}.{png,pdf} + summary_vs_T
set -euo pipefail
cd "$(dirname "$0")/.."
procs=(); overrides=()
for a in "$@"; do case $a in *=*) overrides+=("$a");; *) procs+=("$a");; esac; done
[ ${#procs[@]} -gt 0 ] || procs=(ddim flow)
for proc in "${procs[@]}"; do
  PAR_STAGES=atlas FINAL_STAGES=atlas_merge,atlas_viz \
    scripts/run_parallel.sh "$proc" sweep=atlas classifier=atlas "run_tag=$(date +%Y-%m-%d)" "${overrides[@]}"
done
