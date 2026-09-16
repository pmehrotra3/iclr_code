#!/usr/bin/env bash
# Run ONE process with the sweep split by dimension across all visible GPUs, then merge the
# parts and draw the figures.  PAR_STAGES run per dimension; FINAL_STAGES run once at the end.
# JOBS_PER_GPU (default 2) dimensions share a GPU -- the models are small, so 2-3 per GPU is fine.
#   scripts/run_parallel.sh ddim                                  # evaluate -> merge,visualize
#   scripts/run_parallel.sh ddim classifier=ladder sweep=ladder   # capacity ladder
#   scripts/run_parallel.sh ddim eval.labels=true eval.tag=true   # exact-score control
#   PAR_STAGES=atlas FINAL_STAGES=atlas_merge,atlas_viz \
#       scripts/run_parallel.sh ddim sweep=atlas classifier=atlas # ring atlas over T (trains T=150 models on demand)
set -euo pipefail
cd "$(dirname "$0")/.."
proc=${1:?usage: scripts/run_parallel.sh <process> [hydra overrides...]}; shift
export ATLAS_ROOT=$PWD
mkdir -p output/_logs
PAR_STAGES=${PAR_STAGES:-evaluate}; FINAL_STAGES=${FINAL_STAGES:-merge,visualize}
rm -rf "output/_parts/$proc"        # stale parts from an earlier run would be merged in
ngpu=$(nvidia-smi --list-gpus 2>/dev/null | wc -l); [ "$ngpu" -gt 0 ] || ngpu=1
slots=$((ngpu * ${JOBS_PER_GPU:-2}))
# the dimensions in the chosen sweep (honours a sweep=... or sweep.d=[...] override)
dims=$(python "code/$proc/main.py" "$@" --cfg job --package sweep.d | grep -oE '[0-9]+' | tr '\n' ' ')
i=0
for d in $dims; do
  gpu=$((i % ngpu)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$gpu python "code/$proc/main.py" "$@" "stages=[$PAR_STAGES]" "sweep.d=[$d]" device=cuda \
      "paths.output=$PWD/output/_parts/$proc/d$d" > "output/_logs/${proc}_eval_d$d.log" 2>&1 &
  # keep at most JOBS_PER_GPU jobs per GPU in flight
  if (( i % slots == 0 )); then wait; fi
done
wait
python "code/$proc/main.py" "$@" "stages=[$FINAL_STAGES]" 2>&1 | tee "output/_logs/${proc}_merge_viz.log"
rm -rf "output/_parts/$proc"   # merged into output/$proc; the shards are byte-identical duplicates

