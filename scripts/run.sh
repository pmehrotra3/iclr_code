#!/usr/bin/env bash
# The experiment, for one or both processes: train the learned samplers (cached), run the atlas
# with the (d, K) grid split by dimension across the visible GPUs, merge the shards, draw the
# figures. Every extra argument is a Hydra override and goes to every job.
#   scripts/run.sh                                    # ddim and flow, sweep=full, classifier=atlas
#   scripts/run.sh ddim                               # one process
#   scripts/run.sh ddim flow sweep=d2 classifier=fast # d = 2 only, the four main predictors
#   scripts/run.sh ddim stages=[atlas_viz]            # re-draw from output/ddim/atlas_results.json
#   GPUS=0,1 JOBS_PER_GPU=2 scripts/run.sh flow       # which GPUs, how many shards each
# Results: output/[<run_tag>/]<process>/{atlas_results.json,summary.csv,T_<T>/,best_T_table.*,
#          summary_vs_T.*}; logs in output/_logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
export ATLAS_ROOT=$PWD
[ -f venv/bin/activate ] && source venv/bin/activate
procs=(); overrides=()
for a in "$@"; do case $a in *=*) overrides+=("$a");; *) procs+=("$a");; esac; done
[ ${#procs[@]} -gt 0 ] || procs=(ddim flow)
if [ -n "${GPUS:-}" ]; then IFS=, read -ra gpu <<< "$GPUS"; else
  mapfile -t gpu < <(nvidia-smi --list-gpus 2>/dev/null | awk '{print NR-1}'); [ ${#gpu[@]} -gt 0 ] || gpu=(0); fi
slots=$(( ${#gpu[@]} * ${JOBS_PER_GPU:-2} ))
mkdir -p output/_logs

one_process() {   # $1 = process; shards by d, then merges and draws
  local proc=$1 tag dims i=0 d g
  tag=$(for a in "${overrides[@]}"; do case $a in run_tag=*) echo "${a#run_tag=}";; esac; done)
  dims=$(python "code/$proc/main.py" "${overrides[@]}" --cfg job --package sweep.d | grep -oE '[0-9]+' | tr '\n' ' ')
  rm -rf "output/_parts/$proc"
  for d in $dims; do
    g=${gpu[$((i % ${#gpu[@]}))]}; i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g python "code/$proc/main.py" "${overrides[@]}" "stages=[train,atlas]" "sweep.d=[$d]" \
        device=cuda "paths.output=$PWD/output/_parts/$proc/d$d" > "output/_logs/${proc}${tag:+_$tag}_d$d.log" 2>&1 &
    if (( i % slots == 0 )); then wait; fi
  done
  wait
  CUDA_VISIBLE_DEVICES=${gpu[0]} python "code/$proc/main.py" "${overrides[@]}" "stages=[atlas_merge,atlas_viz]" \
      2>&1 | tee "output/_logs/${proc}${tag:+_$tag}_merge_viz.log"
  rm -rf "output/_parts/$proc"
}

for proc in "${procs[@]}"; do one_process "$proc" & done
wait
