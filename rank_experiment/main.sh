#!/usr/bin/env bash
# rank_experiment/main.sh -- run the rank experiment for DDIM and flow matching and build its
# tables (Euclidean, polar), one folder per process: rank_experiment/abc/ddim, rank_experiment/abc/flow
#
#   ./rank_experiment/main.sh            the full grid of config.yaml
#   ./rank_experiment/main.sh quick      a small grid, a few minutes
#   ./rank_experiment/main.sh tables     rebuild the .tex and .png from the saved CSVs, no compute
#
# Extra arguments go to rank_sweep.py, e.g.   ./rank_experiment/main.sh full --process flow   (one process)
#                                             ./rank_experiment/main.sh full --device cpu
# An interrupted run resumes where it stopped (run.resume in config.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root: the paths in config.yaml start here
PY=${PY:-python3}                            # needs numpy, torch, matplotlib, pyyaml (scipy optional)
MODE=${1:-full}
[ $# -gt 0 ] && shift
case "$MODE" in
  full)   exec "$PY" rank_experiment/rank_sweep.py --config rank_experiment/config.yaml "$@" ;;
  quick)  exec "$PY" rank_experiment/rank_sweep.py --config rank_experiment/config.yaml --quick "$@" ;;
  tables) exec "$PY" rank_experiment/rank_sweep.py --config rank_experiment/config.yaml --tables-only "$@" ;;
  *) echo "usage: $0 [full|quick|tables] [extra args for rank_sweep.py]" >&2; exit 1 ;;
esac
