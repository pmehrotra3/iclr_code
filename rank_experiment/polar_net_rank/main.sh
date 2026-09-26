#!/usr/bin/env bash
# rank_experiment/polar_net_rank/main.sh -- rank of the polar network's projections on the points
# of one pulled-back sphere, for DDIM and flow matching: rank_experiment/polar_net_rank/abc/{ddim,flow}
#
#   ./rank_experiment/polar_net_rank/main.sh            the full grid of config.yaml
#   ./rank_experiment/polar_net_rank/main.sh quick      a small grid, 2,000 anchors per mode
#   ./rank_experiment/polar_net_rank/main.sh tables     rebuild the .tex and .png from the saved CSVs, no compute
#
# Extra arguments go to polar_net_rank.py, e.g.   ... full --process flow    ... full --device cpu
# An interrupted run resumes where it stopped (run.resume in config.yaml).
set -euo pipefail
cd "$(dirname "$0")/../.."                   # repo root: the paths in config.yaml start here
PY=${PY:-python3}                            # needs numpy, torch, scipy, matplotlib, pyyaml
MODE=${1:-full}
[ $# -gt 0 ] && shift
SCRIPT=rank_experiment/polar_net_rank/polar_net_rank.py
CONFIG=rank_experiment/polar_net_rank/config.yaml
case "$MODE" in
  full)   exec "$PY" "$SCRIPT" --config "$CONFIG" "$@" ;;
  quick)  exec "$PY" "$SCRIPT" --config "$CONFIG" --quick "$@" ;;
  tables) exec "$PY" "$SCRIPT" --config "$CONFIG" --tables-only "$@" ;;
  *) echo "usage: $0 [full|quick|tables] [extra args for polar_net_rank.py]" >&2; exit 1 ;;
esac
