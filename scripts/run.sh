#!/usr/bin/env bash
# Run the pipeline for ONE process on one GPU.
#   scripts/run.sh ddim                                  # train -> evaluate -> visualize
#   scripts/run.sh flow stages=[evaluate,visualize] classifier=ladder sweep=ladder
set -euo pipefail
cd "$(dirname "$0")/.."
proc=${1:?usage: scripts/run.sh <process> [hydra overrides...]}; shift
export ATLAS_ROOT=$PWD
mkdir -p output/_logs
python "code/$proc/main.py" "$@" 2>&1 | tee "output/_logs/${proc}_$(date +%Y%m%d_%H%M%S).log"
