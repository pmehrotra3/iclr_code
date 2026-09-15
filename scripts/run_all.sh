#!/usr/bin/env bash
# Every process, end to end: train (sequential per process), parallel evaluate, figures.
#   scripts/run_all.sh                       # ddim + flow with defaults
#   scripts/run_all.sh classifier=polar      # pass any overrides through
set -euo pipefail
cd "$(dirname "$0")/.."
for proc in ddim flow; do
  scripts/run.sh "$proc" stages=[train] "$@"
  scripts/run_parallel.sh "$proc" "$@"
done
