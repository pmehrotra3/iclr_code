#!/usr/bin/env bash
# rank_experiment/polar_net_rank/merge.sh -- gather the per-cell results of euler.sbatch
# (cells/<process>_d<d>_K<K>/<process>/rank.csv) into abc/<process>/rank.csv and build the tables.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
HERE=rank_experiment/polar_net_rank
for p in ddim flow; do
  files=( $HERE/cells/${p}_d*/$p/rank.csv )
  [ -e "${files[0]}" ] || { echo "no cells for $p"; continue; }
  mkdir -p $HERE/abc/$p
  { head -1 "${files[0]}"; for f in "${files[@]}"; do tail -n +2 "$f"; done; } > $HERE/abc/$p/rank.csv
  echo "$p: ${#files[@]} cells"
done
exec "$PY" $HERE/polar_net_rank.py --config $HERE/config.yaml --tables-only
