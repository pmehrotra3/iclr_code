#!/usr/bin/env bash
# scripts/main.sh — GPU-parallel sweep, driven ENTIRELY by command-line overrides.
# conf/ is never modified; everything below is passed to code/main.py as Hydra overrides.
#
#   d        : powers of two   (DIMS)    -> sweep.d  (a full list in ONE run; d is a heatmap axis)
#   K        : 2 .. 32 modes    (KS)      -> sweep.K
#   R        : mode-sphere radius scales with d IN CODE (train.py: R = data.radius * sqrt(d/2))
#   T        : exact-score steps (TS)     -> process.T_true
#   T_train  : learned steps     (TTRAIN) -> process.T_train  (fixed = 150)
#   anchors  : per-mode budgets  (ANCHORS)-> sweep.anchors
#
# One shared timestamp per invocation. Output layout:
#     output/<timestamp>/<process>/T<T>/{results.json, table.tex, table.png, <model>.png,
#                                        anchors_<b>/<model>.png}
#
# Two phases, so parallel jobs never race on checkpoints:
#   1) TRAIN     : one job per process (all d,K) -> data/<process>/checkpoints/ (processes differ)
#   2) EVAL+VIZ  : one job per (process, T), reusing the cached checkpoints, spread across GPUs.
# One process per GPU; NGPU jobs at a time.
#
#   ./scripts/main.sh
#   DEVICE=cpu NGPU=2 DIMS="2 4" TS="100 500" ANCHORS="[2000,10000]" ./scripts/main.sh

set -o pipefail
set -f                                   # no globbing, so [2,4,8] overrides stay literal
cd "$(dirname "$0")/.."
export ATLAS_ROOT="$PWD"

PROCESSES=${PROCESSES:-"ddim flow"}
DIMS=${DIMS:-"2 4 8 16 32 64"}                                   # all powers of two
KS=${KS:-"[2,4,8,16,32]"}                                        # 2 .. 32 modes
TS=${TS:-"100 200 300 400 500 1000"}                             # exact-score step counts
ANCHORS=${ANCHORS:-"[2000,5000,10000,20000,50000,100000]"}      # per-mode budgets
TTRAIN=${TTRAIN:-150}                                            # learned-sampler steps
RADBASE=${RADBASE:-2.0}                                          # data.radius base (scaled by sqrt(d/2) in code)
DEVICE=${DEVICE:-cuda}
EXTRA=${EXTRA:-}                                                 # extra overrides, appended verbatim

if [ -z "${NGPU:-}" ]; then
  command -v nvidia-smi >/dev/null 2>&1 && NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  NGPU=${NGPU:-4}
fi
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1

mkdir -p logs
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
DLIST="[$(echo $DIMS | tr ' ' ',')]"                            # "2 4 8" -> "[2,4,8]"

# ---- run the commands in JOBS[] ("logname|override args") across NGPU GPUs, one per GPU ----
run_pool() {
  local FREE=() RPID=() RGPU=() NP=() NG=() i=0 done_n=0 total=${#JOBS[@]} gpu j pid g log args g0
  for ((g0=0; g0<NGPU; g0++)); do FREE+=("$g0"); done
  while [ $done_n -lt $total ]; do
    while [ ${#FREE[@]} -gt 0 ] && [ $i -lt $total ]; do
      gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
      log="${JOBS[$i]%%|*}"; args="${JOBS[$i]#*|}"; i=$((i+1))
      echo ">>> [gpu $gpu] $log"
      ( CUDA_VISIBLE_DEVICES="$gpu" python code/main.py $args device="$DEVICE" > "logs/$log" 2>&1 \
          && echo "  [ok]   $log" || echo "  [FAIL] $log -> logs/$log" ) &
      RPID+=("$!"); RGPU+=("$gpu")
    done
    NP=(); NG=()
    for j in "${!RPID[@]}"; do
      pid=${RPID[$j]}; g=${RGPU[$j]}
      if kill -0 "$pid" 2>/dev/null; then NP+=("$pid"); NG+=("$g")
      else wait "$pid"; FREE+=("$g"); done_n=$((done_n + 1)); fi
    done
    RPID=(); RGPU=()
    [ ${#NP[@]} -gt 0 ] && { RPID=("${NP[@]}"); RGPU=("${NG[@]}"); }
    [ $done_n -lt $total ] && sleep 3
  done
}

echo "=============================================================="
echo " sweep $STAMP : $NGPU GPU(s), 1 process each"
echo " processes=[$PROCESSES]  d=$DLIST  K=$KS  T_true=[$TS]  T_train=$TTRAIN"
echo " anchors=$ANCHORS  radius(base)=$RADBASE (scaled by sqrt(d/2))  device=$DEVICE"
echo "=============================================================="

# ---- phase 1: train all (d,K) per process ----
echo "-- phase 1: train --"
JOBS=()
for P in $PROCESSES; do
  JOBS+=("${STAMP}_${P}_train.log|process=$P process.T_train=$TTRAIN sweep.d=$DLIST sweep.K=$KS data.radius=$RADBASE stages=[train] train.force_retrain=true run_id=$STAMP $EXTRA")
done
run_pool

# ---- phase 2: evaluate + visualize, one job per (process, T), reusing the checkpoints ----
echo "-- phase 2: evaluate + visualize --"
JOBS=()
for P in $PROCESSES; do
  for T in $TS; do
    JOBS+=("${STAMP}_${P}_T${T}.log|process=$P process.T_train=$TTRAIN process.T_true=$T sweep.d=$DLIST sweep.K=$KS sweep.anchors=$ANCHORS data.radius=$RADBASE stages=[evaluate,visualize] train.force_retrain=false run_id=$STAMP $EXTRA")
  done
done
run_pool

echo "=============================================================="
echo " sweep complete -> output/$STAMP/<process>/T<T>/"
echo "=============================================================="
