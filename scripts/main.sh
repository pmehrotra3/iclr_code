#!/usr/bin/env bash
# scripts/main.sh — GPU-parallel sweep, driven ENTIRELY by command-line overrides.
# conf/ is never modified; everything below is passed to code/main.py as Hydra overrides.
#
#   d        : powers of two                  (DIMS)    -> sweep.d, ONE run per d
#   K        : 2 .. 32 modes                   (KS)      -> sweep.K
#   R        : mode-sphere radius, SCALES WITH d         -> data.radius = f(d), per d
#   T        : exact-score (atlas) steps       (TS)      -> process.T_true
#   T_train  : learned-sampler steps           (TTRAIN)  -> process.T_train (fixed = 150)
#   anchors  : per-mode anchor budgets         (ANCHORS) -> sweep.anchors
#
# Parallelism: one job PER GPU (NGPU jobs at once). A "unit" is one (process, d); it runs its
# whole T-loop on a single GPU (the learned model depends only on (d,K,R,T_train) -> identical
# across T_true, so it trains once and reuses across T). Each d gets its own paths.data subtree,
# so parallel units never collide on checkpoints/manifests. Output: output/<process>/T<T>/d<d>/.
#
#   ./scripts/main.sh
#   NGPU=4 ./scripts/main.sh
#   DEVICE=cpu NGPU=2 DIMS="2 4" TS="100 500" ANCHORS="[2000,10000]" ./scripts/main.sh
#
# NOTE: the two largest budgets (500000, 1000000 anchors PER MODE) are enormous -- with K and d
# that is tens of millions of points backtracked over up to 1000 steps, likely out of memory.
# Drop them (or use 50000/100000) unless you are on a big-memory GPU box.

set -o pipefail
cd "$(dirname "$0")/.."
export ATLAS_ROOT="$PWD"

PROCESSES=${PROCESSES:-"ddim flow"}
DIMS=${DIMS:-"2 4 8 16 32 64 128 256 512 1024"}                                   # all powers of two
KS=${KS:-"[2,4,8,16,32]"}                                        # 2 .. 32 modes
TS=${TS:-"100 200 300 400 500 1000"}                             # exact-score step counts
ANCHORS=${ANCHORS:-"[2000,5000,10000,20000,50000,100000]"}     # per-mode budgets
TTRAIN=${TTRAIN:-150}                                            # learned-sampler steps
RADIUS_EXPR=${RADIUS_EXPR:-"sqrt(2*d)"}                          # R(d); awk expr in `d` (d=2 -> 2.0)
DEVICE=${DEVICE:-cuda}
EXTRA=${EXTRA:-}                                                 # any extra overrides, appended verbatim

# one job per GPU: detect the GPU count, default 4
if [ -z "${NGPU:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  NGPU=${NGPU:-4}
fi
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1

mkdir -p logs
STAMP=$(date +%Y-%m-%d_%H-%M-%S)

run_unit() {   # $1=gpu $2=process $3=d  -- the whole T-loop for one (process,d) on one GPU
  local gpu=$1 P=$2 d=$3 R first=1 T FR LOG
  R=$(awk "BEGIN{d=$d; print $RADIUS_EXPR}")
  for T in $TS; do
    FR=false; [ "$first" = 1 ] && FR=true; first=0        # train once per (process,d), reuse across T
    LOG="logs/${STAMP}_${P}_d${d}_T${T}.log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" python code/main.py \
          process="$P" process.T_train="$TTRAIN" process.T_true="$T" \
          "sweep.d=[$d]" "sweep.K=$KS" "sweep.anchors=$ANCHORS" \
          "data.radius=$R" "paths.data=${ATLAS_ROOT}/data/d${d}" \
          "train.force_retrain=$FR" device="$DEVICE" "run_id=T${T}/d${d}" \
          $EXTRA > "$LOG" 2>&1; then
      echo "  [FAIL] $P d=$d T=$T  -> $LOG"
      return 1
    fi
    echo "  [ok]   $P d=$d T=$T  (gpu $gpu)"
  done
}

# unit list: one (process, d) per entry
UNITS=()
for P in $PROCESSES; do for d in $DIMS; do UNITS+=("$P|$d"); done; done
TOTAL=${#UNITS[@]}

echo "=============================================================="
echo " GPU-parallel sweep: $NGPU GPU(s), 1 process each, $TOTAL units"
echo " processes=[$PROCESSES]  d=[$DIMS]  K=$KS"
echo " T_true=[$TS]  T_train=$TTRAIN  anchors=$ANCHORS  device=$DEVICE  R(d)=$RADIUS_EXPR"
echo "=============================================================="

FREE=(); for ((g=0; g<NGPU; g++)); do FREE+=("$g"); done       # free GPU ids
RPID=(); RGPU=()                                                # running jobs (parallel arrays)
i=0; done_n=0; fail=0

while [ $done_n -lt $TOTAL ]; do
  # fill every free GPU with the next unit
  while [ ${#FREE[@]} -gt 0 ] && [ $i -lt $TOTAL ]; do
    gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
    P=${UNITS[$i]%|*}; d=${UNITS[$i]#*|}; i=$((i+1))
    echo ">>> launch  $P  d=$d  on gpu $gpu   ($((i))/$TOTAL)"
    run_unit "$gpu" "$P" "$d" &
    RPID+=("$!"); RGPU+=("$gpu")
  done
  # reap finished jobs, hand their GPU back to the pool
  NP=(); NG=()
  for j in "${!RPID[@]}"; do
    pid=${RPID[$j]}; g=${RGPU[$j]}
    if kill -0 "$pid" 2>/dev/null; then
      NP+=("$pid"); NG+=("$g")
    else
      wait "$pid"; rc=$?
      FREE+=("$g"); done_n=$((done_n + 1))
      [ $rc -ne 0 ] && fail=$((fail + 1))
    fi
  done
  RPID=(); RGPU=()
  [ ${#NP[@]} -gt 0 ] && { RPID=("${NP[@]}"); RGPU=("${NG[@]}"); }
  [ $done_n -lt $TOTAL ] && sleep 3
done

echo "=============================================================="
echo " sweep complete: $TOTAL units, $fail failed"
echo " results under  output/<process>/T<T>/d<d>/   checkpoints under  data/d<d>/"
echo "=============================================================="
[ $fail -eq 0 ]
