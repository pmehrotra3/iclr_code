#!/usr/bin/env bash
# =====================================================================================
#  run_sweep.sh — launch ddim + flow, each over a T_true sweep, in PARALLEL.
#
#  - Each (sampler, T_true) is a separate job, run at the same time (max parallelism).
#  - Each job writes to its own output folder so nothing overwrites:
#        output/<sampler>/T<val>/ ,  visualization/<sampler>/T<val>/
#  - Live progress: every job streams to its own log AND we print a status line as jobs
#    start / finish.
#  - A timer prints total wall-clock at the end, plus per-job durations.
#
#  Usage:
#     export ATLAS_ROOT=$(pwd)      # from the repo root (where code/ conf/ live)
#     ./run_sweep.sh
#
#  Edit the CONFIG block below to change what gets swept.
# =====================================================================================
set -u

# ----------------------------- CONFIG (edit me) -----------------------------
SAMPLERS=(ddim flow)              # which samplers to run
T_TRUE_VALUES=(20 50 100 200)     # the T_true sweep
D_SWEEP="[2,4,8,16,32]"           # dimensions   (quoted; passed as-is to hydra)
K_SWEEP="[2,4,8,16]"              # modes
ANCHORS="[2000,10000,50000]"      # anchor counts
EXTRA_OVERRIDES=""                # e.g. "train.base_steps=1000 device=cuda"

# Max jobs to run at once. Default = ALL of them (true max parallelism).
# Lower this (e.g. MAX_PARALLEL=2) if you hit out-of-memory.
MAX_PARALLEL=$(( ${#SAMPLERS[@]} * ${#T_TRUE_VALUES[@]} ))
# ---------------------------------------------------------------------------

: "${ATLAS_ROOT:=$(pwd)}"
export ATLAS_ROOT
LOG_DIR="${ATLAS_ROOT}/output/_logs"
mkdir -p "${LOG_DIR}"

# ---- portable millisecond clock (works on macOS + Linux) ----
now_s() { date +%s; }
fmt_hms() {  # seconds -> Hh Mm Ss
  local s=$1; printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ---- build the job list ----
JOBS=()          # "sampler|T_true"
for smp in "${SAMPLERS[@]}"; do
  for T in "${T_TRUE_VALUES[@]}"; do
    JOBS+=("${smp}|${T}")
  done
done
TOTAL=${#JOBS[@]}

echo "======================================================================"
echo " Diffusion-atlas parallel sweep"
echo " samplers      : ${SAMPLERS[*]}"
echo " T_true values : ${T_TRUE_VALUES[*]}"
echo " total jobs    : ${TOTAL}"
echo " max parallel  : ${MAX_PARALLEL}"
echo " logs          : ${LOG_DIR}"
echo "======================================================================"

declare -A JOB_START     # pid -> start seconds
declare -A JOB_NAME      # pid -> readable name
declare -A JOB_LOG       # pid -> logfile
declare -a JOB_RESULTS   # "name  status  duration"

SWEEP_START=$(now_s)
launched=0
finished=0
running=0

launch_job() {
  local spec="$1"
  local smp="${spec%%|*}"
  local T="${spec##*|}"
  local name="${smp}_T${T}"
  local log="${LOG_DIR}/${name}.log"

  # per-job output + viz dirs, so parallel jobs never clash
  local out="${ATLAS_ROOT}/output/${smp}/T${T}"
  local viz="${ATLAS_ROOT}/visualization/${smp}/T${T}"

  # NOTE: brackets are quoted so zsh/bash never glob them
  python code/main.py \
      sampler="${smp}" \
      sweep.T_true="${T}" \
      "sweep.d=${D_SWEEP}" \
      "sweep.K=${K_SWEEP}" \
      "sweep.anchors=${ANCHORS}" \
      "paths.output=${out}" \
      "paths.viz=${viz}" \
      ${EXTRA_OVERRIDES} \
      > "${log}" 2>&1 &

  local pid=$!
  JOB_START[$pid]=$(now_s)
  JOB_NAME[$pid]="${name}"
  JOB_LOG[$pid]="${log}"
  launched=$((launched+1))
  running=$((running+1))
  echo "[$(date +%H:%M:%S)] START  ${name}   (job ${launched}/${TOTAL}, pid ${pid})  -> ${log}"
}

reap_one() {
  # wait for ANY job to finish, record its result
  local pid
  wait -n 2>/dev/null
  # find which of our tracked pids died
  for pid in "${!JOB_NAME[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; local rc=$?
      local dur=$(( $(now_s) - ${JOB_START[$pid]} ))
      local status="ok"; [ $rc -ne 0 ] && status="FAILED(rc=$rc)"
      finished=$((finished+1)); running=$((running-1))
      echo "[$(date +%H:%M:%S)] DONE   ${JOB_NAME[$pid]}   ${status}   $(fmt_hms $dur)   (${finished}/${TOTAL} complete)"
      JOB_RESULTS+=("$(printf '%-14s %-14s %s' "${JOB_NAME[$pid]}" "${status}" "$(fmt_hms $dur)")")
      unset 'JOB_NAME[$pid]' 'JOB_START[$pid]' 'JOB_LOG[$pid]'
      return 0
    fi
  done
}

# ---- main scheduling loop: keep up to MAX_PARALLEL running ----
idx=0
while [ $idx -lt $TOTAL ] || [ $running -gt 0 ]; do
  # fill up to the parallel limit
  while [ $running -lt $MAX_PARALLEL ] && [ $idx -lt $TOTAL ]; do
    launch_job "${JOBS[$idx]}"
    idx=$((idx+1))
  done
  # wait for at least one to finish before looping
  if [ $running -gt 0 ]; then
    reap_one
  fi
done

SWEEP_END=$(now_s)
TOTAL_DUR=$(( SWEEP_END - SWEEP_START ))

echo "======================================================================"
echo " ALL JOBS COMPLETE"
echo "----------------------------------------------------------------------"
printf ' %-14s %-14s %s\n' "JOB" "STATUS" "DURATION"
for line in "${JOB_RESULTS[@]}"; do echo " ${line}"; done
echo "----------------------------------------------------------------------"
echo " TOTAL WALL-CLOCK: $(fmt_hms ${TOTAL_DUR})"
echo " logs in: ${LOG_DIR}"
echo "======================================================================"
