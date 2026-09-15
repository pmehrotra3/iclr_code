#!/usr/bin/env bash
# =====================================================================================
#  run_all.sh — one command to run BOTH samplers (ddim + flow) on a single NVIDIA GPU,
#  in parallel, then emit figures + a ready-to-paste LaTeX table per sampler.
#
#  Live progress + a wall-clock timer are printed to the console. Each sampler streams
#  its full log to output/_logs/<sampler>.log.
#
#  Usage:
#     export ATLAS_ROOT=$(pwd)     # from the repo root (code/ conf/ live here)
#     ./run_all.sh
# =====================================================================================
set -u

# ----------------------------- CONFIG (edit me) -----------------------------
SAMPLERS=(ddim flow)
D_SWEEP="[2,4,8,16,32]"
K_SWEEP="[2,4,8,16]"
ANCHORS="[2000,10000,50000]"
T_TRUE=100
EXTRA_OVERRIDES="device=cuda"     # single NVIDIA GPU
# ---------------------------------------------------------------------------

: "${ATLAS_ROOT:=$(pwd)}"
export ATLAS_ROOT
LOG_DIR="${ATLAS_ROOT}/output/_logs"
mkdir -p "${LOG_DIR}"

now_s() { date +%s; }
fmt_hms() { local s=$1; printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

echo "======================================================================"
echo " Diffusion-atlas: ddim + flow on one GPU (parallel)"
echo " d=${D_SWEEP}  K=${K_SWEEP}  anchors=${ANCHORS}  T_true=${T_TRUE}"
echo " logs: ${LOG_DIR}"
echo "======================================================================"

declare -A PID_NAME PID_START
declare -a RESULTS
START=$(now_s)

for smp in "${SAMPLERS[@]}"; do
  log="${LOG_DIR}/${smp}.log"
  python code/main.py \
      sampler="${smp}" \
      sweep.T_true="${T_TRUE}" \
      "sweep.d=${D_SWEEP}" \
      "sweep.K=${K_SWEEP}" \
      "sweep.anchors=${ANCHORS}" \
      ${EXTRA_OVERRIDES} \
      > "${log}" 2>&1 &
  pid=$!
  PID_NAME[$pid]="${smp}"; PID_START[$pid]=$(now_s)
  echo "[$(date +%H:%M:%S)] START  ${smp}  (pid ${pid})  -> ${log}"
done

# live progress: poll logs + reap finishers
remaining=${#PID_NAME[@]}
while [ "${remaining}" -gt 0 ]; do
  sleep 5
  for pid in "${!PID_NAME[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      dur=$(( $(now_s) - ${PID_START[$pid]} ))
      status="ok"; [ $rc -ne 0 ] && status="FAILED(rc=$rc)"
      echo "[$(date +%H:%M:%S)] DONE   ${PID_NAME[$pid]}  ${status}  $(fmt_hms $dur)"
      RESULTS+=("$(printf '%-8s %-14s %s' "${PID_NAME[$pid]}" "${status}" "$(fmt_hms $dur)")")
      unset 'PID_NAME[$pid]' 'PID_START[$pid]'
      remaining=$((remaining-1))
    else
      # heartbeat: last progress line from this sampler's log
      last=$(grep -E '^\[(train|eval)' "${LOG_DIR}/${PID_NAME[$pid]}.log" 2>/dev/null | tail -1)
      [ -n "${last}" ] && echo "   .. ${PID_NAME[$pid]}: ${last}"
    fi
  done
done

TOTAL=$(( $(now_s) - START ))
echo "======================================================================"
echo " ALL DONE"
printf ' %-8s %-14s %s\n' "SAMPLER" "STATUS" "DURATION"
for line in "${RESULTS[@]}"; do echo " ${line}"; done
echo "----------------------------------------------------------------------"
echo " TOTAL WALL-CLOCK: $(fmt_hms ${TOTAL})"
echo " figures + LaTeX tables in: ${ATLAS_ROOT}/visualization/<sampler>/"
echo "   - table_<sampler>.tex   (paste into the paper; needs \\usepackage{booktabs})"
echo "======================================================================"
