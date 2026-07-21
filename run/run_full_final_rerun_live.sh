#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REAL_FINAL_DIR="results/real_vehicle_data/real_05x_4k_3hz_v1/99_FINAL_RESULTS"
SIM_FINAL_DIR="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
INNER_RUNNER="run/run_full_final_rerun.sh"
SECTION="all"
STARTED_AT="$(date --iso-8601=seconds)"
STATUS_PID=""
RUN_CODE=""

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  if [[ "${args[$index]}" == "--section" && $((index + 1)) -lt ${#args[@]} ]]; then
    SECTION="${args[$((index + 1))]}"
  fi
done

mkdir -p "$REAL_FINAL_DIR" "$SIM_FINAL_DIR"
rm -rf results/_full_final_rerun_logs

status_targets=()
log_targets=()
case "$SECTION" in
  simulation)
    status_targets+=("$SIM_FINAL_DIR/LIVE_STATUS.txt")
    log_targets+=("$SIM_FINAL_DIR/OVERNIGHT_LIVE.log")
    ;;
  real)
    status_targets+=("$REAL_FINAL_DIR/LIVE_STATUS.txt")
    log_targets+=("$REAL_FINAL_DIR/OVERNIGHT_LIVE.log")
    ;;
  all|preflight)
    status_targets+=(
      "$SIM_FINAL_DIR/LIVE_STATUS.txt"
      "$REAL_FINAL_DIR/LIVE_STATUS.txt"
    )
    log_targets+=(
      "$SIM_FINAL_DIR/OVERNIGHT_LIVE.log"
      "$REAL_FINAL_DIR/OVERNIGHT_LIVE.log"
    )
    ;;
  *)
    echo "[ERROR] unsupported section: $SECTION" >&2
    exit 2
    ;;
esac

for target in "${log_targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  : > "$target"
done

latest_log_line() {
  local source="${log_targets[0]}"
  if [[ -s "$source" ]]; then
    tail -n 1 "$source" | tr '\t\r\n' '   '
  fi
}

write_status() {
  local state="$1"
  local exit_code="${2:-}"
  local last_line=""
  local temporary

  last_line="$(latest_log_line)"
  for target in "${status_targets[@]}"; do
    mkdir -p "$(dirname "$target")"
    temporary="${target}.tmp.$$"
    {
      echo "state=$state"
      echo "section=$SECTION"
      echo "wrapper_pid=$$"
      echo "started_at=$STARTED_AT"
      echo "updated_at=$(date --iso-8601=seconds)"
      if [[ -n "$exit_code" ]]; then
        echo "exit_code=$exit_code"
      fi
      echo "last_log_line=$last_line"
    } > "$temporary"
    mv -f "$temporary" "$target"
  done
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$STATUS_PID" ]]; then
    kill "$STATUS_PID" 2>/dev/null || true
    wait "$STATUS_PID" 2>/dev/null || true
  fi
  if [[ -n "$RUN_CODE" ]]; then
    if [[ "$RUN_CODE" -eq 0 ]]; then
      write_status "COMPLETED" "$RUN_CODE"
    else
      write_status "FAILED" "$RUN_CODE"
    fi
  elif [[ "$code" -eq 0 ]]; then
    write_status "COMPLETED" "$code"
  else
    write_status "INTERRUPTED" "$code"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

write_status "STARTING"
(
  while true; do
    write_status "RUNNING"
    sleep 5
  done
) &
STATUS_PID=$!

set +e
bash "$INNER_RUNNER" "$@" 2>&1 | tee "${log_targets[@]}"
RUN_CODE=${PIPESTATUS[0]}
set -e

kill "$STATUS_PID" 2>/dev/null || true
wait "$STATUS_PID" 2>/dev/null || true
STATUS_PID=""

if [[ "$RUN_CODE" -ne 0 ]]; then
  write_status "FAILED" "$RUN_CODE"
  exit "$RUN_CODE"
fi

write_status "COMPLETED" 0
exit 0
