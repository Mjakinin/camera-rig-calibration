#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REAL_FINAL_DIR="results/real_vehicle_data/real_05x_4k_3hz_v1/99_FINAL_RESULTS"
SIM_FINAL_DIR="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
INNER_RUNNER="run/run_full_final_rerun.sh"
SECTION="all"

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  if [[ "${args[$index]}" == "--section" && $((index + 1)) -lt ${#args[@]} ]]; then
    SECTION="${args[$((index + 1))]}"
  fi
done

mkdir -p "$REAL_FINAL_DIR" "$SIM_FINAL_DIR"
rm -rf results/_full_final_rerun_logs

TMP_DIR="$(mktemp -d -t camera-rig-live-XXXXXX)"
TMP_LOG="$TMP_DIR/OVERNIGHT_LIVE.log"
STARTED_AT="$(date --iso-8601=seconds)"
RUN_PID=""
SYNC_PID=""

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

write_status() {
  local state="$1"
  local exit_code="${2:-}"
  local last_line=""
  local temporary

  if [[ -s "$TMP_LOG" ]]; then
    last_line="$(tail -n 1 "$TMP_LOG" | tr '\t\r\n' '   ')"
  fi

  for target in "${status_targets[@]}"; do
    mkdir -p "$(dirname "$target")"
    temporary="${target}.tmp.$$"
    {
      echo "state=$state"
      echo "section=$SECTION"
      echo "wrapper_pid=$$"
      echo "pipeline_pid=${RUN_PID:-NOT_STARTED}"
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

mirror_log() {
  local target
  for target in "${log_targets[@]}"; do
    mkdir -p "$(dirname "$target")"
    cp -f "$TMP_LOG" "$target"
  done
}

cleanup() {
  local code=$?
  set +e
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  if [[ -f "$TMP_LOG" ]]; then
    mirror_log
  fi
  if [[ "$code" -eq 0 ]]; then
    write_status "COMPLETED" "$code"
  else
    write_status "FAILED" "$code"
  fi
  rm -rf "$TMP_DIR"
  exit "$code"
}
trap cleanup EXIT INT TERM

: > "$TMP_LOG"
write_status "STARTING"

bash "$INNER_RUNNER" "$@" > "$TMP_LOG" 2>&1 &
RUN_PID=$!
write_status "RUNNING"

(
  while kill -0 "$RUN_PID" 2>/dev/null; do
    mirror_log
    write_status "RUNNING"
    sleep 2
  done
) &
SYNC_PID=$!

set +e
wait "$RUN_PID"
RUN_CODE=$?
set -e

kill "$SYNC_PID" 2>/dev/null || true
wait "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
mirror_log

if [[ "$RUN_CODE" -ne 0 ]]; then
  write_status "FAILED" "$RUN_CODE"
  exit "$RUN_CODE"
fi

write_status "COMPLETED" 0
exit 0
