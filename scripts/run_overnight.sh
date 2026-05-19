#!/usr/bin/env bash
# Runs the full classical pipeline end-to-end with timestamped logging.
# Each stage logs to logs/ and aborts the chain on failure.
# macOS / Linux equivalent of run_overnight.ps1.

set -u  # not -e: we handle exit codes per stage manually

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")

run_stage() {
    local name=$1
    local script=$2
    echo ""
    echo "========================================"
    echo "  STAGE: $name"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    local log="$LOG_DIR/${TS}_${name}.log"
    uv run python -u "$script" 2>&1 | tee "$log"
    local code=${PIPESTATUS[0]}
    echo ""
    if [[ $code -ne 0 ]]; then
        echo "[FAIL] $name exit=$code  (log: $log)"
        return $code
    fi
    echo "[OK] $name finished at $(date '+%H:%M:%S')  (log: $log)"
    return 0
}

echo "===== OVERNIGHT classical pipeline ====="
echo "Logs dir: $LOG_DIR"
started_at=$(date +%s)

run_stage "train"   "scripts/run_classical_train.py"    || exit $?
run_stage "hardneg" "scripts/run_classical_hard_neg.py" || exit $?
run_stage "infer"   "scripts/run_classical_infer.py"    || exit $?

elapsed=$(( $(date +%s) - started_at ))
hours=$((elapsed / 3600))
mins=$(( (elapsed % 3600) / 60 ))
secs=$((elapsed % 60))
echo ""
echo "===== ALL DONE ====="
printf "Total elapsed: %dh %dm %ds\n" "$hours" "$mins" "$secs"
echo "Predictions: reports/predictions/classical_test.json"
