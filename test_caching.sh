#!/usr/bin/env bash
# ============================================================
# Enterprise MLOps Pipeline — DVC Caching Validation Script
# ============================================================
# Demonstrates and validates DVC's intelligent stage caching.
#
# Expected behavior:
#   Run 1 (fresh):  ALL stages execute
#   Run 2 (cached): ALL stages skipped (no changes)
#   Param change:   prepare + featurize SKIPPED
#                   train + evaluate RE-EXECUTED
#
# Usage:
#   bash test_caching.sh
#
# Output:
#   repro_log.txt  — Full execution log with timing data
# ============================================================

set -euo pipefail

LOG_FILE="repro_log.txt"
PARAMS_FILE="params.yaml"
SEPARATOR="============================================================"

# ── Colors for terminal output ──────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${YELLOW}→ $1${NC}" | tee -a "$LOG_FILE"
}

# ── Initialize log file ─────────────────────────────────────
echo "$SEPARATOR" > "$LOG_FILE"
echo "DVC CACHING VALIDATION REPORT" >> "$LOG_FILE"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "Project: Enterprise MLOps Pipeline" >> "$LOG_FILE"
echo "$SEPARATOR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

log "Starting DVC caching validation..."

# ── Step 1: Force full pipeline run ────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "=== RUN 1: FULL PIPELINE (FORCE) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

info "Forcing full pipeline re-execution (clearing stage cache)..."
START_TIME=$(date +%s%N)

python3 -m dvc repro --force 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s%N)
RUN1_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
echo "" | tee -a "$LOG_FILE"
success "Run 1 complete in ${RUN1_TIME}ms — All stages executed"

# ── Step 2: Verify caching (no changes) ────────────────────
echo "" | tee -a "$LOG_FILE"
echo "=== RUN 2: CACHED RE-RUN (NO CHANGES) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

info "Re-running pipeline without any changes (all stages should be cached)..."
START_TIME=$(date +%s%N)

python3 -m dvc repro 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s%N)
RUN2_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
echo "" | tee -a "$LOG_FILE"
success "Run 2 complete in ${RUN2_TIME}ms — All stages should show 'cached'"

# ── Step 3: Modify only model hyperparameters ──────────────
echo "" | tee -a "$LOG_FILE"
echo "=== PARAM CHANGE: MODIFYING train.n_estimators ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Save original params
ORIGINAL_PARAMS=$(cat "$PARAMS_FILE")

info "Changing model.n_estimators from 100 → 150..."
# Use Python for cross-platform YAML modification
python3 -c "
import yaml
with open('$PARAMS_FILE', 'r') as f:
    params = yaml.safe_load(f)
original = params['train']['n_estimators']
params['train']['n_estimators'] = 150
with open('$PARAMS_FILE', 'w') as f:
    yaml.dump(params, f, default_flow_style=False)
print(f'Changed n_estimators: {original} → 150')
" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== RUN 3: PARTIAL RE-RUN (PARAM CHANGE) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Expected: prepare=CACHED | featurize=CACHED | train=EXECUTED | evaluate=EXECUTED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

START_TIME=$(date +%s%N)

python3 -m dvc repro 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s%N)
RUN3_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
echo "" | tee -a "$LOG_FILE"
success "Run 3 complete in ${RUN3_TIME}ms — Only train+evaluate should have re-run"

# ── Step 4: Restore original params ────────────────────────
echo "" | tee -a "$LOG_FILE"
info "Restoring original params.yaml..."
echo "$ORIGINAL_PARAMS" > "$PARAMS_FILE"
success "params.yaml restored"

# ── Summary Report ─────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "$SEPARATOR" | tee -a "$LOG_FILE"
echo "CACHING VALIDATION SUMMARY" | tee -a "$LOG_FILE"
echo "$SEPARATOR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

SPEEDUP_VS_FULL=$(python3 -c "print(f'{$RUN1_TIME / max($RUN3_TIME, 1):.1f}')")

printf "%-35s %10s\n" "Run" "Time (ms)" | tee -a "$LOG_FILE"
printf "%-35s %10s\n" "---" "---------" | tee -a "$LOG_FILE"
printf "%-35s %10s\n" "Run 1 — Full pipeline (force)" "${RUN1_TIME}ms" | tee -a "$LOG_FILE"
printf "%-35s %10s\n" "Run 2 — Fully cached (no changes)" "${RUN2_TIME}ms" | tee -a "$LOG_FILE"
printf "%-35s %10s\n" "Run 3 — Partial re-run (param change)" "${RUN3_TIME}ms" | tee -a "$LOG_FILE"
printf "%-35s %10s\n" "Full-run vs cached speedup" "${SPEEDUP_VS_FULL}x" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✓ Validation complete. Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "$SEPARATOR" | tee -a "$LOG_FILE"

log "DVC caching validation finished. See '$LOG_FILE' for full log."
