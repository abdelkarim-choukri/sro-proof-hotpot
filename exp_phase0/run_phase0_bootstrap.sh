#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  run_phase0_bootstrap.sh — Bootstrap Significance Tests
# ══════════════════════════════════════════════════════════════════════
#
#  WHAT THIS DOES:
#    Tests whether the Phase 0 ablation differences are statistically
#    significant using paired bootstrap resampling (B=10,000).
#
#  KEY TESTS:
#    1. Z_full vs Z2 (chain marginal) — is +0.77pp significant?
#    2. Z1 vs Z2 (majority vs surface) — is -0.20pp significant?
#    3. Z3 vs Monolithic — chain-only ≈ monolithic?
#    4. Z_full vs Monolithic — two-stage beats monolithic?
#
#  COMPUTE:
#    CPU only, ~1-2 minutes for 10,000 bootstrap samples.
#
#  USAGE:
#    bash exp_phase0/run_phase0_bootstrap.sh
#
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
SCRIPT="${PROJ_ROOT}/tools/phase0_bootstrap.py"
PREDS_DIR="${PROJ_ROOT}/exp_phase0/results"
OUT_DIR="${PROJ_ROOT}/exp_phase0/results"

GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
MONO_PREDS="${PROJ_ROOT}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  BOOTSTRAP SIGNIFICANCE TESTS"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight checks ──
echo "Checking files ..."
for F in "$GOLD" "$MONO_PREDS"; do
    if [[ ! -f "$F" ]]; then
        echo "  ✗ MISSING: $F"
        exit 1
    fi
done

for TAG in z1_majority z2_surface z3_chain z_full; do
    F="${PREDS_DIR}/${TAG}_preds.jsonl"
    if [[ ! -f "$F" ]]; then
        echo "  ✗ MISSING: $F"
        echo "  Run Phase 0 ablations first!"
        exit 1
    else
        echo "  ✓ ${TAG}_preds.jsonl"
    fi
done

if [[ ! -f "$SCRIPT" ]]; then
    echo "Script not found at ${SCRIPT}"
    echo "Please copy phase0_bootstrap.py to ${PROJ_ROOT}/tools/"
    exit 1
fi

# ── Run bootstrap ──
echo ""
echo "Running bootstrap (B=10000, ~1-2 min) ..."
echo ""

${PYTHON} "${SCRIPT}" \
    --preds_dir     "${PREDS_DIR}" \
    --gold          "${GOLD}" \
    --mono_preds    "${MONO_PREDS}" \
    --out_dir       "${OUT_DIR}" \
    --n_bootstrap   10000 \
    --seed          42

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Report:  ${OUT_DIR}/bootstrap_report.txt"
echo "Results: ${OUT_DIR}/bootstrap_results.json"
echo ""