#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  run_phase0_ablations.sh — The Decisive Ablations
# ══════════════════════════════════════════════════════════════════════
#
#  WHAT THIS DOES:
#    Runs four systems on the same Stage-1-filtered candidate pool
#    to isolate what actually drives the +0.42pp gain:
#
#    Z1  = majority vote on filtered pool (self-consistency baseline)
#    Z2  = XGBoost with surface features only (no chain features)
#    Z3  = XGBoost with chain features only (no answer_freq etc.)
#    Z_full = XGBoost with all 19 features (should reproduce 0.4708)
#
#  HOW TO READ THE RESULTS:
#    If Z_full - Z2 ≥ 0.002:  chain features matter → paper Path A
#    If Z_full - Z2 < 0.001:  chain features ornamental → paper Path B
#    If Z1 ≈ Z2:              XGBoost adds nothing over majority voting
#    If Z3 << Z2:             chain features can't stand alone
#
#  COMPUTE:
#    CPU only. ~10-20 minutes total (5-fold CV × 4 ablations).
#    No GPU needed.
#
#  USAGE:
#    # First, do a dry run to verify file schemas:
#    bash exp_phase0/run_phase0_ablations.sh --inspect
#
#    # Then run for real:
#    bash exp_phase0/run_phase0_ablations.sh
#
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──
PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
SCRIPT="${PROJ_ROOT}/tools/phase0_ablations.py"
OUT_DIR="${PROJ_ROOT}/exp_phase0/results"

# ── Input files ──
# These are the defaults baked into the Python script.
# Override with --candidates, --hop_scores, etc. if your paths differ.
CANDIDATES="${PROJ_ROOT}/exp0c/candidates/dev_M5_7b_K200.jsonl"
HOP_SCORES="${PROJ_ROOT}/exp0c/preds/dev_hop_scores.jsonl"
QA_SCORES="${PROJ_ROOT}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl"
LEX_FEATURES="${PROJ_ROOT}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl"
MONO_PREDS="${PROJ_ROOT}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

# ── Pre-flight checks ──
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  PHASE 0: THE DECISIVE ABLATIONS"
echo "══════════════════════════════════════════════════════════════"
echo ""

echo "Checking files ..."
MISSING=0
for F in "$CANDIDATES" "$HOP_SCORES" "$QA_SCORES" "$LEX_FEATURES" "$MONO_PREDS" "$GOLD"; do
    if [[ ! -f "$F" ]]; then
        echo "  ✗ MISSING: $F"
        MISSING=1
    else
        echo "  ✓ $(basename $F)"
    fi
done

if [[ "$MISSING" -eq 1 ]]; then
    echo ""
    echo "ERROR: Some input files are missing."
    echo "Check the paths at the top of this script and adjust."
    echo ""
    echo "Common fixes:"
    echo "  - If qa_scores path differs: edit QA_SCORES variable above"
    echo "  - If lex_features path differs: edit LEX_FEATURES variable above"
    echo "  - Run with --inspect to see what the script expects"
    exit 1
fi

# ── Check if script exists; if not, copy it ──
if [[ ! -f "$SCRIPT" ]]; then
    echo ""
    echo "Script not found at ${SCRIPT}"
    echo "Please copy phase0_ablations.py to ${PROJ_ROOT}/tools/"
    exit 1
fi

# ── Handle --inspect flag ──
if [[ "${1:-}" == "--inspect" ]]; then
    echo ""
    echo "Running schema inspection only ..."
    echo ""
    ${PYTHON} "${SCRIPT}" \
        --proj_root "${PROJ_ROOT}" \
        --out_dir   "${OUT_DIR}" \
        --inspect_only
    echo ""
    echo "Schema inspection complete."
    echo "If the field names look correct, run without --inspect."
    exit 0
fi

# ── Create output directory ──
mkdir -p "${OUT_DIR}"

# ── Run the ablations ──
echo ""
echo "Starting ablations (CPU only, ~10-20 min) ..."
echo "Log: ${OUT_DIR}/phase0.log"
echo ""

${PYTHON} "${SCRIPT}" \
    --proj_root     "${PROJ_ROOT}" \
    --out_dir       "${OUT_DIR}" \
    --candidates    "${CANDIDATES}" \
    --hop_scores    "${HOP_SCORES}" \
    --qa_scores     "${QA_SCORES}" \
    --lex_features  "${LEX_FEATURES}" \
    --mono_preds    "${MONO_PREDS}" \
    --gold          "${GOLD}" \
    --n_folds 5 \
    --seed 42

# ── Post-run summary ──
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Results:      ${OUT_DIR}/phase0_results.json"
echo "Log:          ${OUT_DIR}/phase0.log"
echo "Predictions:  ${OUT_DIR}/z*_preds.jsonl"
echo ""
echo "Next steps:"
echo "  1. Read the INTERPRETATION section in the log"
echo "  2. If chain_marginal ≥ 0.2pp → chain-aware claim survives"
echo "  3. If chain_marginal < 0.1pp → reframe paper around diagnostics"
echo ""