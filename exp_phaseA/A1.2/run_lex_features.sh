#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseA/A1.2/run_lex_features.sh
#  Phase A1.2 — Lexical Grounding Features
#
#  Pure string-matching features. No model, no GPU, CPU only.
#  Estimated runtime: ~5 min
#
#  Usage:
#    bash exp_phaseA/A1.2/run_lex_features.sh
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

EVIDENCE="${PROJ_ROOT}/exp0c/evidence/dev_K200_chains.jsonl"
CANDIDATES="${PROJ_ROOT}/exp0c/candidates/dev_M5_7b_K200.jsonl"

OUT_DIR="${PROJ_ROOT}/exp_phaseA/A1.2/lex_features"
OUT_JSONL="${OUT_DIR}/dev_lex_features.jsonl"
OUT_JSON="${OUT_DIR}/summary.json"
LOG="${OUT_DIR}/run.log"

echo "================================================================"
echo "  Phase A1.2 — Lexical Grounding Features (CPU only)"
echo "================================================================"
echo "  Evidence:   ${EVIDENCE}"
echo "  Candidates: ${CANDIDATES}"
echo "  Output:     ${OUT_JSONL}"
echo "================================================================"
echo ""

mkdir -p "${OUT_DIR}"

if [[ ! -f "${EVIDENCE}" ]]; then
    echo "ERROR: evidence file not found: ${EVIDENCE}" >&2; exit 1
fi
if [[ ! -f "${CANDIDATES}" ]]; then
    echo "ERROR: candidates file not found: ${CANDIDATES}" >&2; exit 1
fi

${PYTHON} "${TOOLS_DIR}/exp_a1_lex_features.py" \
    --evidence   "${EVIDENCE}" \
    --candidates "${CANDIDATES}" \
    --out_jsonl  "${OUT_JSONL}" \
    --out_json   "${OUT_JSON}" \
    2>&1 | tee "${LOG}"

echo ""
echo "================================================================"
echo "  A1.2 complete."
echo "  Features: ${OUT_JSONL}"
echo "  Summary:  ${OUT_JSON}"
echo "  Log:      ${LOG}"
echo "  Next:     A2.1 flat vs chain-aware ablation table"
echo "================================================================"