#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseA/A1.1/run_qa_hop_score.sh
#  Phase A1.1 — QA Cross-Encoder Hop Scoring
#
#  Scores every candidate in exp0c against hop1, hop2, and flat context
#  using deepset/deberta-v3-base-squad2 (SQuAD 2.0).
#
#  Output features per candidate:
#    qa_hop1, qa_hop2, qa_flat, qa_hop_balance, qa_min_hop
#
#  PREREQ: Run download_deberta_squad2.sh first
#  Estimated runtime: ~90 min on A100
#
#  Usage:
#    bash exp_phaseA/A1.1/run_qa_hop_score.sh
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Inputs — exp0c is the 0.4666 EM baseline (K=200 + high-T fallback)
EVIDENCE="${PROJ_ROOT}/exp0c/evidence/dev_K200_chains.jsonl"
CANDIDATES="${PROJ_ROOT}/exp0c/candidates/dev_M5_7b_K200.jsonl"
MODEL="/var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2"

# Outputs live inside this experiment directory
OUT_DIR="${PROJ_ROOT}/exp_phaseA/A1.1/qa_scores"
OUT_JSONL="${OUT_DIR}/dev_qa_hop_scores.jsonl"
OUT_JSON="${OUT_DIR}/summary.json"
LOG="${OUT_DIR}/run.log"

echo "================================================================"
echo "  Phase A1.1 — QA Cross-Encoder Hop Scoring"
echo "================================================================"
echo "  Evidence:   ${EVIDENCE}"
echo "  Candidates: ${CANDIDATES}"
echo "  Model:      ${MODEL}"
echo "  Output:     ${OUT_JSONL}"
echo "================================================================"
echo ""

mkdir -p "${OUT_DIR}"

# Verify prereqs
if [[ ! -f "${EVIDENCE}" ]]; then
    echo "ERROR: evidence file not found: ${EVIDENCE}" >&2; exit 1
fi
if [[ ! -f "${CANDIDATES}" ]]; then
    echo "ERROR: candidates file not found: ${CANDIDATES}" >&2; exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
    echo "ERROR: model not found: ${MODEL}" >&2
    echo "       Run download_deberta_squad2.sh first." >&2; exit 1
fi

${PYTHON} "${TOOLS_DIR}/exp_a1_qa_hop_score.py" \
    --evidence   "${EVIDENCE}" \
    --candidates "${CANDIDATES}" \
    --model      "${MODEL}" \
    --out_jsonl  "${OUT_JSONL}" \
    --out_json   "${OUT_JSON}" \
    --batch_size 32 \
    2>&1 | tee "${LOG}"

echo ""
echo "================================================================"
echo "  A1.1 complete."
echo "  Scores:  ${OUT_JSONL}"
echo "  Summary: ${OUT_JSON}"
echo "  Log:     ${LOG}"
echo "  Next:    run exp_phaseA/A1.2/run_lex_features.sh (CPU, ~10 min)"
echo "================================================================"