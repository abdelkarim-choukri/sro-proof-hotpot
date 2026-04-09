#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseA/A2.1/run_principle_table.sh
#  Phase A2.1 — Flat vs Chain-Aware Ablation Table
#
#  Trains 6 XGBoost models (3 scorers × flat|chain-aware).
#  Produces the principle table for the paper.
#  Estimated runtime: ~30 min (6 × 5-fold CV)
#
#  Usage:
#    bash exp_phaseA/A2.1/run_principle_table.sh
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

HOP_SCORES="${PROJ_ROOT}/exp0c/preds/dev_hop_scores.jsonl"
QA_SCORES="${PROJ_ROOT}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl"
LEX_SCORES="${PROJ_ROOT}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl"
EVIDENCE="${PROJ_ROOT}/exp0c/evidence/dev_K200_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

OUT_DIR="${PROJ_ROOT}/exp_phaseA/A2.1"
OUT_JSON="${OUT_DIR}/principle_table.json"
LOG="${OUT_DIR}/principle_table.log"

echo "================================================================"
echo "  Phase A2.1 — Flat vs Chain-Aware Ablation Table"
echo "  6 models: NLI, QA, Lexical × (flat | chain-aware)"
echo "================================================================"
echo ""

mkdir -p "${OUT_DIR}"

for f in "${HOP_SCORES}" "${QA_SCORES}" "${LEX_SCORES}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file not found: ${f}" >&2; exit 1
    fi
done

${PYTHON} "${TOOLS_DIR}/exp_a2_principle_table.py" \
    --hop_scores  "${HOP_SCORES}" \
    --qa_scores   "${QA_SCORES}" \
    --lex_scores  "${LEX_SCORES}" \
    --evidence    "${EVIDENCE}" \
    --gold        "${GOLD}" \
    --out_json    "${OUT_JSON}" \
    --log         "${LOG}"

echo ""
echo "================================================================"
echo "  A2.1 complete."
echo "  Results: ${OUT_JSON}"
echo "  Log:     ${LOG}"
echo "================================================================"