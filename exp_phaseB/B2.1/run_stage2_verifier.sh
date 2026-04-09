#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseB/B2.1/run_stage2_verifier.sh
#  Phase B2.1 — Stage 2 Chain-Aware Verifier
#
#  XGBoost trained ONLY on candidates that survived Stage 1.
#  19 features: 10 surface (no is_bad/is_unknown) + 3 NLI + 3 QA + 3 lex
#  Estimated runtime: ~10 min (1 model × 5-fold CV)
#
#  PREREQ: exp_phaseB/B1.1/run_stage1_filter.sh must have run first
#          AND summary.json must show false_negatives=0
#
#  Usage:
#    bash exp_phaseB/B2.1/run_stage2_verifier.sh
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

HOP_SCORES="${PROJ_ROOT}/exp0c/preds/dev_hop_scores.jsonl"
QA_SCORES="${PROJ_ROOT}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl"
LEX_SCORES="${PROJ_ROOT}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl"
STAGE1="${PROJ_ROOT}/exp_phaseB/B1.1/filter_output/dev_stage1_filtered.jsonl"
EVIDENCE="${PROJ_ROOT}/exp0c/evidence/dev_K200_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

OUT_DIR="${PROJ_ROOT}/exp_phaseB/B2.1/results"
OUT_JSON="${OUT_DIR}/stage2_verifier.json"
OUT_PREDS="${OUT_DIR}/dev_stage2_preds.jsonl"
LOG="${OUT_DIR}/stage2_verifier.log"

echo "================================================================"
echo "  Phase B2.1 — Stage 2 Chain-Aware Verifier"
echo "  19 features: surface (no is_bad) + NLI + QA + lex chain"
echo "  Baseline to beat: 0.4666 EM"
echo "================================================================"
echo ""

mkdir -p "${OUT_DIR}"

# Verify Stage 1 ran and was clean
if [[ ! -f "${STAGE1}" ]]; then
    echo "ERROR: Stage 1 filter output not found: ${STAGE1}" >&2
    echo "       Run exp_phaseB/B1.1/run_stage1_filter.sh first." >&2
    exit 1
fi

FN=$(python3 -c "
import json
s = json.load(open('${PROJ_ROOT}/exp_phaseB/B1.1/filter_output/summary.json'))
print(s['false_negatives'])
")
if [[ "${FN}" != "0" ]]; then
    echo "ERROR: Stage 1 has false_negatives=${FN} (correct answers removed)." >&2
    echo "       Do not run Stage 2 until Stage 1 is clean." >&2
    exit 1
fi
echo "  Stage 1 safety check: false_negatives=0 ✓"
echo ""

for f in "${HOP_SCORES}" "${QA_SCORES}" "${LEX_SCORES}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file not found: ${f}" >&2; exit 1
    fi
done

${PYTHON} "${TOOLS_DIR}/exp_b2_stage2_verifier.py" \
    --hop_scores    "${HOP_SCORES}" \
    --qa_scores     "${QA_SCORES}" \
    --lex_scores    "${LEX_SCORES}" \
    --stage1_filter "${STAGE1}" \
    --evidence      "${EVIDENCE}" \
    --gold          "${GOLD}" \
    --out_json      "${OUT_JSON}" \
    --out_preds     "${OUT_PREDS}" \
    --log           "${LOG}"

echo ""
echo "================================================================"
echo "  B2.1 complete."
echo "  Results: ${OUT_JSON}"
echo "  Preds:   ${OUT_PREDS}"
echo "  Log:     ${LOG}"
echo "================================================================"