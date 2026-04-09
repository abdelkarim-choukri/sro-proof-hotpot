#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseB/B1.1/run_stage1_filter.sh
#  Phase B1.1 — Stage 1 Garbage Filter
#
#  Logistic regression filter that removes is_bad/is_unknown candidates.
#  CRITICAL: validates that 0 correct answers are removed before proceeding.
#  Estimated runtime: ~1 min (CPU only)
#
#  Usage:
#    bash exp_phaseB/B1.1/run_stage1_filter.sh
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

HOP_SCORES="${PROJ_ROOT}/exp0c/preds/dev_hop_scores.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

OUT_DIR="${PROJ_ROOT}/exp_phaseB/B1.1/filter_output"
OUT_JSONL="${OUT_DIR}/dev_stage1_filtered.jsonl"
OUT_JSON="${OUT_DIR}/summary.json"
LOG="${OUT_DIR}/run.log"

echo "================================================================"
echo "  Phase B1.1 — Stage 1 Garbage Filter"
echo "================================================================"
echo ""

mkdir -p "${OUT_DIR}"

for f in "${HOP_SCORES}" "${GOLD}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file not found: ${f}" >&2; exit 1
    fi
done

${PYTHON} "${TOOLS_DIR}/exp_b1_stage1_filter.py" \
    --hop_scores "${HOP_SCORES}" \
    --gold       "${GOLD}" \
    --out_jsonl  "${OUT_JSONL}" \
    --out_json   "${OUT_JSON}" \
    2>&1 | tee "${LOG}"

echo ""
echo "================================================================"
echo "  B1.1 complete."
echo "  Filter: ${OUT_JSONL}"
echo "  Summary: ${OUT_JSON}"
echo "  Log:     ${LOG}"
echo "  BEFORE PROCEEDING: verify false_negatives=0 in summary"
echo "  Next: bash exp_phaseB/B2.1/run_stage2_verifier.sh"
echo "================================================================"