#!/usr/bin/env bash
# ==========================================================================
# run_q11v2.sh — Corrected Q11 on Stage-2 pool, both 7B and 1.5B settings
#
# CPU-only. ~5-7 minutes total (2-3 min per setting).
# Reads precomputed hop/QA/lex scores. No GPU, no model inference.
#
# Usage:
#   cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#   bash run_q11v2.sh
# ==========================================================================

set -euo pipefail

PROJ="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

# Fall back to base python if the project env isn't found
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(which python3)"
    echo "WARN: project conda env not found, using ${PYTHON}"
fi

SCRIPT="${PROJ}/tools/exp_q11v2_contrastive_stage2.py"
GOLD="${PROJ}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

OUT_DIR="${PROJ}/exp_q11v2"
mkdir -p "${OUT_DIR}/logs"

# Sanity check the script is in place
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: script not found at ${SCRIPT}"
    echo "Place exp_q11v2_contrastive_stage2.py in tools/ first."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────
# SETTING 1: HotpotQA MDR 7B M=5 (deployment regime, locked baseline)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo "  SETTING 1/2: HotpotQA MDR 7B M=5"
echo "=========================================================================="

${PYTHON} "${SCRIPT}" \
    --hop_scores    "${PROJ}/exp5b/preds/dev_hop_scores.jsonl" \
    --qa_scores     "${PROJ}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl" \
    --lex_features  "${PROJ}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl" \
    --candidates    "${PROJ}/exp5b/candidates/dev_M5_7b_hightemp.jsonl" \
    --gold          "${GOLD}" \
    --setting_label "hotpotqa_mdr_7b_m5" \
    --out_json      "${OUT_DIR}/hotpotqa_mdr_7b_m5_results.json" \
    --n_folds 5 \
    --seed 42 \
    2>&1 | tee "${OUT_DIR}/logs/hotpotqa_mdr_7b_m5.log"

# ─────────────────────────────────────────────────────────────────────────
# SETTING 2: HotpotQA MDR 1.5B M=5 (noisier-generator regime)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo "  SETTING 2/2: HotpotQA MDR 1.5B M=5"
echo "=========================================================================="

${PYTHON} "${SCRIPT}" \
    --hop_scores    "${PROJ}/exp1b/preds/dev_hop_scores.jsonl" \
    --qa_scores     "${PROJ}/exp1b/preds/dev_qa_hop_scores.jsonl" \
    --lex_features  "${PROJ}/exp1b/preds/dev_lex_features.jsonl" \
    --candidates    "${PROJ}/exp1b/candidates/dev_M5_candidates_qwen.jsonl" \
    --gold          "${GOLD}" \
    --setting_label "hotpotqa_mdr_1p5b_m5" \
    --out_json      "${OUT_DIR}/hotpotqa_mdr_1p5b_m5_results.json" \
    --n_folds 5 \
    --seed 42 \
    2>&1 | tee "${OUT_DIR}/logs/hotpotqa_mdr_1p5b_m5.log"

echo ""
echo "=========================================================================="
echo "  DONE"
echo "=========================================================================="
echo "  Results:"
echo "    ${OUT_DIR}/hotpotqa_mdr_7b_m5_results.json"
echo "    ${OUT_DIR}/hotpotqa_mdr_1p5b_m5_results.json"
echo ""
echo "  Logs:"
echo "    ${OUT_DIR}/logs/"
echo ""
echo "  Paste both result JSONs back to me for interpretation."
echo "=========================================================================="