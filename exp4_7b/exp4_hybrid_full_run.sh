#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

echo "================================================================"
echo "  Full-Scale Hybrid Verifier (τ=0.55, all 7405 questions)"
echo "================================================================"
echo ""
echo "  vLLM must be serving 7B on port 8000"
echo ""

${PYTHON} ${PROJ_ROOT}/tools/exp4_hybrid_full.py \
    --chain_preds   ${PROJ_ROOT}/exp4_7b/preds/dev_chain_verifier_mean_preds.jsonl \
    --hop_scores    ${PROJ_ROOT}/exp4_7b/preds/dev_hop_scores.jsonl \
    --evidence      ${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl \
    --gold          ${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
    --oracle_perqid ${PROJ_ROOT}/exp4_7b/metrics/oracle_M5_dev_perqid.jsonl \
    --out_dir       ${PROJ_ROOT}/exp4_7b/hybrid_full \
    --threshold     0.55 \
    --llm_base_url  http://127.0.0.1:8000/v1 \
    --llm_model_id  "${MODEL_ID}" \
    --temperature   0.0 \
    --max_new_tokens 512

echo ""
echo "Done."