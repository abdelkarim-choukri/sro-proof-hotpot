#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

echo "=== Hybrid XGB + LLM Verifier (Threshold Sweep) ==="
echo ""

${PYTHON} ${PROJ_ROOT}/tools/exp4_hybrid_verifier.py \
    --pilot         ${PROJ_ROOT}/exp4_7b/pilot/pilot_questions.jsonl \
    --out_dir       ${PROJ_ROOT}/exp4_7b/pilot/hybrid \
    --llm_base_url  http://127.0.0.1:8000/v1 \
    --llm_model_id  "${MODEL_ID}" \
    --temperature   0.0 \
    --max_new_tokens 512

echo ""
echo "Done."