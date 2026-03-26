#!/usr/bin/env bash
# ==========================================================================
#  exp3b_q11_run.sh — Q11: Contrastive Cross-Candidate Features
#
#  Does the verifier improve when it sees where each candidate ranks
#  relative to its peers on per-hop NLI scores?
#
#  No new inference. Reads existing hop_scores. CPU only. ~2 minutes.
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

echo "=== Q11: Contrastive Cross-Candidate Features ==="
echo ""

${PYTHON} ${PROJ_ROOT}/tools/exp3b_q11_contrastive_verifier.py \
    --hop_scores   ${PROJ_ROOT}/exp3b/preds/dev_hop_scores.jsonl \
    --evidence     ${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl \
    --gold         ${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
    --out_json     ${PROJ_ROOT}/exp3b/metrics/q11_contrastive_verifier.json \
    --out_preds    ${PROJ_ROOT}/exp3b/preds/dev_q11_contrastive_preds.jsonl \
    --log          ${PROJ_ROOT}/exp3b/logs/q11_contrastive.log \
    --n_folds 5 --seed 42

echo ""
echo "Done. Results: ${PROJ_ROOT}/exp3b/metrics/q11_contrastive_verifier.json"