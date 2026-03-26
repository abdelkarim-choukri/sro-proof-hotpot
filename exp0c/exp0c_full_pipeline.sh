#!/usr/bin/env bash
# ==========================================================================
#  exp0c_full_pipeline.sh — Full Pipeline on K=200 Evidence
#
#  Step 1: Generate M=5 with 7B + high-temp fallback (GPU via vLLM)
#  Step 2: Oracle@5
#  Step 3: NLI scoring (kill vLLM first)
#  Step 4: Hop-level NLI
#  Step 5: XGBoost verifier
#  Step 6: Comparison vs K=100 (Exp4 + Exp5b)
#
#  PREREQ: vLLM serving 7B on port 8000
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

EVIDENCE_K200="${PROJ_ROOT}/exp0c/evidence/dev_K200_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

EXP="${PROJ_ROOT}/exp0c"
CANDIDATES="${EXP}/candidates/dev_M5_7b_K200.jsonl"

echo "================================================================"
echo "  K=200 Full Pipeline (7B + high-temp fallback)"
echo "  Feasible: 6543/7405 (was 6452 at K=100)"
echo "================================================================"

mkdir -p "${EXP}/candidates" "${EXP}/preds" "${EXP}/metrics" "${EXP}/logs"

# ── Step 1: Generate M=5 ──
echo ""
echo "━━━ Step 1: Generate M=5 (7B + T=1.0 fallback) ━━━"

if [[ -f "${CANDIDATES}" ]] && [[ $(wc -l < "${CANDIDATES}") -ge 7405 ]]; then
    echo "  ✓ Already complete ($(wc -l < "${CANDIDATES}") lines)"
else
    echo "  vLLM must be serving 7B on port 8000"
    ${PYTHON} ${TOOLS_DIR}/exp5_generate_candidates.py \
        --evidence      "${EVIDENCE_K200}" \
        --gold          "${GOLD}" \
        --prompt_file   "${PROMPT_V2}" \
        --prompt_mode   flat \
        --out_jsonl     "${CANDIDATES}" \
        --manifest      "${EXP}/manifest.json" \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  "${MODEL_ID}" \
        --split dev --m 5 --seed 12345 \
        --temperature 0.7 \
        --fallback_temperature 1.0 \
        --resume \
        2>&1 | tee "${EXP}/logs/generate.log"
fi
echo "  ✓ Candidates: ${CANDIDATES}"

# ── Step 2: Oracle@5 ──
echo ""
echo "━━━ Step 2: Oracle@5 ━━━"

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence "${EVIDENCE_K200}" --candidates "${CANDIDATES}" --gold "${GOLD}" \
    --split dev --m 5 \
    --out_json "${EXP}/metrics/oracle_M5.json" \
    --out_jsonl "${EXP}/metrics/oracle_M5_perqid.jsonl" \
    --out_sha256 "${EXP}/metrics/oracle_M5.sha256" \
    --manifest "${EXP}/manifest.json" \
    2>&1 | tee "${EXP}/logs/oracle.log"

echo ""
echo "  ★ Kill vLLM now before NLI scoring (pkill -f vllm.entrypoints)"
echo "  Press Enter when GPU is free..."
read -r

# ── Step 3: Flat NLI ──
echo ""
echo "━━━ Step 3: Flat NLI Scoring ━━━"

NLI_PREDS="${EXP}/preds/dev_nli_preds.jsonl"

if [[ -f "${NLI_PREDS}" ]] && [[ $(wc -l < "${NLI_PREDS}") -ge 7405 ]]; then
    echo "  ✓ Already complete"
else
    ${PYTHON} ${TOOLS_DIR}/exp1_nli_baseline.py \
        --candidates "${CANDIDATES}" --evidence "${EVIDENCE_K200}" --gold "${GOLD}" \
        --model "${NLI_MODEL}" \
        --out_metrics "${EXP}/metrics/nli_baseline.json" \
        --out_preds "${NLI_PREDS}" \
        --batch_size 64 \
        2>&1 | tee "${EXP}/logs/nli.log"
fi

# ── Step 4: Hop-Level NLI ──
echo ""
echo "━━━ Step 4: Hop-Level NLI ━━━"

HOP_SCORES="${EXP}/preds/dev_hop_scores.jsonl"

if [[ -f "${HOP_SCORES}" ]] && [[ $(wc -l < "${HOP_SCORES}") -ge 7405 ]]; then
    echo "  ✓ Already complete"
else
    ${PYTHON} ${TOOLS_DIR}/exp2_q1_signal_independence.py \
        --candidates "${CANDIDATES}" --nli_preds "${NLI_PREDS}" \
        --evidence "${EVIDENCE_K200}" --gold "${GOLD}" \
        --model "${NLI_MODEL}" \
        --out_hop_scores "${HOP_SCORES}" \
        --out_json "${EXP}/metrics/q1_signal.json" \
        --out_jsonl "${EXP}/metrics/q1_signal_perqid.jsonl" \
        --log "${EXP}/logs/q1.log" --batch_size 64 \
        2>&1 | tee "${EXP}/logs/q1_stdout.log"
fi

# ── Step 5: XGBoost ──
echo ""
echo "━━━ Step 5: Chain-Aware XGBoost ━━━"

CHAIN_JSON="${EXP}/metrics/q2q3q4_chain_verifier.json"

${PYTHON} ${TOOLS_DIR}/exp2_q2q3q4_chain_verifier.py \
    --hop_scores "${HOP_SCORES}" --evidence "${EVIDENCE_K200}" --gold "${GOLD}" \
    --out_json "${CHAIN_JSON}" \
    --out_preds_min "${EXP}/preds/dev_chain_verifier_min_preds.jsonl" \
    --out_preds_mean "${EXP}/preds/dev_chain_verifier_mean_preds.jsonl" \
    --log "${EXP}/logs/q2q3q4.log" --n_folds 5 --seed 42 \
    2>&1 | tee "${EXP}/logs/q2q3q4_stdout.log"

# ── Step 6: Comparison ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: K=200 vs K=100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# K=200 results
o200 = json.load(open('${EXP}/metrics/oracle_M5.json'))
n200 = json.load(open('${EXP}/metrics/nli_baseline.json'))
c200 = json.load(open('${CHAIN_JSON}'))
best200 = c200['mean_pooling'] if c200['mean_pooling']['overall']['em'] >= c200['min_pooling']['overall']['em'] else c200['min_pooling']

# Baselines
configs = [
    ('1.5B M=5 K=100 (Exp1b)',   0.3982, 0.3011, 0.3409, 6452),
    ('1.5B M=10 K=100 (Exp3b)',  0.4344, 0.2925, 0.3622, 6452),
    ('7B M=5 K=100 (Exp4)',      0.4879, 0.4536, 0.4609, 6452),
    ('7B M=5 K=100 +hT (Exp5b)', 0.5025, 0.4575, 0.4658, 6452),
]

o_em = o200['overall']['oracle_em']
n_em = n200['overall']['nli_em']
x_em = best200['overall']['em']
x_f1 = best200['overall']['f1']
feas = o200.get('subset_docrecall1',{}).get('oracle_em', 0)

W = 80
print()
print('=' * W)
print(f'  {\"Config\":<40} {\"Feasible\":>8} {\"Oracle\":>8} {\"NLI\":>8} {\"XGB\":>8}')
print('-' * W)
for label, o, n, x, fe in configs:
    print(f'  {label:<40} {fe:>8} {o:>8.4f} {n:>8.4f} {x:>8.4f}')
print(f'  {\"7B M=5 K=200 +hT (Exp0c) *\":<40} {6543:>8} {o_em:>8.4f} {n_em:>8.4f} {x_em:>8.4f}')
print('-' * W)
print(f'  {\"Delta K=200 vs K=100+hT\":<40} {\" +91\":>8} {o_em-0.5025:>+8.4f} {n_em-0.4575:>+8.4f} {x_em-0.4658:>+8.4f}')
print('=' * W)

print(f'\n  Feasible oracle: {feas:.4f}')
print(f'  Bridge EM:  {best200[\"bridge\"][\"em\"]:.4f}')
print(f'  Comp EM:    {best200[\"comparison\"][\"em\"]:.4f}')
print(f'  Feas EM:    {best200[\"feasible\"][\"em\"]:.4f}')

gap = o_em - x_em
print(f'\n  Verifier gap: {gap:.4f}')
print(f'  Bucket A: 953 -> 862 (-91 questions)')
print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  K=200 pipeline complete. Results in: ${EXP}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"