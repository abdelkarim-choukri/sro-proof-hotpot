#!/usr/bin/env bash
# ==========================================================================
#  exp4_7b_full_pipeline.sh — Full 7B Generator + Verification Pipeline
#
#  Runs the complete stack with Qwen2.5-7B-Instruct as generator:
#    Step 1: Generate M=5 candidates (GPU via vLLM, ~2-3 hrs)
#    Step 2: Compute oracle@5
#    Step 3: Flat NLI scoring (GPU, ~15 min)
#    Step 4: Hop-level NLI scoring (GPU, ~10 min)
#    Step 5: Chain-aware XGBoost verifier (CPU, ~2 min)
#    Step 6: Full comparison vs 1.5B baselines
#
#  PREREQ: vLLM serving 7B model on port 8000 (tensor-parallel-size 2)
#
#  Estimated total runtime: ~3 hours
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Shared inputs
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"

# Exp4 directory
EXP4="${PROJ_ROOT}/exp4_7b"
CANDIDATES="${EXP4}/candidates/dev_M5_candidates_7b.jsonl"
MANIFEST="${EXP4}/manifest.json"

echo "================================================================"
echo "  Exp4: Full 7B Generator Pipeline (M=5, all 7405 questions)"
echo "================================================================"
echo ""

for f in "${EVIDENCE}" "${GOLD}" "${PROMPT_V2}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f"
        exit 1
    fi
done

mkdir -p "${EXP4}/candidates" "${EXP4}/preds" "${EXP4}/metrics" "${EXP4}/logs"

# ==========================================================================
#  Step 1: Generate M=5 candidates with 7B
# ==========================================================================
echo "━━━ Step 1: Generate M=5 candidates (Qwen2.5-7B-Instruct) ━━━"
echo "  vLLM must be serving 7B on port 8000"
echo ""

if [[ -f "${CANDIDATES}" ]] && [[ $(wc -l < "${CANDIDATES}") -ge 7405 ]]; then
    echo "  ✓ Already complete ($(wc -l < "${CANDIDATES}") lines) — skipping"
else
    N_DONE=0
    if [[ -f "${CANDIDATES}" ]]; then
        N_DONE=$(wc -l < "${CANDIDATES}")
        echo "  Resuming from ${N_DONE} already done"
    fi

    ${PYTHON} ${TOOLS_DIR}/exp3_generate_candidates.py \
        --evidence        "${EVIDENCE}" \
        --gold            "${GOLD}" \
        --prompt_file     "${PROMPT_V2}" \
        --prompt_version  v2 \
        --evidence_format flat \
        --out_jsonl       "${CANDIDATES}" \
        --manifest        "${MANIFEST}" \
        --llm_base_url    http://127.0.0.1:8000/v1 \
        --llm_model_id    /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
        --split dev \
        --m 5 \
        --seed 12345 \
        --resume \
        2>&1 | tee "${EXP4}/logs/generate_7b.log"
fi

echo ""
echo "  ✓ Candidates: ${CANDIDATES}"
echo ""

# ==========================================================================
#  Step 2: Oracle@5
# ==========================================================================
echo "━━━ Step 2: Compute Oracle@5 ━━━"

ORACLE_JSON="${EXP4}/metrics/oracle_M5_dev.json"
ORACLE_JSONL="${EXP4}/metrics/oracle_M5_dev_perqid.jsonl"

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence    "${EVIDENCE}" \
    --candidates  "${CANDIDATES}" \
    --gold        "${GOLD}" \
    --split dev --m 5 \
    --out_json    "${ORACLE_JSON}" \
    --out_jsonl   "${ORACLE_JSONL}" \
    --out_sha256  "${EXP4}/metrics/oracle_M5_dev.sha256" \
    --manifest    "${MANIFEST}" \
    2>&1 | tee "${EXP4}/logs/oracle.log"

echo ""
echo "  ✓ Oracle: ${ORACLE_JSON}"
echo ""

# ==========================================================================
#  Step 3: Flat NLI scoring
#  NOTE: vLLM can be stopped now — NLI uses transformers directly
# ==========================================================================
echo "━━━ Step 3: Flat NLI Scoring ━━━"
echo "  (You can stop vLLM now — NLI runs on GPU via transformers)"
echo ""

NLI_PREDS="${EXP4}/preds/dev_nli_preds.jsonl"

if [[ -f "${NLI_PREDS}" ]] && [[ $(wc -l < "${NLI_PREDS}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp1_nli_baseline.py \
        --candidates  "${CANDIDATES}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${GOLD}" \
        --model       "${NLI_MODEL}" \
        --out_metrics "${EXP4}/metrics/nli_baseline.json" \
        --out_preds   "${NLI_PREDS}" \
        --batch_size  64 \
        2>&1 | tee "${EXP4}/logs/nli_baseline.log"
fi

echo "  ✓ NLI preds: ${NLI_PREDS}"
echo ""

# ==========================================================================
#  Step 4: Hop-level NLI scoring
# ==========================================================================
echo "━━━ Step 4: Hop-Level NLI Scoring ━━━"

HOP_SCORES="${EXP4}/preds/dev_hop_scores.jsonl"

if [[ -f "${HOP_SCORES}" ]] && [[ $(wc -l < "${HOP_SCORES}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp2_q1_signal_independence.py \
        --candidates    "${CANDIDATES}" \
        --nli_preds     "${NLI_PREDS}" \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --model         "${NLI_MODEL}" \
        --out_hop_scores "${HOP_SCORES}" \
        --out_json      "${EXP4}/metrics/q1_signal_independence.json" \
        --out_jsonl     "${EXP4}/metrics/q1_signal_independence_perqid.jsonl" \
        --log           "${EXP4}/logs/q1_signal.log" \
        --batch_size    64 \
        2>&1 | tee "${EXP4}/logs/q1_signal_stdout.log"
fi

echo "  ✓ Hop scores: ${HOP_SCORES}"
echo ""

# ==========================================================================
#  Step 5: Chain-Aware XGBoost Verifier
# ==========================================================================
echo "━━━ Step 5: Chain-Aware XGBoost Verifier ━━━"

CHAIN_JSON="${EXP4}/metrics/q2q3q4_chain_verifier.json"

${PYTHON} ${TOOLS_DIR}/exp2_q2q3q4_chain_verifier.py \
    --hop_scores      "${HOP_SCORES}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${CHAIN_JSON}" \
    --out_preds_min   "${EXP4}/preds/dev_chain_verifier_min_preds.jsonl" \
    --out_preds_mean  "${EXP4}/preds/dev_chain_verifier_mean_preds.jsonl" \
    --log             "${EXP4}/logs/q2q3q4.log" \
    --n_folds 5 --seed 42 \
    2>&1 | tee "${EXP4}/logs/q2q3q4_stdout.log"

echo "  ✓ Verifier: ${CHAIN_JSON}"
echo ""

# ==========================================================================
#  Step 6: Full comparison
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: 7B (M=5) vs 1.5B (M=5 and M=10)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# ── baselines ──
# exp1b: 1.5B M=5
EXP1B_ORACLE    = 0.3982
EXP1B_NLI       = 0.3011
EXP1B_XGB       = 0.3409
EXP1B_FEAS_XGB  = 0.3743

# exp3b: 1.5B M=10
EXP3B_ORACLE    = 0.4344
EXP3B_NLI       = 0.2925
EXP3B_XGB       = 0.3622
EXP3B_FEAS_XGB  = 0.4005

# ── exp4: 7B M=5 ──
oracle = json.load(open('${ORACLE_JSON}'))
o_em = oracle['overall']['oracle_em']
o_f1 = oracle['overall']['oracle_f1']
feas_oracle = oracle.get('subset_docrecall1', {}).get('oracle_em', 0)

nli_metrics = {}
try:
    nli_metrics = json.load(open('${EXP4}/metrics/nli_baseline.json'))
except: pass
nli_em = nli_metrics.get('overall', {}).get('nli_em', 0)

chain = json.load(open('${CHAIN_JSON}'))
min_em = chain['min_pooling']['overall']['em']
min_f1 = chain['min_pooling']['overall']['f1']
mean_em = chain['mean_pooling']['overall']['em']
mean_f1 = chain['mean_pooling']['overall']['f1']

if min_em >= mean_em:
    best_tag, best = 'min', chain['min_pooling']
else:
    best_tag, best = 'mean', chain['mean_pooling']

xgb_em = best['overall']['em']
xgb_f1 = best['overall']['f1']
feas_xgb = best['feasible']['em']
bridge_em = best['bridge']['em']
comp_em = best['comparison']['em']

W = 76
print()
print('=' * W)
print(f\"  {'Method':<42} {'Oracle':>8} {'NLI':>8} {'XGB':>8} {'F1':>8}\")
print('-' * W)
print(f\"  {'1.5B M=5  (exp1b)':<42} {EXP1B_ORACLE:>8.4f} {EXP1B_NLI:>8.4f} {EXP1B_XGB:>8.4f} {'':>8}\")
print(f\"  {'1.5B M=10 (exp3b)':<42} {EXP3B_ORACLE:>8.4f} {EXP3B_NLI:>8.4f} {EXP3B_XGB:>8.4f} {'':>8}\")
print(f\"  {'7B M=5   (exp4) ★':<42} {o_em:>8.4f} {nli_em:>8.4f} {xgb_em:>8.4f} {xgb_f1:>8.4f}\")
print('-' * W)
print(f\"  {'Delta 7B vs 1.5B-M5':<42} {o_em-EXP1B_ORACLE:>+8.4f} {nli_em-EXP1B_NLI:>+8.4f} {xgb_em-EXP1B_XGB:>+8.4f}\")
print(f\"  {'Delta 7B vs 1.5B-M10':<42} {o_em-EXP3B_ORACLE:>+8.4f} {nli_em-EXP3B_NLI:>+8.4f} {xgb_em-EXP3B_XGB:>+8.4f}\")
print('=' * W)

print(f\"\n  Bridge EM:     {bridge_em:.4f}\")
print(f\"  Comparison EM: {comp_em:.4f}\")
print(f\"  Feasible EM:   {feas_xgb:.4f}  (1.5B-M5: {EXP1B_FEAS_XGB}  1.5B-M10: {EXP3B_FEAS_XGB})\")
print(f\"  Feasible oracle: {feas_oracle:.4f}\")

# Capture rate
oracle_gain = o_em - EXP1B_ORACLE
xgb_gain = xgb_em - EXP1B_XGB
capture = xgb_gain / oracle_gain if oracle_gain > 0 else 0
print(f\"\n  Oracle gain vs 1.5B-M5:  {oracle_gain:+.4f}\")
print(f\"  Verifier gain vs 1.5B-M5: {xgb_gain:+.4f}\")
print(f\"  Capture rate: {capture:.0%}\")

# Feature importances
print(f\"\n  Feature importances (7B, top-10):\")
for i, (f, v) in enumerate(list(best['feature_importances'].items())[:10]):
    print(f\"    #{i+1:>2}  {f:<24}  {v:.4f}\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp4 complete. Results in: ${EXP4}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"