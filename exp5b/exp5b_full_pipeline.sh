#!/usr/bin/env bash
# ==========================================================================
#  exp5b_full_pipeline.sh — Full Verification Pipeline on Exp5b Candidates
#  (7B flat prompt, T=0.7 normal / T=1.0 fallback)
#
#  Steps: Oracle (done) → NLI → Hop scores → XGBoost → Compare
#  No vLLM needed — NLI uses transformers directly on GPU
#  Estimated: ~25 min total
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"

EXP5B="${PROJ_ROOT}/exp5b"
CANDIDATES="${EXP5B}/candidates/dev_M5_7b_hightemp.jsonl"

echo "================================================================"
echo "  Exp5b: Full Pipeline (7B + High-Temp Fallback)"
echo "================================================================"

for f in "${CANDIDATES}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "$f" ]]; then echo "ERROR: missing $f"; exit 1; fi
done

mkdir -p "${EXP5B}/preds" "${EXP5B}/metrics" "${EXP5B}/logs"

# ── Step 1: Flat NLI Scoring ──
echo ""
echo "━━━ Step 1: Flat NLI Scoring ━━━"

NLI_PREDS="${EXP5B}/preds/dev_nli_preds.jsonl"

if [[ -f "${NLI_PREDS}" ]] && [[ $(wc -l < "${NLI_PREDS}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp1_nli_baseline.py \
        --candidates  "${CANDIDATES}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${GOLD}" \
        --model       "${NLI_MODEL}" \
        --out_metrics "${EXP5B}/metrics/nli_baseline.json" \
        --out_preds   "${NLI_PREDS}" \
        --batch_size  64 \
        2>&1 | tee "${EXP5B}/logs/nli_baseline.log"
fi
echo "  ✓ NLI preds: ${NLI_PREDS}"

# ── Step 2: Hop-Level NLI Scoring ──
echo ""
echo "━━━ Step 2: Hop-Level NLI Scoring ━━━"

HOP_SCORES="${EXP5B}/preds/dev_hop_scores.jsonl"

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
        --out_json      "${EXP5B}/metrics/q1_signal_independence.json" \
        --out_jsonl     "${EXP5B}/metrics/q1_signal_independence_perqid.jsonl" \
        --log           "${EXP5B}/logs/q1_signal.log" \
        --batch_size    64 \
        2>&1 | tee "${EXP5B}/logs/q1_signal_stdout.log"
fi
echo "  ✓ Hop scores: ${HOP_SCORES}"

# ── Step 3: Chain-Aware XGBoost Verifier ──
echo ""
echo "━━━ Step 3: Chain-Aware XGBoost Verifier ━━━"

CHAIN_JSON="${EXP5B}/metrics/q2q3q4_chain_verifier.json"

${PYTHON} ${TOOLS_DIR}/exp2_q2q3q4_chain_verifier.py \
    --hop_scores      "${HOP_SCORES}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${CHAIN_JSON}" \
    --out_preds_min   "${EXP5B}/preds/dev_chain_verifier_min_preds.jsonl" \
    --out_preds_mean  "${EXP5B}/preds/dev_chain_verifier_mean_preds.jsonl" \
    --log             "${EXP5B}/logs/q2q3q4.log" \
    --n_folds 5 --seed 42 \
    2>&1 | tee "${EXP5B}/logs/q2q3q4_stdout.log"

echo "  ✓ Verifier: ${CHAIN_JSON}"

# ── Step 4: Comparison ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: Exp5b vs All Baselines"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# Baselines
EXP1B_XGB = 0.3409
EXP3B_XGB = 0.3622
EXP4_ORACLE = 0.4879
EXP4_XGB = 0.4609

# Exp5b oracle (already computed)
oracle_5b = json.load(open('${EXP5B}/metrics/oracle_M5.json'))
o_em = oracle_5b['overall']['oracle_em']
o_f1 = oracle_5b['overall']['oracle_f1']
feas_oracle = oracle_5b.get('subset_docrecall1', {}).get('oracle_em', 0)

# Exp5b NLI
nli_5b = json.load(open('${EXP5B}/metrics/nli_baseline.json'))
nli_em = nli_5b.get('overall', {}).get('nli_em', 0)

# Exp5b chain XGB
chain_5b = json.load(open('${CHAIN_JSON}'))
min_em = chain_5b['min_pooling']['overall']['em']
mean_em = chain_5b['mean_pooling']['overall']['em']
if min_em >= mean_em:
    best_tag, best = 'min', chain_5b['min_pooling']
else:
    best_tag, best = 'mean', chain_5b['mean_pooling']

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
print(f\"  {'1.5B M=5 (Exp1b)':<42} {'0.3982':>8} {'0.3011':>8} {EXP1B_XGB:>8.4f}\")
print(f\"  {'1.5B M=10 (Exp3b)':<42} {'0.4344':>8} {'0.2925':>8} {EXP3B_XGB:>8.4f}\")
print(f\"  {'7B flat T=0.7 (Exp4)':<42} {EXP4_ORACLE:>8.4f} {'0.4536':>8} {EXP4_XGB:>8.4f}\")
print(f\"  {'7B flat T=0.7/1.0 fb (Exp5b) \u2605':<42} {o_em:>8.4f} {nli_em:>8.4f} {xgb_em:>8.4f} {xgb_f1:>8.4f}\")
print('-' * W)
print(f\"  {'Delta Exp5b vs Exp4':<42} {o_em-EXP4_ORACLE:>+8.4f} {nli_em-0.4536:>+8.4f} {xgb_em-EXP4_XGB:>+8.4f}\")
print('=' * W)

print(f\"\n  Bridge EM:     {bridge_em:.4f}\")
print(f\"  Comparison EM: {comp_em:.4f}\")
print(f\"  Feasible EM:   {feas_xgb:.4f}  (Exp4: 0.5108)\")
print(f\"  Feasible oracle: {feas_oracle:.4f}\")

# Capture rate
oracle_gain = o_em - EXP4_ORACLE
xgb_gain = xgb_em - EXP4_XGB
capture = xgb_gain / oracle_gain if oracle_gain > 0 else 0
print(f\"\n  Oracle gain:   {oracle_gain:+.4f}\")
print(f\"  Verifier gain: {xgb_gain:+.4f}\")
if oracle_gain > 0:
    print(f\"  Capture rate:  {capture:.0%}\")

# Feature importances
print(f\"\n  Feature importances (Exp5b, top-10):\")
for i, (f, v) in enumerate(list(best['feature_importances'].items())[:10]):
    print(f\"    #{i+1:>2}  {f:<24}  {v:.4f}\")

# Verifier gap
gap = o_em - xgb_em
print(f\"\n  Verifier gap: {gap:.4f} (oracle {o_em:.4f} - XGB {xgb_em:.4f})\")
print(f\"  Exp4 gap was: {EXP4_ORACLE - EXP4_XGB:.4f}\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp5b pipeline complete. Results in: ${EXP5B}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"