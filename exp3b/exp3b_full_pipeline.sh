#!/usr/bin/env bash
# ==========================================================================
#  exp3b_full_pipeline.sh — Full Verification Pipeline on M=10 Candidates
#
#  Runs the complete verification stack on the M=10 candidates from exp3b:
#    Step 1: Flat NLI scoring          (GPU, ~15 min)
#    Step 2: Hop-level NLI scoring     (GPU, ~25 min)
#    Step 3: Surface XGBoost verifier  (CPU, ~2 min)
#    Step 4: Chain-aware XGBoost       (CPU, ~2 min)
#    Step 5: Comparison vs exp1b M=5
#
#  Prerequisites:
#    - exp3b candidates already generated: exp3b/candidates/dev_M10_candidates_flat.jsonl
#    - GPU available for NLI model
#    - No vLLM needed (NLI uses transformers directly)
#
#  Decision gate:
#    xgb_em(M=10) > xgb_em(M=5 = 0.3409) by ≥ 1.5pp  →  M=10 is the new baseline
#    xgb_em(M=10) ≈ xgb_em(M=5) within 1pp             →  verifier can't exploit extra cands
# ==========================================================================

set -euo pipefail

# ── paths ──
PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Shared inputs (read-only)
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"

# Exp3b inputs (from generation step)
CANDIDATES="${PROJ_ROOT}/exp3b/candidates/dev_M10_candidates_flat.jsonl"

# Exp3b outputs
EXP3B_DIR="${PROJ_ROOT}/exp3b"
NLI_PREDS="${EXP3B_DIR}/preds/dev_nli_preds.jsonl"
HOP_SCORES="${EXP3B_DIR}/preds/dev_hop_scores.jsonl"
HOP_PERQID="${EXP3B_DIR}/metrics/q1_signal_independence_perqid.jsonl"
HOP_JSON="${EXP3B_DIR}/metrics/q1_signal_independence.json"
CHAIN_JSON="${EXP3B_DIR}/metrics/q2q3q4_chain_verifier.json"
CHAIN_PREDS_MIN="${EXP3B_DIR}/preds/dev_chain_verifier_min_preds.jsonl"
CHAIN_PREDS_MEAN="${EXP3B_DIR}/preds/dev_chain_verifier_mean_preds.jsonl"
MANIFEST="${EXP3B_DIR}/manifest.json"

# ── pre-flight ──
echo "=== Exp3b: Full Verification Pipeline (M=10) ==="
echo "Candidates: ${CANDIDATES}"
echo "Evidence:   ${EVIDENCE}"
echo "NLI model:  ${NLI_MODEL}"
echo ""

for f in "${CANDIDATES}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing input file: $f"
        exit 1
    fi
done

if [[ ! -d "${NLI_MODEL}" ]]; then
    echo "ERROR: NLI model not found at ${NLI_MODEL}"
    exit 1
fi

mkdir -p "${EXP3B_DIR}/preds" "${EXP3B_DIR}/metrics" "${EXP3B_DIR}/logs"

# ==========================================================================
#  STEP 1: Flat NLI Scoring
#  Scores each of 10 candidates against flattened evidence
#  Output: dev_nli_preds.jsonl  {qid, pred, scores: [10 floats], best_idx}
# ==========================================================================
echo ""
echo "━━━ Step 1: Flat NLI Scoring (10 candidates per question) ━━━"

if [[ -f "${NLI_PREDS}" ]]; then
    N_DONE=$(wc -l < "${NLI_PREDS}")
    echo "  Found existing ${NLI_PREDS} with ${N_DONE} lines"
    if [[ "${N_DONE}" -ge 7405 ]]; then
        echo "  ✓ Already complete — skipping"
    else
        echo "  Incomplete — re-running"
        rm -f "${NLI_PREDS}"
    fi
fi

if [[ ! -f "${NLI_PREDS}" ]] || [[ $(wc -l < "${NLI_PREDS}") -lt 7405 ]]; then
    ${PYTHON} ${TOOLS_DIR}/exp1_nli_baseline.py \
        --candidates  "${CANDIDATES}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${GOLD}" \
        --model       "${NLI_MODEL}" \
        --out_metrics "${EXP3B_DIR}/metrics/exp3b_nli_baseline.json" \
        --out_preds   "${NLI_PREDS}" \
        --batch_size  64 \
        2>&1 | tee "${EXP3B_DIR}/logs/exp3b_nli_baseline.log"
fi

echo "  ✓ Flat NLI preds: ${NLI_PREDS}"

# ==========================================================================
#  STEP 2: Hop-Level NLI Scoring (Signal Independence — Q1)
#  Scores each of 10 candidates against hop1 and hop2 separately
#  Output: dev_hop_scores.jsonl  (reusable for chain-aware XGBoost)
# ==========================================================================
echo ""
echo "━━━ Step 2: Hop-Level NLI Scoring (10 candidates × 2 hops) ━━━"

if [[ -f "${HOP_SCORES}" ]]; then
    N_DONE=$(wc -l < "${HOP_SCORES}")
    echo "  Found existing ${HOP_SCORES} with ${N_DONE} lines"
    if [[ "${N_DONE}" -ge 7405 ]]; then
        echo "  ✓ Already complete — skipping"
    else
        echo "  Incomplete — re-running"
        rm -f "${HOP_SCORES}"
    fi
fi

if [[ ! -f "${HOP_SCORES}" ]] || [[ $(wc -l < "${HOP_SCORES}") -lt 7405 ]]; then
    ${PYTHON} ${TOOLS_DIR}/exp2_q1_signal_independence.py \
        --candidates  "${CANDIDATES}" \
        --nli_preds   "${NLI_PREDS}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${GOLD}" \
        --model       "${NLI_MODEL}" \
        --out_hop_scores "${HOP_SCORES}" \
        --out_json    "${HOP_JSON}" \
        --out_jsonl   "${HOP_PERQID}" \
        --log         "${EXP3B_DIR}/logs/exp3b_q1_signal.log" \
        --batch_size  64 \
        2>&1 | tee "${EXP3B_DIR}/logs/exp3b_q1_signal_stdout.log"
fi

echo "  ✓ Hop scores: ${HOP_SCORES}"

# ==========================================================================
#  STEP 3: Chain-Aware XGBoost Verifier (Q2/Q3/Q4)
#  Uses hop scores + surface features → 5-fold CV XGBoost
#  Compares min vs mean pooling, bridge vs comparison
# ==========================================================================
echo ""
echo "━━━ Step 3: Chain-Aware XGBoost Verifier ━━━"

${PYTHON} ${TOOLS_DIR}/exp2_q2q3q4_chain_verifier.py \
    --hop_scores      "${HOP_SCORES}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${CHAIN_JSON}" \
    --out_preds_min   "${CHAIN_PREDS_MIN}" \
    --out_preds_mean  "${CHAIN_PREDS_MEAN}" \
    --log             "${EXP3B_DIR}/logs/exp3b_q2q3q4.log" \
    --n_folds 5 --seed 42 \
    2>&1 | tee "${EXP3B_DIR}/logs/exp3b_q2q3q4_stdout.log"

echo "  ✓ Chain verifier results: ${CHAIN_JSON}"

# ==========================================================================
#  STEP 4: Summary — Compare M=10 vs M=5 (exp1b)
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: M=10 Pipeline vs M=5 (exp1b) Baselines"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# ── exp1b baselines (M=5) ──
EXP1B_ORACLE_EM      = 0.3982
EXP1B_ORACLE_F1      = 0.5350
EXP1B_NLI_EM         = 0.3011
EXP1B_XGB_EM         = 0.3409
EXP1B_XGB_F1         = 0.4638
EXP1B_FEASIBLE_XGB   = 0.3743

# ── exp3b oracle (M=10, already computed) ──
oracle = json.load(open('${EXP3B_DIR}/metrics/oracle_M10_dev.json'))
oracle_em = oracle['overall']['oracle_em']
oracle_f1 = oracle['overall']['oracle_f1']
feas_oracle = oracle.get('subset_docrecall1', {}).get('oracle_em', 0)

# ── exp3b NLI baseline (M=10) ──
nli = {}
nli_path = '${EXP3B_DIR}/metrics/exp3b_nli_baseline.json'
try:
    nli = json.load(open(nli_path))
except:
    print(f'  [warn] Could not load {nli_path}')

nli_em = nli.get('overall', {}).get('nli_em', nli.get('overall', {}).get('em', 0))
nli_f1 = nli.get('overall', {}).get('nli_f1', nli.get('overall', {}).get('f1', 0))

# ── exp3b chain-aware XGBoost (M=10) ──
chain = json.load(open('${CHAIN_JSON}'))
# pick the better of min/mean pooling
min_em = chain['min_pooling']['overall']['em']
mean_em = chain['mean_pooling']['overall']['em']
if min_em >= mean_em:
    best_tag = 'min'
    best = chain['min_pooling']
else:
    best_tag = 'mean'
    best = chain['mean_pooling']

xgb_em = best['overall']['em']
xgb_f1 = best['overall']['f1']
feas_xgb = best['feasible']['em']
bridge_em = best['bridge']['em']
comp_em = best['comparison']['em']

W = 72
print()
print('=' * W)
print(f\"  {'Method':<32} {'EM':>8} {'F1':>8}  {'vs M=5':>8}\")
print('-' * W)

rows = [
    ('Oracle@5 (M=5 exp1b)',    EXP1B_ORACLE_EM, EXP1B_ORACLE_F1, '—'),
    ('Oracle@10 (M=10 exp3b)',  oracle_em,        oracle_f1,
     f'+{oracle_em - EXP1B_ORACLE_EM:.4f}'),
    ('', 0, 0, ''),
    ('NLI baseline (M=5)',      EXP1B_NLI_EM,     0,                '—'),
    ('NLI baseline (M=10)',     nli_em,            nli_f1,
     f'{nli_em - EXP1B_NLI_EM:+.4f}' if nli_em else '?'),
    ('', 0, 0, ''),
    ('XGB surface (M=5 exp1b)', EXP1B_XGB_EM,     EXP1B_XGB_F1,    '—'),
    (f'Chain XGB [{best_tag}] (M=10)', xgb_em,    xgb_f1,
     f'{xgb_em - EXP1B_XGB_EM:+.4f}'),
]

for label, em_val, f1_val, delta in rows:
    if not label:
        print()
        continue
    if f1_val:
        print(f\"  {label:<32} {em_val:>8.4f} {f1_val:>8.4f}  {delta:>8}\")
    else:
        print(f\"  {label:<32} {em_val:>8.4f} {'':>8}  {delta:>8}\")

print('=' * W)

print(f\"\n  Bridge EM (M=10):     {bridge_em:.4f}\")
print(f\"  Comparison EM (M=10): {comp_em:.4f}\")
print(f\"  Feasible EM (M=10):   {feas_xgb:.4f}  (M=5: {EXP1B_FEASIBLE_XGB:.4f}  \"
      f\"delta: {feas_xgb - EXP1B_FEASIBLE_XGB:+.4f})\")

# ── decision ──
delta_xgb = xgb_em - EXP1B_XGB_EM
delta_oracle = oracle_em - EXP1B_ORACLE_EM
captured = delta_xgb / delta_oracle if delta_oracle > 0 else 0

print()
print(f\"  Oracle ceiling gained:  {delta_oracle:+.4f}  (M=10 vs M=5)\")
print(f\"  Verifier EM gained:     {delta_xgb:+.4f}\")
print(f\"  Capture rate:           {captured:.0%} of oracle gain transferred to verifier\")
print()

if delta_xgb >= 0.015:
    print(f\"  → SUCCESS: +{delta_xgb:.4f} EM. M=10 is the new baseline.\")
    print(f\"    The verifier captured {captured:.0%} of the oracle ceiling gain.\")
elif delta_xgb >= 0.005:
    print(f\"  → PARTIAL: +{delta_xgb:.4f} EM. Some gain but verifier struggles with 10 cands.\")
    print(f\"    The extra candidates help but the reranker may need tuning for M=10.\")
elif delta_xgb > 0:
    print(f\"  → MARGINAL: +{delta_xgb:.4f} EM. Verifier barely benefits from extra cands.\")
    print(f\"    The verifier can't distinguish correct from wrong among 10 options.\")
else:
    print(f\"  → NO GAIN: {delta_xgb:+.4f} EM. More candidates didn't help the verifier.\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp3b full pipeline complete."
echo "  All results in: ${EXP3B_DIR}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"