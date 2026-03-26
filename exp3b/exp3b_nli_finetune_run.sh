#!/usr/bin/env bash
# ==========================================================================
#  exp3b_nli_finetune_run.sh — NLI Fine-Tuning on HotpotQA + Pipeline Re-Run
#
#  Phase 1: Prepare training data from HotpotQA train split  (~2 min, CPU)
#  Phase 2: Fine-tune nli-roberta-base                       (~2-4 hrs, GPU)
#  Phase 3: Re-score M=10 candidates with fine-tuned model   (~15 min, GPU)
#  Phase 4: Re-run chain-aware XGBoost verifier              (~2 min, CPU)
#  Phase 5: Compare against baseline
#
#  IMPORTANT: Uses HotpotQA TRAIN split only for fine-tuning.
#  Dev set is NEVER seen during training.
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Inputs
HOTPOT_TRAIN="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_train_v1.1.json"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
BASE_NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"
CANDIDATES="${PROJ_ROOT}/exp3b/candidates/dev_M10_candidates_flat.jsonl"

# Outputs
FT_DIR="${PROJ_ROOT}/exp3b/nli_finetune"
FT_DATA="${FT_DIR}/data"
FT_MODEL="${FT_DIR}/model"
FT_PREDS_DIR="${PROJ_ROOT}/exp3b/preds_finetuned"
FT_METRICS_DIR="${PROJ_ROOT}/exp3b/metrics_finetuned"

echo "=== NLI Fine-Tuning Pipeline ==="
echo "Train data:  ${HOTPOT_TRAIN}"
echo "Base model:  ${BASE_NLI_MODEL}"
echo "Candidates:  ${CANDIDATES}"
echo ""

for f in "${HOTPOT_TRAIN}" "${GOLD}" "${EVIDENCE}" "${CANDIDATES}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f"
        exit 1
    fi
done

mkdir -p "${FT_DATA}" "${FT_MODEL}" "${FT_PREDS_DIR}" "${FT_METRICS_DIR}" \
         "${PROJ_ROOT}/exp3b/logs"

# ==========================================================================
#  Phase 1: Prepare fine-tuning data
# ==========================================================================
echo "━━━ Phase 1: Prepare Fine-Tuning Data ━━━"

if [[ -f "${FT_DATA}/train.jsonl" ]] && [[ -f "${FT_DATA}/val.jsonl" ]]; then
    echo "  Data already prepared — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp3b_nli_prepare_finetune_data.py \
        --hotpot_train  "${HOTPOT_TRAIN}" \
        --out_dir       "${FT_DATA}" \
        --val_frac      0.05 \
        --max_neg_per_question 6 \
        --seed          42 \
        2>&1 | tee "${PROJ_ROOT}/exp3b/logs/nli_data_prep.log"
fi

echo "  ✓ Training data ready"
echo ""

# ==========================================================================
#  Phase 2: Fine-tune
# ==========================================================================
echo "━━━ Phase 2: Fine-Tune NLI Model ━━━"

if [[ -f "${FT_MODEL}/config.json" ]]; then
    echo "  Fine-tuned model already exists — skipping"
    echo "  (Delete ${FT_MODEL} to re-train)"
else
    ${PYTHON} ${TOOLS_DIR}/exp3b_nli_finetune.py \
        --base_model    "${BASE_NLI_MODEL}" \
        --train_data    "${FT_DATA}/train.jsonl" \
        --val_data      "${FT_DATA}/val.jsonl" \
        --out_model_dir "${FT_MODEL}" \
        --epochs        3 \
        --batch_size    32 \
        --lr            2e-5 \
        --warmup_ratio  0.1 \
        --max_length    256 \
        --seed          42 \
        --patience      2 \
        2>&1 | tee "${PROJ_ROOT}/exp3b/logs/nli_finetune.log"
fi

echo "  ✓ Fine-tuned model: ${FT_MODEL}"
echo ""

# ==========================================================================
#  Phase 3: Re-score candidates with fine-tuned NLI
# ==========================================================================
echo "━━━ Phase 3: Flat NLI Scoring (fine-tuned model) ━━━"

FT_NLI_PREDS="${FT_PREDS_DIR}/dev_nli_preds.jsonl"

if [[ -f "${FT_NLI_PREDS}" ]] && [[ $(wc -l < "${FT_NLI_PREDS}") -ge 7405 ]]; then
    echo "  Flat NLI preds already complete — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp1_nli_baseline.py \
        --candidates  "${CANDIDATES}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${GOLD}" \
        --model       "${FT_MODEL}" \
        --out_metrics "${FT_METRICS_DIR}/nli_baseline.json" \
        --out_preds   "${FT_NLI_PREDS}" \
        --batch_size  64 \
        2>&1 | tee "${PROJ_ROOT}/exp3b/logs/nli_ft_flat_scoring.log"
fi

echo "  ✓ Flat NLI preds: ${FT_NLI_PREDS}"
echo ""

# ==========================================================================
#  Phase 3b: Hop-level NLI scoring (fine-tuned model)
# ==========================================================================
echo "━━━ Phase 3b: Hop-Level NLI Scoring (fine-tuned model) ━━━"

FT_HOP_SCORES="${FT_PREDS_DIR}/dev_hop_scores.jsonl"
FT_Q1_JSON="${FT_METRICS_DIR}/q1_signal_independence.json"
FT_Q1_JSONL="${FT_METRICS_DIR}/q1_signal_independence_perqid.jsonl"

if [[ -f "${FT_HOP_SCORES}" ]] && [[ $(wc -l < "${FT_HOP_SCORES}") -ge 7405 ]]; then
    echo "  Hop scores already complete — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp2_q1_signal_independence.py \
        --candidates    "${CANDIDATES}" \
        --nli_preds     "${FT_NLI_PREDS}" \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --model         "${FT_MODEL}" \
        --out_hop_scores "${FT_HOP_SCORES}" \
        --out_json      "${FT_Q1_JSON}" \
        --out_jsonl     "${FT_Q1_JSONL}" \
        --log           "${PROJ_ROOT}/exp3b/logs/nli_ft_hop_scoring.log" \
        --batch_size    64 \
        2>&1 | tee "${PROJ_ROOT}/exp3b/logs/nli_ft_hop_scoring_stdout.log"
fi

echo "  ✓ Hop scores: ${FT_HOP_SCORES}"
echo ""

# ==========================================================================
#  Phase 4: Chain-Aware XGBoost Verifier
# ==========================================================================
echo "━━━ Phase 4: Chain-Aware XGBoost Verifier ━━━"

FT_CHAIN_JSON="${FT_METRICS_DIR}/q2q3q4_chain_verifier.json"
FT_CHAIN_PREDS_MIN="${FT_PREDS_DIR}/dev_chain_verifier_min_preds.jsonl"
FT_CHAIN_PREDS_MEAN="${FT_PREDS_DIR}/dev_chain_verifier_mean_preds.jsonl"

${PYTHON} ${TOOLS_DIR}/exp2_q2q3q4_chain_verifier.py \
    --hop_scores      "${FT_HOP_SCORES}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${FT_CHAIN_JSON}" \
    --out_preds_min   "${FT_CHAIN_PREDS_MIN}" \
    --out_preds_mean  "${FT_CHAIN_PREDS_MEAN}" \
    --log             "${PROJ_ROOT}/exp3b/logs/nli_ft_chain_verifier.log" \
    --n_folds 5 --seed 42 \
    2>&1 | tee "${PROJ_ROOT}/exp3b/logs/nli_ft_chain_verifier_stdout.log"

echo "  ✓ Chain verifier: ${FT_CHAIN_JSON}"
echo ""

# ==========================================================================
#  Phase 5: Compare fine-tuned vs baseline
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: Fine-Tuned NLI vs Baseline NLI (both on M=10)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# ── baseline (exp3b with original nli-roberta-base) ──
base_chain = json.load(open('${PROJ_ROOT}/exp3b/metrics/q2q3q4_chain_verifier.json'))
base_nli   = json.load(open('${PROJ_ROOT}/exp3b/metrics/exp3b_nli_baseline.json'))

base_min = base_chain['min_pooling']
base_xgb_em  = base_min['overall']['em']
base_xgb_f1  = base_min['overall']['f1']
base_feas_em = base_min['feasible']['em']
base_nli_em  = base_nli.get('overall', {}).get('nli_em', 0)

# ── fine-tuned ──
ft_chain = json.load(open('${FT_CHAIN_JSON}'))
ft_nli   = json.load(open('${FT_METRICS_DIR}/nli_baseline.json'))

ft_min = ft_chain['min_pooling']
ft_mean = ft_chain['mean_pooling']
best_tag = 'min' if ft_min['overall']['em'] >= ft_mean['overall']['em'] else 'mean'
ft_best = ft_min if best_tag == 'min' else ft_mean

ft_xgb_em  = ft_best['overall']['em']
ft_xgb_f1  = ft_best['overall']['f1']
ft_feas_em = ft_best['feasible']['em']
ft_nli_em  = ft_nli.get('overall', {}).get('nli_em', 0)

W = 72
print()
print('=' * W)
print(f\"  {'Method':<38} {'EM':>8} {'F1':>8} {'vs Base':>10}\")
print('-' * W)
print(f\"  {'NLI baseline (original)':<38} {base_nli_em:>8.4f} {'':>8} {'—':>10}\")
print(f\"  {'NLI baseline (fine-tuned)':<38} {ft_nli_em:>8.4f} {'':>8} {ft_nli_em - base_nli_em:>+10.4f}\")
print()
print(f\"  {'Chain XGB [min] (original NLI)':<38} {base_xgb_em:>8.4f} {base_xgb_f1:>8.4f} {'—':>10}\")
print(f\"  {'Chain XGB [{best_tag}] (fine-tuned NLI)':<38} {ft_xgb_em:>8.4f} {ft_xgb_f1:>8.4f} {ft_xgb_em - base_xgb_em:>+10.4f}\")
print()
print(f\"  {'Feasible EM (original)':<38} {base_feas_em:>8.4f}\")
print(f\"  {'Feasible EM (fine-tuned)':<38} {ft_feas_em:>8.4f} {'':>8} {ft_feas_em - base_feas_em:>+10.4f}\")
print('=' * W)

# Feature importances comparison
print()
print('  Feature importances (fine-tuned model, top-10):')
for i, (f, v) in enumerate(list(ft_best['feature_importances'].items())[:10]):
    base_v = base_min['feature_importances'].get(f, 0)
    delta = v - base_v
    marker = f' ({delta:+.3f})' if abs(delta) > 0.005 else ''
    print(f\"    #{i+1:>2}  {f:<24}  {v:.4f}{marker}\")

# Decision
delta = ft_xgb_em - base_xgb_em
print()
if delta >= 0.01:
    print(f\"  → STRONG IMPROVEMENT: +{delta:.4f} EM. Fine-tuned NLI is clearly better.\")
    print(f\"    NLI features should now rank higher in importances.\")
elif delta >= 0.005:
    print(f\"  → MODEST IMPROVEMENT: +{delta:.4f} EM. Fine-tuning helps but doesn't transform.\")
elif delta >= -0.003:
    print(f\"  → NO EFFECT: {delta:+.4f} EM. NLI quality was not the C2 bottleneck.\")
    print(f\"    The remaining C2 errors may be fundamentally ambiguous.\")
else:
    print(f\"  → REGRESSED: {delta:+.4f} EM. Fine-tuning hurt. Possible overfitting.\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NLI fine-tuning pipeline complete."
echo "  Results in: ${FT_METRICS_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"