#!/usr/bin/env bash
# ==========================================================================
#  exp3b_analysis.sh — Q6/Q7/Q8/Q9/Q10 Analysis on M=10 Results
#
#  Runs the full diagnostic suite on the M=10 chain-aware verifier:
#    Step 1: Q6/Q7 — Abstention curve (AUROC + accuracy/coverage)
#    Step 2: Q8   — Calibration (ECE + optional Platt scaling)
#    Step 3: Q9   — Failure taxonomy (A/B/C1/C2/D buckets)
#            Q10  — Chain-aware vs flat NLI differential
#    Step 4: Summary comparison vs exp1b M=5
#
#  All CPU-only — no GPU, no vLLM needed.
#  Estimated runtime: ~30 seconds total
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Shared inputs
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

# Exp3b inputs (from pipeline run)
EXP3B_DIR="${PROJ_ROOT}/exp3b"
CHAIN_PREDS="${EXP3B_DIR}/preds/dev_chain_verifier_min_preds.jsonl"
NLI_PREDS="${EXP3B_DIR}/preds/dev_nli_preds.jsonl"
ORACLE_PERQID="${EXP3B_DIR}/metrics/oracle_M10_dev_perqid.jsonl"

# ── pre-flight ──
echo "=== Exp3b: Diagnostic Analysis Suite (M=10) ==="
echo ""

for f in "${CHAIN_PREDS}" "${NLI_PREDS}" "${ORACLE_PERQID}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f"
        exit 1
    fi
done

mkdir -p "${EXP3B_DIR}/metrics" "${EXP3B_DIR}/logs"

# ==========================================================================
#  STEP 1: Q6/Q7 — Abstention Analysis
# ==========================================================================
echo "━━━ Step 1: Q6/Q7 Abstention Analysis ━━━"

${PYTHON} ${TOOLS_DIR}/exp2_q6q7_abstention.py \
    --verifier_preds  "${CHAIN_PREDS}" \
    --oracle_perqid   "${ORACLE_PERQID}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP3B_DIR}/metrics/q6q7_abstention.json" \
    --log             "${EXP3B_DIR}/logs/q6q7_abstention.log" \
    2>&1 | tee "${EXP3B_DIR}/logs/q6q7_stdout.log"

echo ""
echo "  ✓ Abstention results: ${EXP3B_DIR}/metrics/q6q7_abstention.json"

# ==========================================================================
#  STEP 2: Q8 — Calibration
# ==========================================================================
echo ""
echo "━━━ Step 2: Q8 Calibration ━━━"

${PYTHON} ${TOOLS_DIR}/exp2_q8_calibration.py \
    --verifier_preds  "${CHAIN_PREDS}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP3B_DIR}/metrics/q8_calibration.json" \
    --log             "${EXP3B_DIR}/logs/q8_calibration.log" \
    2>&1 | tee "${EXP3B_DIR}/logs/q8_stdout.log"

echo ""
echo "  ✓ Calibration results: ${EXP3B_DIR}/metrics/q8_calibration.json"

# ==========================================================================
#  STEP 3: Q9/Q10 — Failure Taxonomy + Differential
# ==========================================================================
echo ""
echo "━━━ Step 3: Q9/Q10 Failure Taxonomy ━━━"

${PYTHON} ${TOOLS_DIR}/exp2_q9q10_failure_analysis.py \
    --chain_preds     "${CHAIN_PREDS}" \
    --nli_preds       "${NLI_PREDS}" \
    --oracle_perqid   "${ORACLE_PERQID}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP3B_DIR}/metrics/q9q10_failure_analysis.json" \
    --log             "${EXP3B_DIR}/logs/q9q10_failure_analysis.log" \
    2>&1 | tee "${EXP3B_DIR}/logs/q9q10_stdout.log"

echo ""
echo "  ✓ Failure analysis: ${EXP3B_DIR}/metrics/q9q10_failure_analysis.json"

# ==========================================================================
#  STEP 4: Side-by-side comparison vs exp1b M=5
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  M=10 vs M=5 — Full Diagnostic Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

# ── load M=5 (exp1b) baselines ──
m5_abs = json.load(open('${PROJ_ROOT}/exp1b/metrics/q6q7_abstention.json'))
m5_cal = json.load(open('${PROJ_ROOT}/exp1b/metrics/q8_calibration.json'))
m5_fail = json.load(open('${PROJ_ROOT}/exp1b/metrics/q9q10_failure_analysis.json'))

# ── load M=10 (exp3b) results ──
m10_abs = json.load(open('${EXP3B_DIR}/metrics/q6q7_abstention.json'))
m10_cal = json.load(open('${EXP3B_DIR}/metrics/q8_calibration.json'))
m10_fail = json.load(open('${EXP3B_DIR}/metrics/q9q10_failure_analysis.json'))

W = 72

# ── Abstention ──
print()
print('=' * W)
print('  ABSTENTION (Q6/Q7)')
print('=' * W)
print(f\"  {'Metric':<40} {'M=5':>10} {'M=10':>10}\")
print('  ' + '-' * (W - 2))

m5_auroc = m5_abs['q6_separation']['auroc']
m10_auroc = m10_abs['q6_separation']['auroc']
print(f\"  {'AUROC (has-correct vs all-wrong)':<40} {m5_auroc:>10.4f} {m10_auroc:>10.4f}\")

m5_em100 = m5_abs['q7_coverage_curve']['em_at_100pct_coverage']
m10_em100 = m10_abs['q7_coverage_curve']['em_at_100pct_coverage']
print(f\"  {'EM at 100% coverage':<40} {m5_em100:>10.4f} {m10_em100:>10.4f}\")

m5_em70 = m5_abs['q7_coverage_curve'].get('em_at_70pct_coverage', 0)
m10_em70 = m10_abs['q7_coverage_curve'].get('em_at_70pct_coverage', 0)
print(f\"  {'EM at ~70% coverage':<40} {m5_em70:>10.4f} {m10_em70:>10.4f}\")

m5_tau = m5_abs['q7_coverage_curve'].get('optimal_tau', 0)
m5_tau_em = m5_abs['q7_coverage_curve'].get('em_at_optimal_tau', 0)
m5_tau_cov = m5_abs['q7_coverage_curve'].get('coverage_at_optimal_tau', 0)
m10_tau = m10_abs['q7_coverage_curve'].get('optimal_tau', 0)
m10_tau_em = m10_abs['q7_coverage_curve'].get('em_at_optimal_tau', 0)
m10_tau_cov = m10_abs['q7_coverage_curve'].get('coverage_at_optimal_tau', 0)
print(f\"  {'Optimal τ (≥50% cov)':<40} {'τ='+str(m5_tau):>10} {'τ='+str(m10_tau):>10}\")
print(f\"  {'EM at optimal τ':<40} {m5_tau_em:>10.4f} {m10_tau_em:>10.4f}\")
print(f\"  {'Coverage at optimal τ':<40} {m5_tau_cov:>10.1%} {m10_tau_cov:>10.1%}\")

# ── Calibration ──
print()
print('=' * W)
print('  CALIBRATION (Q8)')
print('=' * W)
m5_ece = m5_cal['chain_aware_verifier']['verifier_level_ece']
m10_ece = m10_cal['chain_aware_verifier']['verifier_level_ece']
print(f\"  {'Verifier-level ECE':<40} {m5_ece:>10.4f} {m10_ece:>10.4f}\")
m10_clf_ece = m10_cal['chain_aware_verifier']['classifier_level_ece']
print(f\"  {'Classifier-level ECE (M=10)':<40} {'':>10} {m10_clf_ece:>10.4f}\")
print(f\"  {'Q8 decision':<40}\")
print(f\"    M=5:  {m5_cal['q8_decision']}\")
print(f\"    M=10: {m10_cal['q8_decision']}\")

# ── Failure Taxonomy ──
print()
print('=' * W)
print('  FAILURE TAXONOMY (Q9)')
print('=' * W)
print(f\"  {'Bucket':<40} {'M=5 %':>10} {'M=10 %':>10} {'Delta':>8}\")
print('  ' + '-' * (W - 2))

for key, label in [
    ('A_retrieval_failure',         'A — Retrieval failure'),
    ('B_oracle_failure',            'B — Generator failure'),
    ('C1_verifier_bad',             'C1 — Verifier picked garbage'),
    ('C2_verifier_plausible_wrong', 'C2 — Verifier picked plausible wrong'),
    ('D_success',                   'D — Success'),
]:
    m5_pct = m5_fail['q9_failure_taxonomy'][key]['pct']
    m10_pct = m10_fail['q9_failure_taxonomy'][key]['pct']
    delta = m10_pct - m5_pct
    print(f\"  {label:<40} {m5_pct:>10.1%} {m10_pct:>10.1%} {delta:>+8.1%}\")

# ── Q10 Differential ──
print()
print('=' * W)
print('  CHAIN vs FLAT NLI DIFFERENTIAL (Q10)')
print('=' * W)
print(f\"  {'Group':<30} {'M=5 N':>8} {'M=10 N':>8}\")
print('  ' + '-' * (W - 2))
for grp in ['BOTH_CORRECT', 'CHAIN_WINS', 'FLAT_WINS', 'BOTH_WRONG']:
    m5_n = m5_fail['q10_differential'][grp]['n']
    m10_n = m10_fail['q10_differential'][grp]['n']
    print(f\"  {grp:<30} {m5_n:>8} {m10_n:>8}\")

m5_net = m5_fail['q10_differential']['net_gain_chain_over_flat']
m10_net = m10_fail['q10_differential']['net_gain_chain_over_flat']
print(f\"  {'Net gain (chain over flat)':<30} {m5_net:>8} {m10_net:>8}\")

print()
print('=' * W)
print('  Analysis complete. All results in: ${EXP3B_DIR}/metrics/')
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp3b analysis complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"