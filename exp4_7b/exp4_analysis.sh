#!/usr/bin/env bash
# ==========================================================================
#  exp4_analysis.sh — Q6/Q7 + Q9/Q10 on 7B Results
#  CPU only, ~30 seconds
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

EXP4="${PROJ_ROOT}/exp4_7b"
CHAIN_PREDS="${EXP4}/preds/dev_chain_verifier_mean_preds.jsonl"
NLI_PREDS="${EXP4}/preds/dev_nli_preds.jsonl"
ORACLE_PERQID="${EXP4}/metrics/oracle_M5_dev_perqid.jsonl"

echo "=== Exp4: 7B Diagnostic Analysis ==="

for f in "${CHAIN_PREDS}" "${NLI_PREDS}" "${ORACLE_PERQID}" "${EVIDENCE}" "${GOLD}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f"
        exit 1
    fi
done

# ── Q6/Q7 Abstention ──
echo ""
echo "━━━ Q6/Q7 Abstention ━━━"
${PYTHON} ${TOOLS_DIR}/exp2_q6q7_abstention.py \
    --verifier_preds  "${CHAIN_PREDS}" \
    --oracle_perqid   "${ORACLE_PERQID}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP4}/metrics/q6q7_abstention.json" \
    --log             "${EXP4}/logs/q6q7.log"

# ── Q8 Calibration ──
echo ""
echo "━━━ Q8 Calibration ━━━"
${PYTHON} ${TOOLS_DIR}/exp2_q8_calibration.py \
    --verifier_preds  "${CHAIN_PREDS}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP4}/metrics/q8_calibration.json" \
    --log             "${EXP4}/logs/q8.log"

# ── Q9/Q10 Failure Taxonomy ──
echo ""
echo "━━━ Q9/Q10 Failure Taxonomy ━━━"
${PYTHON} ${TOOLS_DIR}/exp2_q9q10_failure_analysis.py \
    --chain_preds     "${CHAIN_PREDS}" \
    --nli_preds       "${NLI_PREDS}" \
    --oracle_perqid   "${ORACLE_PERQID}" \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --out_json        "${EXP4}/metrics/q9q10_failure_analysis.json" \
    --log             "${EXP4}/logs/q9q10.log"

# ── Three-way comparison ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Three-Way Diagnostic Comparison: 1.5B-M5 / 1.5B-M10 / 7B-M5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

m5  = {
    'abs': json.load(open('${PROJ_ROOT}/exp1b/metrics/q6q7_abstention.json')),
    'cal': json.load(open('${PROJ_ROOT}/exp1b/metrics/q8_calibration.json')),
    'fail': json.load(open('${PROJ_ROOT}/exp1b/metrics/q9q10_failure_analysis.json')),
}
m10 = {
    'abs': json.load(open('${PROJ_ROOT}/exp3b/metrics/q6q7_abstention.json')),
    'cal': json.load(open('${PROJ_ROOT}/exp3b/metrics/q8_calibration.json')),
    'fail': json.load(open('${PROJ_ROOT}/exp3b/metrics/q9q10_failure_analysis.json')),
}
s7b = {
    'abs': json.load(open('${EXP4}/metrics/q6q7_abstention.json')),
    'cal': json.load(open('${EXP4}/metrics/q8_calibration.json')),
    'fail': json.load(open('${EXP4}/metrics/q9q10_failure_analysis.json')),
}

W = 78

# ── FAILURE TAXONOMY ──
print()
print('=' * W)
print('  FAILURE TAXONOMY (Q9)')
print('=' * W)
print(f\"  {'Bucket':<40} {'1.5B-M5':>8} {'1.5B-M10':>9} {'7B-M5':>8}\")
print('  ' + '-' * (W - 2))

for key, label in [
    ('A_retrieval_failure',         'A — Retrieval failure'),
    ('B_oracle_failure',            'B — Generator failure'),
    ('C1_verifier_bad',             'C1 — Picked garbage'),
    ('C2_verifier_plausible_wrong', 'C2 — Picked plausible wrong'),
    ('D_success',                   'D — Success'),
]:
    p5  = m5['fail']['q9_failure_taxonomy'][key]['pct']
    p10 = m10['fail']['q9_failure_taxonomy'][key]['pct']
    p7b = s7b['fail']['q9_failure_taxonomy'][key]['pct']
    print(f\"  {label:<40} {p5:>8.1%} {p10:>9.1%} {p7b:>8.1%}\")

# Counts for 7B
print()
print('  7B bucket counts:')
for key, label in [
    ('A_retrieval_failure', 'A'), ('B_oracle_failure', 'B'),
    ('C1_verifier_bad', 'C1'), ('C2_verifier_plausible_wrong', 'C2'),
    ('D_success', 'D'),
]:
    n = s7b['fail']['q9_failure_taxonomy'][key]['n']
    print(f\"    {label}: {n}\")

# ── ABSTENTION ──
print()
print('=' * W)
print('  ABSTENTION (Q6/Q7)')
print('=' * W)
print(f\"  {'Metric':<40} {'1.5B-M5':>8} {'1.5B-M10':>9} {'7B-M5':>8}\")
print('  ' + '-' * (W - 2))

for src, label in [
    (lambda d: d['abs']['q6_separation']['auroc'], 'AUROC'),
    (lambda d: d['abs']['q7_coverage_curve']['em_at_100pct_coverage'], 'EM at 100% coverage'),
    (lambda d: d['abs']['q7_coverage_curve'].get('em_at_70pct_coverage', 0), 'EM at ~70% coverage'),
    (lambda d: d['abs']['q7_coverage_curve'].get('optimal_tau', 0), 'Optimal tau'),
    (lambda d: d['abs']['q7_coverage_curve'].get('em_at_optimal_tau', 0), 'EM at optimal tau'),
    (lambda d: d['abs']['q7_coverage_curve'].get('coverage_at_optimal_tau', 0), 'Coverage at optimal tau'),
]:
    v5 = src(m5)
    v10 = src(m10)
    v7b = src(s7b)
    if isinstance(v5, float) and v5 < 1:
        print(f\"  {label:<40} {v5:>8.4f} {v10:>9.4f} {v7b:>8.4f}\")
    else:
        print(f\"  {label:<40} {str(v5):>8} {str(v10):>9} {str(v7b):>8}\")

# Print coverage curve for 7B
print()
print('  7B coverage curve:')
print(f\"    {'Coverage':>10}  {'N':>6}  {'EM':>8}\")
for pt in s7b['abs']['q7_coverage_curve']['curve']:
    if pt['coverage'] in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] or abs(pt['coverage'] - 0.5) < 0.02:
        print(f\"    {pt['coverage']:>10.0%}  {pt['n']:>6}  {pt['em']:>8.4f}\")

# ── CALIBRATION ──
print()
print('=' * W)
print('  CALIBRATION (Q8)')
print('=' * W)
e5  = m5['cal']['chain_aware_verifier']['verifier_level_ece']
e10 = m10['cal']['chain_aware_verifier']['verifier_level_ece']
e7b = s7b['cal']['chain_aware_verifier']['verifier_level_ece']
print(f\"  {'Verifier-level ECE':<40} {e5:>8.4f} {e10:>9.4f} {e7b:>8.4f}\")

# ── Q10 DIFFERENTIAL ──
print()
print('=' * W)
print('  CHAIN vs FLAT NLI (Q10)')
print('=' * W)
print(f\"  {'Group':<30} {'1.5B-M5':>8} {'1.5B-M10':>9} {'7B-M5':>8}\")
print('  ' + '-' * (W - 2))
for grp in ['BOTH_CORRECT', 'CHAIN_WINS', 'FLAT_WINS', 'BOTH_WRONG']:
    n5  = m5['fail']['q10_differential'][grp]['n']
    n10 = m10['fail']['q10_differential'][grp]['n']
    n7b = s7b['fail']['q10_differential'][grp]['n']
    print(f\"  {grp:<30} {n5:>8} {n10:>9} {n7b:>8}\")
net5  = m5['fail']['q10_differential']['net_gain_chain_over_flat']
net10 = m10['fail']['q10_differential']['net_gain_chain_over_flat']
net7b = s7b['fail']['q10_differential']['net_gain_chain_over_flat']
print(f\"  {'Net gain (chain over flat)':<30} {net5:>8} {net10:>9} {net7b:>8}\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp4 analysis complete. Results in: ${EXP4}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"