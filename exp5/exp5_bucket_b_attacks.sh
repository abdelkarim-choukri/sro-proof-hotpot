#!/usr/bin/env bash
# ==========================================================================
#  exp5_bucket_b_attacks.sh — Three Bucket B Experiments
#
#  Exp5a: Chain-aware prompting (hop-structured prompt for 7B)
#  Exp5b: Higher temperature fallback (T=1.0 for seed+i cases)
#  Exp5c: Chain-aware + high-temp combined
#
#  All use same model (7B), same evidence (K=100), M=5.
#  Each measures oracle@5 and compares to Exp4 baseline (48.8%).
#
#  PREREQ: vLLM serving 7B on port 8000
#  Estimated runtime: ~3 hrs per experiment (3 × ~3 hrs total)
#  Recommendation: run Exp5a first, check result, then decide on 5b/5c
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

# Exp4 baseline
EXP4_ORACLE=0.4879

echo "================================================================"
echo "  Exp5: Bucket B Attacks (Generator Improvement)"
echo "================================================================"
echo "  Exp4 baseline oracle@5 = ${EXP4_ORACLE}"
echo "  vLLM must be serving 7B on port 8000"
echo ""

# ══════════════════════════════════════════════════════════════
#  Exp5a: Chain-Aware Prompting
# ══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp5a: Chain-Aware Prompting (7B)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXP5A="${PROJ_ROOT}/exp5a"
CANDS_5A="${EXP5A}/candidates/dev_M5_7b_chain_aware.jsonl"
mkdir -p "${EXP5A}/candidates" "${EXP5A}/metrics" "${EXP5A}/logs" "${EXP5A}/inputs"

# Copy prompt if not there
if [[ ! -f "${EXP5A}/inputs/prompt_v4_chain_aware_7b.txt" ]]; then
    cp "${PROJ_ROOT}/exp5/inputs/prompt_v4_chain_aware_7b.txt" \
       "${EXP5A}/inputs/prompt_v4_chain_aware_7b.txt" 2>/dev/null || true
fi

echo "  Generating M=5 with chain-aware prompt..."
if [[ -f "${CANDS_5A}" ]] && [[ $(wc -l < "${CANDS_5A}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping generation"
else
    ${PYTHON} ${TOOLS_DIR}/exp5_generate_candidates.py \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --prompt_file   "${EXP5A}/inputs/prompt_v4_chain_aware_7b.txt" \
        --prompt_mode   chain_aware \
        --out_jsonl     "${CANDS_5A}" \
        --manifest      "${EXP5A}/manifest.json" \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  "${MODEL_ID}" \
        --split dev --m 5 --seed 12345 \
        --temperature 0.7 \
        --resume \
        2>&1 | tee "${EXP5A}/logs/generate.log"
fi

echo "  Computing oracle@5..."
${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence "${EVIDENCE}" --candidates "${CANDS_5A}" --gold "${GOLD}" \
    --split dev --m 5 \
    --out_json "${EXP5A}/metrics/oracle_M5.json" \
    --out_jsonl "${EXP5A}/metrics/oracle_M5_perqid.jsonl" \
    --out_sha256 "${EXP5A}/metrics/oracle_M5.sha256" \
    --manifest "${EXP5A}/manifest.json" \
    2>&1 | tee "${EXP5A}/logs/oracle.log"

echo ""

# ══════════════════════════════════════════════════════════════
#  Exp5b: Higher Temperature Fallback (flat prompt, T_fallback=1.0)
# ══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp5b: High-Temperature Fallback (T=1.0, flat prompt)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXP5B="${PROJ_ROOT}/exp5b"
CANDS_5B="${EXP5B}/candidates/dev_M5_7b_hightemp.jsonl"
mkdir -p "${EXP5B}/candidates" "${EXP5B}/metrics" "${EXP5B}/logs"

echo "  Generating M=5 with flat prompt, fallback T=1.0..."
if [[ -f "${CANDS_5B}" ]] && [[ $(wc -l < "${CANDS_5B}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping generation"
else
    ${PYTHON} ${TOOLS_DIR}/exp5_generate_candidates.py \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --prompt_file   "${PROJ_ROOT}/exp1/inputs/prompt_v2.txt" \
        --prompt_mode   flat \
        --out_jsonl     "${CANDS_5B}" \
        --manifest      "${EXP5B}/manifest.json" \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  "${MODEL_ID}" \
        --split dev --m 5 --seed 12345 \
        --temperature 0.7 \
        --fallback_temperature 1.0 \
        --resume \
        2>&1 | tee "${EXP5B}/logs/generate.log"
fi

echo "  Computing oracle@5..."
${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence "${EVIDENCE}" --candidates "${CANDS_5B}" --gold "${GOLD}" \
    --split dev --m 5 \
    --out_json "${EXP5B}/metrics/oracle_M5.json" \
    --out_jsonl "${EXP5B}/metrics/oracle_M5_perqid.jsonl" \
    --out_sha256 "${EXP5B}/metrics/oracle_M5.sha256" \
    --manifest "${EXP5B}/manifest.json" \
    2>&1 | tee "${EXP5B}/logs/oracle.log"

echo ""

# ══════════════════════════════════════════════════════════════
#  Exp5c: Chain-Aware + High-Temp Combined
# ══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp5c: Chain-Aware + High-Temp Combined"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXP5C="${PROJ_ROOT}/exp5c"
CANDS_5C="${EXP5C}/candidates/dev_M5_7b_chain_hightemp.jsonl"
mkdir -p "${EXP5C}/candidates" "${EXP5C}/metrics" "${EXP5C}/logs" "${EXP5C}/inputs"

if [[ ! -f "${EXP5C}/inputs/prompt_v4_chain_aware_7b.txt" ]]; then
    cp "${PROJ_ROOT}/exp5/inputs/prompt_v4_chain_aware_7b.txt" \
       "${EXP5C}/inputs/prompt_v4_chain_aware_7b.txt" 2>/dev/null || true
fi

echo "  Generating M=5 with chain-aware prompt + fallback T=1.0..."
if [[ -f "${CANDS_5C}" ]] && [[ $(wc -l < "${CANDS_5C}") -ge 7405 ]]; then
    echo "  ✓ Already complete — skipping generation"
else
    ${PYTHON} ${TOOLS_DIR}/exp5_generate_candidates.py \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --prompt_file   "${EXP5C}/inputs/prompt_v4_chain_aware_7b.txt" \
        --prompt_mode   chain_aware \
        --out_jsonl     "${CANDS_5C}" \
        --manifest      "${EXP5C}/manifest.json" \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  "${MODEL_ID}" \
        --split dev --m 5 --seed 12345 \
        --temperature 0.7 \
        --fallback_temperature 1.0 \
        --resume \
        2>&1 | tee "${EXP5C}/logs/generate.log"
fi

echo "  Computing oracle@5..."
${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence "${EVIDENCE}" --candidates "${CANDS_5C}" --gold "${GOLD}" \
    --split dev --m 5 \
    --out_json "${EXP5C}/metrics/oracle_M5.json" \
    --out_jsonl "${EXP5C}/metrics/oracle_M5_perqid.jsonl" \
    --out_sha256 "${EXP5C}/metrics/oracle_M5.sha256" \
    --manifest "${EXP5C}/manifest.json" \
    2>&1 | tee "${EXP5C}/logs/oracle.log"

# ══════════════════════════════════════════════════════════════
#  Comparison
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  COMPARISON: Oracle@5 Across All Generator Configurations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

configs = {
    '1.5B flat M=5 (Exp1b)':     {'oracle_em': 0.3982, 'oracle_f1': 0.5350},
    '1.5B flat M=10 (Exp3b)':    {'oracle_em': 0.4344, 'oracle_f1': 0.5718},
    '7B flat M=5 (Exp4)':        {'oracle_em': ${EXP4_ORACLE}, 'oracle_f1': 0.6041},
}

for label, path in [
    ('7B chain-aware (Exp5a)',     '${EXP5A}/metrics/oracle_M5.json'),
    ('7B flat+highT fb (Exp5b)',   '${EXP5B}/metrics/oracle_M5.json'),
    ('7B chain+highT fb (Exp5c)',  '${EXP5C}/metrics/oracle_M5.json'),
]:
    try:
        r = json.load(open(path))
        configs[label] = r['overall']
    except:
        configs[label] = {'oracle_em': 0, 'oracle_f1': 0}

W = 74
print()
print('=' * W)
print(f\"  {'Configuration':<35} {'Oracle EM':>10} {'Oracle F1':>10} {'vs Exp4':>10}\")
print('-' * W)

exp4_em = ${EXP4_ORACLE}
for label, m in configs.items():
    em_v = m.get('oracle_em', 0)
    f1_v = m.get('oracle_f1', 0)
    delta = em_v - exp4_em if em_v > 0 else 0
    d_str = f'{delta:+.4f}' if em_v > 0 else '—'
    marker = ' ★' if delta > 0.01 else ''
    print(f\"  {label:<35} {em_v:>10.4f} {f1_v:>10.4f} {d_str:>10}{marker}\")

print('=' * W)

# Fallback rate comparison
print()
print('  Generation stats:')
for label, path in [
    ('Exp5a (chain-aware)',    '${EXP5A}/manifest.json'),
    ('Exp5b (high-temp fb)',   '${EXP5B}/manifest.json'),
    ('Exp5c (chain+highT)',    '${EXP5C}/manifest.json'),
]:
    try:
        m = json.load(open(path))
        d = m.get('exp5_generate_done', {})
        n5 = d.get('n5_ok', 0)
        fb = d.get('fallback', 0)
        total = n5 + fb
        fb_rate = fb / total if total else 0
        print(f\"    {label:<30} n5_ok={n5}  fallback={fb}  ({fb_rate:.1%} fallback)\")
    except:
        print(f\"    {label:<30} (not available)\")

# Decision
print()
best_label = max(configs, key=lambda k: configs[k].get('oracle_em', 0))
best_em = configs[best_label]['oracle_em']
if best_em > exp4_em + 0.015:
    print(f\"  → PROCEED: {best_label} lifts oracle by +{best_em - exp4_em:.4f}.\")
    print(f\"    Run full verification pipeline on the winning candidates.\")
elif best_em > exp4_em + 0.005:
    print(f\"  → MODEST: {best_label} gains +{best_em - exp4_em:.4f}.\")
    print(f\"    Worth running pipeline but gains may be small.\")
else:
    print(f\"  → NO IMPROVEMENT: best is {best_label} at {best_em:.4f}.\")
    print(f\"    Generator prompt/temperature changes don't help the 7B model.\")
    print(f\"    Next lever: K=200 retrieval (Bucket A attack).\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp5 complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"