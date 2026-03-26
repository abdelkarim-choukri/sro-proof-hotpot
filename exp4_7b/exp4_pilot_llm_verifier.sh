#!/usr/bin/env bash
# ==========================================================================
#  exp4_pilot_llm_verifier.sh — LLM Verifier Pilot (C2 + D questions)
#
#  Step 1: Select 188 C2 + 112 D = 300 pilot questions
#  Step 2: Run Option 1 (scores in prompt, all candidates)
#  Step 3: Run Option 2 (pre-filter top 2, then LLM)
#  Step 4: Compare both options + XGB baseline
#
#  PREREQ: vLLM serving 7B on port 8000
#  Estimated runtime: ~20-30 min (300 questions × 2 options)
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"
EXP4="${PROJ_ROOT}/exp4_7b"
PILOT_DIR="${EXP4}/pilot"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

echo "================================================================"
echo "  LLM Verifier Pilot: 300 Questions (188 C2 + 112 D)"
echo "================================================================"
echo ""

mkdir -p "${PILOT_DIR}"

# ==========================================================================
#  Step 1: Select pilot questions
# ==========================================================================
echo "━━━ Step 1: Select pilot questions ━━━"

if [[ -f "${PILOT_DIR}/pilot_questions.jsonl" ]]; then
    echo "  Already selected — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp4_pilot_select.py \
        --chain_preds   "${EXP4}/preds/dev_chain_verifier_mean_preds.jsonl" \
        --oracle_perqid "${EXP4}/metrics/oracle_M5_dev_perqid.jsonl" \
        --hop_scores    "${EXP4}/preds/dev_hop_scores.jsonl" \
        --evidence      "${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl" \
        --gold          "${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json" \
        --out_jsonl     "${PILOT_DIR}/pilot_questions.jsonl" \
        --out_summary   "${PILOT_DIR}/pilot_summary.json" \
        --n_d_questions 112 \
        --seed 42
fi

echo "  ✓ Pilot questions ready"
echo ""

# ==========================================================================
#  Step 2: Option 1 — Scores in prompt (all candidates)
# ==========================================================================
echo "━━━ Step 2: Option 1 — Scores in Prompt ━━━"

${PYTHON} ${TOOLS_DIR}/exp4_llm_verifier.py \
    --pilot       "${PILOT_DIR}/pilot_questions.jsonl" \
    --mode        scores_in_prompt \
    --out_preds   "${PILOT_DIR}/preds_option1.jsonl" \
    --llm_base_url http://127.0.0.1:8000/v1 \
    --llm_model_id "${MODEL_ID}" \
    --temperature 0.0 \
    --max_new_tokens 512 \
    --resume \
    2>&1 | tee "${EXP4}/logs/pilot_option1.log"

echo ""

# ==========================================================================
#  Step 3: Option 2 — Pre-filter top 2
# ==========================================================================
echo "━━━ Step 3: Option 2 — Pre-filter Top 2 ━━━"

${PYTHON} ${TOOLS_DIR}/exp4_llm_verifier.py \
    --pilot       "${PILOT_DIR}/pilot_questions.jsonl" \
    --mode        prefilter \
    --out_preds   "${PILOT_DIR}/preds_option2.jsonl" \
    --llm_base_url http://127.0.0.1:8000/v1 \
    --llm_model_id "${MODEL_ID}" \
    --temperature 0.0 \
    --max_new_tokens 512 \
    --resume \
    2>&1 | tee "${EXP4}/logs/pilot_option2.log"

echo ""

# ==========================================================================
#  Step 4: Compare
# ==========================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  HEAD-TO-HEAD: XGB vs Option 1 vs Option 2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json, re, string, collections

def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred, gold):
    return int(normalize(pred) == normalize(gold))

# Load pilot
pilot = {}
for line in open('${PILOT_DIR}/pilot_questions.jsonl'):
    r = json.loads(line)
    pilot[r['qid']] = r

# Load preds
def load_preds(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r['qid']] = r
    return out

opt1 = load_preds('${PILOT_DIR}/preds_option1.jsonl')
opt2 = load_preds('${PILOT_DIR}/preds_option2.jsonl')

W = 70
print()
print('=' * W)

# Per-bucket results
for bucket_name in ['C2', 'D', 'ALL']:
    xgb_em = o1_em = o2_em = 0
    n = 0
    for qid, rec in pilot.items():
        if bucket_name != 'ALL' and rec['bucket'] != bucket_name:
            continue
        gold = rec['gold']
        n += 1
        xgb_em += em(rec['xgb_pred'], gold)
        if qid in opt1:
            o1_em += em(opt1[qid]['pred'], gold)
        if qid in opt2:
            o2_em += em(opt2[qid]['pred'], gold)

    print(f'  {bucket_name} questions (n={n}):')
    print(f\"    {'Method':<35} {'EM':>6} {'Rate':>8}\")
    print(f'    ' + '-' * (W - 6))
    print(f\"    {'XGB baseline':<35} {xgb_em:>6} {xgb_em/max(n,1):>8.1%}\")
    print(f\"    {'Option 1 (scores in prompt)':<35} {o1_em:>6} {o1_em/max(n,1):>8.1%}\")
    print(f\"    {'Option 2 (pre-filter top 2)':<35} {o2_em:>6} {o2_em/max(n,1):>8.1%}\")
    print()

# C2 recovery detail
print('-' * W)
c2_qids = [q for q, r in pilot.items() if r['bucket'] == 'C2']
d_qids  = [q for q, r in pilot.items() if r['bucket'] == 'D']

for label, preds in [('Option 1', opt1), ('Option 2', opt2)]:
    c2_recovered = sum(em(preds[q]['pred'], pilot[q]['gold'])
                       for q in c2_qids if q in preds)
    d_lost = sum(1 - em(preds[q]['pred'], pilot[q]['gold'])
                 for q in d_qids if q in preds)
    net = c2_recovered - d_lost
    print(f'  {label}: C2 recovered={c2_recovered}/{len(c2_qids)}  '
          f'D lost={d_lost}/{len(d_qids)}  net={net:+d}')

# Gate decision
print()
best_c2 = 0
best_label = ''
for label, preds in [('Option 1', opt1), ('Option 2', opt2)]:
    c2_rec = sum(em(preds[q]['pred'], pilot[q]['gold'])
                 for q in c2_qids if q in preds)
    if c2_rec > best_c2:
        best_c2 = c2_rec
        best_label = label

c2_rate = best_c2 / max(len(c2_qids), 1)
print(f'  Best C2 recovery: {best_label} with {best_c2}/{len(c2_qids)} ({c2_rate:.1%})')
print()

if c2_rate >= 0.40:
    print(f'  → GATE PASSED: {c2_rate:.1%} ≥ 40%. Scale to full dev set.')
elif c2_rate >= 0.30:
    print(f'  → GATE PASSED (marginal): {c2_rate:.1%} ≥ 30%. Worth scaling.')
elif c2_rate >= 0.15:
    print(f'  → MODEST: {c2_rate:.1%}. Some C2 recovery but limited.')
    print(f'    Consider cost/benefit before scaling.')
else:
    print(f'  → GATE FAILED: {c2_rate:.1%} < 15%. LLM verifier does not help.')
    print(f'    The evidence is genuinely ambiguous for these questions.')

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pilot complete. Results in: ${PILOT_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"