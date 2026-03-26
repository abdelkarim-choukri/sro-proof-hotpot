#!/usr/bin/env bash
# ==========================================================================
#  exp3b_7b_sample_run.sh — 7B Generator Sample Test (300 Bucket B Questions)
#
#  Goal: Does Qwen2.5-7B-Instruct produce correct answers on questions
#        where Qwen2.5-1.5B-Instruct failed (Bucket B)?
#
#  Steps:
#    1. Select 300 Bucket B questions (CPU, instant)
#    2. Generate M=5 candidates with 7B (GPU, ~10 min for 300 questions)
#    3. Also generate M=5 with 1.5B on same 300 for fair comparison
#    4. Compute oracle@5 for both and compare
#
#  Decision gate:
#    7B oracle@5 on these 300 ≥ 0.25  →  7B is clearly better, scale to full
#    7B oracle@5 on these 300 ≥ 0.15  →  modest improvement, worth investigating
#    7B oracle@5 on these 300 < 0.10  →  even 7B can't crack these, not worth it
#
#  NOTE: These 300 questions have oracle@5=0.0 for 1.5B (that's how they were
#  selected — Bucket B means the 1.5B never generated the correct answer).
#  So any non-zero oracle for 7B is pure gain.
#
#  IMPORTANT: vLLM must be started separately BEFORE running this script.
#  See instructions below.
# ==========================================================================

set -euo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Inputs
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
ORACLE_M10="${PROJ_ROOT}/exp3b/metrics/oracle_M10_dev_perqid.jsonl"
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"

# 7B experiment directory
EXP_DIR="${PROJ_ROOT}/exp3b_7b"

echo "================================================================"
echo "  7B Generator Sample Test (300 Bucket B Questions)"
echo "================================================================"
echo ""

# ==========================================================================
#  Step 1: Select 300 Bucket B questions
# ==========================================================================
echo "━━━ Step 1: Select 300 Bucket B questions ━━━"

if [[ -f "${EXP_DIR}/sample_qids.json" ]]; then
    echo "  Sample already selected — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp3b_7b_sample_select.py \
        --oracle_perqid   "${ORACLE_M10}" \
        --evidence        "${EVIDENCE}" \
        --gold            "${GOLD}" \
        --out_qids        "${EXP_DIR}/sample_qids.json" \
        --out_evidence    "${EXP_DIR}/evidence/sample_300_chains.jsonl" \
        --n_sample        300 \
        --seed            42
fi

SAMPLE_EVIDENCE="${EXP_DIR}/evidence/sample_300_chains.jsonl"
echo "  ✓ Sample: 300 questions in ${SAMPLE_EVIDENCE}"
echo ""

# ==========================================================================
#  Step 2: Generate M=5 with 7B model
# ==========================================================================
echo "━━━ Step 2: Generate M=5 candidates with 7B ━━━"
echo ""
echo "  IMPORTANT: vLLM must be running with the 7B model."
echo "  If not already running, start it in another terminal:"
echo ""
echo "    CUDA_VISIBLE_DEVICES=0,1 /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\"
echo "        -m vllm.entrypoints.openai.api_server \\"
echo "        --model /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \\"
echo "        --port 8000 \\"
echo "        --dtype auto \\"
echo "        --tensor-parallel-size 2 \\"
echo "        --max-model-len 4096"
echo ""
echo "  Wait for 'Uvicorn running on http://0.0.0.0:8000' then press Enter..."
read -r

CANDS_7B="${EXP_DIR}/candidates/sample_M5_7b.jsonl"
mkdir -p "${EXP_DIR}/candidates" "${EXP_DIR}/metrics" "${EXP_DIR}/logs"

if [[ -f "${CANDS_7B}" ]] && [[ $(wc -l < "${CANDS_7B}") -ge 300 ]]; then
    echo "  7B candidates already generated — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp3_generate_candidates.py \
        --evidence        "${SAMPLE_EVIDENCE}" \
        --gold            "${GOLD}" \
        --prompt_file     "${PROMPT_V2}" \
        --prompt_version  v2 \
        --evidence_format flat \
        --out_jsonl       "${CANDS_7B}" \
        --manifest        "${EXP_DIR}/manifest.json" \
        --llm_base_url    http://127.0.0.1:8000/v1 \
        --llm_model_id    qwen2.5-7b-instruct \
        --split dev \
        --m 5 \
        --seed 12345 \
        --resume \
        2>&1 | tee "${EXP_DIR}/logs/generate_7b.log"
fi

echo "  ✓ 7B candidates: ${CANDS_7B}"
echo ""

# ==========================================================================
#  Step 3: Generate M=5 with 1.5B on same 300 (fair comparison)
# ==========================================================================
echo "━━━ Step 3: Generate M=5 with 1.5B (same 300 questions) ━━━"
echo ""
echo "  NOTE: You need to restart vLLM with the 1.5B model for this step."
echo "  Kill the 7B server (Ctrl+C), then start 1.5B:"
echo ""
echo "    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\"
echo "        -m vllm.entrypoints.openai.api_server \\"
echo "        --model /var/tmp/u24sf51014/sro/models/qwen2.5-1.5b-instruct \\"
echo "        --port 8000 \\"
echo "        --dtype auto \\"
echo "        --max-model-len 4096"
echo ""
echo "  Wait for 'Uvicorn running' then press Enter..."
read -r

CANDS_1_5B="${EXP_DIR}/candidates/sample_M5_1.5b.jsonl"

if [[ -f "${CANDS_1_5B}" ]] && [[ $(wc -l < "${CANDS_1_5B}") -ge 300 ]]; then
    echo "  1.5B candidates already generated — skipping"
else
    ${PYTHON} ${TOOLS_DIR}/exp3_generate_candidates.py \
        --evidence        "${SAMPLE_EVIDENCE}" \
        --gold            "${GOLD}" \
        --prompt_file     "${PROMPT_V2}" \
        --prompt_version  v2 \
        --evidence_format flat \
        --out_jsonl       "${CANDS_1_5B}" \
        --manifest        "${EXP_DIR}/manifest.json" \
        --llm_base_url    http://127.0.0.1:8000/v1 \
        --llm_model_id    qwen2.5-1.5b-instruct \
        --split dev \
        --m 5 \
        --seed 12345 \
        --resume \
        2>&1 | tee "${EXP_DIR}/logs/generate_1.5b.log"
fi

echo "  ✓ 1.5B candidates: ${CANDS_1_5B}"
echo ""

# ==========================================================================
#  Step 4: Compute oracle@5 for both and compare
# ==========================================================================
echo "━━━ Step 4: Oracle@5 Comparison ━━━"

ORACLE_7B="${EXP_DIR}/metrics/oracle_7b.json"
ORACLE_7B_PERQID="${EXP_DIR}/metrics/oracle_7b_perqid.jsonl"
ORACLE_1_5B="${EXP_DIR}/metrics/oracle_1.5b.json"
ORACLE_1_5B_PERQID="${EXP_DIR}/metrics/oracle_1.5b_perqid.jsonl"

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence    "${SAMPLE_EVIDENCE}" \
    --candidates  "${CANDS_7B}" \
    --gold        "${GOLD}" \
    --split dev --m 5 \
    --out_json    "${ORACLE_7B}" \
    --out_jsonl   "${ORACLE_7B_PERQID}" \
    --out_sha256  "${EXP_DIR}/metrics/oracle_7b.json.sha256" \
    --manifest    "${EXP_DIR}/manifest.json" \
    2>&1 | tee "${EXP_DIR}/logs/oracle_7b.log"

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence    "${SAMPLE_EVIDENCE}" \
    --candidates  "${CANDS_1_5B}" \
    --gold        "${GOLD}" \
    --split dev --m 5 \
    --out_json    "${ORACLE_1_5B}" \
    --out_jsonl   "${ORACLE_1_5B_PERQID}" \
    --out_sha256  "${EXP_DIR}/metrics/oracle_1.5b.json.sha256" \
    --manifest    "${EXP_DIR}/manifest.json" \
    2>&1 | tee "${EXP_DIR}/logs/oracle_1.5b.log"

# ==========================================================================
#  Step 5: Decision gate
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS: 7B vs 1.5B on 300 Bucket B Questions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${PYTHON} -c "
import json

o7b  = json.load(open('${ORACLE_7B}'))
o15b = json.load(open('${ORACLE_1_5B}'))

em_7b  = o7b['overall']['oracle_em']
f1_7b  = o7b['overall']['oracle_f1']
em_15b = o15b['overall']['oracle_em']
f1_15b = o15b['overall']['oracle_f1']
n      = o7b['overall']['n']

# Per-question comparison
o7b_pq  = {}
for line in open('${ORACLE_7B_PERQID}'):
    r = json.loads(line)
    o7b_pq[r['qid']] = r['best_em']

o15b_pq = {}
for line in open('${ORACLE_1_5B_PERQID}'):
    r = json.loads(line)
    o15b_pq[r['qid']] = r['best_em']

wins_7b = sum(1 for q in o7b_pq if o7b_pq[q] > o15b_pq.get(q, 0))
wins_15b = sum(1 for q in o15b_pq if o15b_pq.get(q, 0) > o7b_pq.get(q, 0))
both_zero = sum(1 for q in o7b_pq if o7b_pq[q] == 0 and o15b_pq.get(q, 0) == 0)

W = 68
print()
print('=' * W)
print(f'  These {n} questions are ALL Bucket B for 1.5B M=10')
print(f'  (1.5B never produced the correct answer in 10 tries)')
print('=' * W)
print(f\"  {'Model':<30} {'Oracle EM':>10} {'Oracle F1':>10}\")
print('  ' + '-' * (W - 2))
print(f\"  {'Qwen2.5-1.5B (M=5, fresh)':<30} {em_15b:>10.4f} {f1_15b:>10.4f}\")
print(f\"  {'Qwen2.5-7B (M=5)':<30} {em_7b:>10.4f} {f1_7b:>10.4f}\")
print(f\"  {'Delta':<30} {em_7b - em_15b:>+10.4f} {f1_7b - f1_15b:>+10.4f}\")
print('=' * W)

print(f\"\n  Per-question breakdown:\")
print(f\"    7B wins (7B correct, 1.5B wrong):  {wins_7b}\")
print(f\"    1.5B wins (1.5B correct, 7B wrong): {wins_15b}\")
print(f\"    Both still wrong:                   {both_zero}\")
print(f\"    Total:                              {n}\")

# Decision
print()
if em_7b >= 0.25:
    print(f\"  → STRONG: oracle@5 = {em_7b:.1%} on Bucket B questions.\")
    print(f\"    7B cracks {wins_7b}/{n} questions that 1.5B couldn't.\")
    print(f\"    Scale to full dev set — expected overall oracle lift: ~{wins_7b/n * 0.454:.1%}\")
    print(f\"    (0.454 = Bucket B fraction of all 7405 questions)\")
elif em_7b >= 0.15:
    print(f\"  → MODEST: oracle@5 = {em_7b:.1%}. Some improvement but not transformative.\")
    print(f\"    7B helps on {wins_7b}/{n} questions. Consider cost/benefit.\")
elif em_7b >= 0.05:
    print(f\"  → MARGINAL: oracle@5 = {em_7b:.1%}. 7B barely helps on Bucket B.\")
    print(f\"    The problem may be evidence quality, not model capacity.\")
else:
    print(f\"  → NO HELP: oracle@5 = {em_7b:.1%}. Even 7B can't answer these.\")
    print(f\"    These questions are fundamentally hard — not a model capacity issue.\")

print()
print('=' * W)
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  7B sample test complete."
echo "  Results in: ${EXP_DIR}/metrics/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"