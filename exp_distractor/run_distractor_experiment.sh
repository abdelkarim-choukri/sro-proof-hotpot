#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  run_distractor_experiment.sh — Complete Distractor-Setting Experiment
# ══════════════════════════════════════════════════════════════════════
#
#  WHAT THIS DOES:
#    Runs the full pipeline in the distractor setting:
#      Step 0: Prepare evidence (convert distractor format → pipeline format)
#      Step 1: Start vLLM with 7B model (both GPUs, tensor-parallel=2)
#      Step 2: Generate M=1 greedy answers (~1.5 hours)
#      Step 3: Generate M=5 sampled answers (~4-5 hours)
#      Step 4: Kill vLLM, free GPUs for scoring
#      Step 5: NLI hop scoring (GPU, ~15 min)
#      Step 6: QA cross-encoder hop scoring (GPU, ~35 min)
#      Step 7: Lexical features (CPU, ~10 min)
#      Step 8: Oracle computation
#      Step 9: Phase 0 ablations + bootstrap on distractor results
#
#  WHY:
#    The distractor setting eliminates Bucket A (retrieval failure = 0%).
#    Gold paragraphs are ALWAYS present in the 10 context paragraphs.
#    This lets us fairly compare against extractive SOTA that also gets gold.
#
#  PAPER TABLE (what this produces):
#    Row 1: M=1 greedy (comparable to extractive SOTA)
#    Row 2: M=5 majority vote (self-consistency, cite Wang et al.)
#    Row 3: M=5 chain-only verifier (YOUR contribution, no answer_freq)
#    Row 4: M=5 full verifier (chain + surface combined)
#    Row 5: Oracle@5 (ceiling)
#
#  2 GPUs REQUIRED:
#    Both GPUs are used together for vLLM tensor parallelism (Steps 2-3).
#    After generation, one GPU is used for NLI/QA scoring (Steps 5-6).
#
#  TOTAL TIME: ~7-8 hours (dominated by generation)
#
#  USAGE:
#    # Run each step individually (recommended — so you can monitor):
#    bash exp_distractor/run_distractor_experiment.sh step0   # ~30 sec
#    bash exp_distractor/run_distractor_experiment.sh step1   # starts vLLM
#    bash exp_distractor/run_distractor_experiment.sh step2   # ~1.5 hours
#    bash exp_distractor/run_distractor_experiment.sh step3   # ~4-5 hours
#    bash exp_distractor/run_distractor_experiment.sh step4   # kill vLLM
#    bash exp_distractor/run_distractor_experiment.sh step5   # ~15 min
#    bash exp_distractor/run_distractor_experiment.sh step6   # ~35 min
#    bash exp_distractor/run_distractor_experiment.sh step7   # ~10 min
#    bash exp_distractor/run_distractor_experiment.sh step8   # ~1 min
#    bash exp_distractor/run_distractor_experiment.sh step9   # ~20 min
#
#    # Or run everything in sequence:
#    bash exp_distractor/run_distractor_experiment.sh all
#
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──
PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Input files
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"
QA_MODEL="/var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2"
MODEL_7B="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

# Output directory
EXP="${PROJ_ROOT}/exp_distractor"

# Derived paths
EVIDENCE="${EXP}/evidence/dev_distractor_chains.jsonl"
CANDS_M1="${EXP}/candidates/dev_M1_greedy.jsonl"
CANDS_M5="${EXP}/candidates/dev_M5_sampling.jsonl"

# Monolithic fallback preds (from MDR pipeline, for Phase 0 fallback)
MONO_PREDS="${PROJ_ROOT}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"

STEP="${1:-help}"

# ── Create directories ──
mkdir -p "${EXP}/evidence" "${EXP}/candidates" "${EXP}/preds" \
         "${EXP}/metrics" "${EXP}/logs" "${EXP}/results"


# ══════════════════════════════════════════════════════════════════════
step0() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 0: Prepare Distractor Evidence ━━━"
    echo "  Converts HotpotQA distractor format → pipeline evidence JSONL"
    echo "  Identifies hop1/hop2 from supporting_facts for NLI/QA/lex scoring"
    echo ""

    ${PYTHON} "${TOOLS_DIR}/distractor_prepare_evidence.py" \
        --gold          "${GOLD}" \
        --out_evidence  "${EVIDENCE}" \
        --out_stats     "${EXP}/evidence/prep_stats.json"

    echo ""
    echo "  ✓ Evidence ready: ${EVIDENCE}"
    echo "  ✓ Stats: ${EXP}/evidence/prep_stats.json"
    echo ""
    echo "  Next: bash $0 step1  (start vLLM)"
}


# ══════════════════════════════════════════════════════════════════════
step1() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 1: Start vLLM Server ━━━"
    echo ""
    echo "  Run this in ANOTHER terminal (it needs to stay running):"
    echo ""
    echo "    CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} \\"
    echo "        -m vllm.entrypoints.openai.api_server \\"
    echo "        --model ${MODEL_7B} \\"
    echo "        --port 8000 \\"
    echo "        --dtype auto \\"
    echo "        --tensor-parallel-size 2 \\"
    echo "        --max-model-len 4096"
    echo ""
    echo "  Wait for 'Uvicorn running on http://0.0.0.0:8000'"
    echo "  Then run: bash $0 step2"
    echo ""

    # Quick health check
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "  ✓ vLLM already running on port 8000"
        echo "  Ready for step2"
    else
        echo "  ✗ vLLM not detected on port 8000"
        echo "  Please start it as shown above"
    fi
}


# ══════════════════════════════════════════════════════════════════════
step2() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 2: Generate M=1 Greedy (baseline for extractive comparison) ━━━"
    echo "  Temperature: 0.0 (pure greedy — single deterministic answer)"
    echo "  This row is directly comparable to extractive SOTA systems"
    echo "  Estimated time: ~1.5 hours"
    echo ""

    # Verify vLLM is running
    if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "  ERROR: vLLM not running on port 8000. Run step1 first."
        exit 1
    fi

    if [[ -f "${CANDS_M1}" ]] && [[ $(wc -l < "${CANDS_M1}") -ge 7405 ]]; then
        echo "  ✓ Already complete ($(wc -l < "${CANDS_M1}") lines) — skipping"
    else
        ${PYTHON} "${TOOLS_DIR}/distractor_generate.py" \
            --evidence      "${EVIDENCE}" \
            --gold          "${GOLD}" \
            --prompt_file   "${PROMPT_V2}" \
            --out_jsonl     "${CANDS_M1}" \
            --manifest      "${EXP}/manifest_m1.json" \
            --llm_base_url  http://127.0.0.1:8000/v1 \
            --llm_model_id  "${MODEL_7B}" \
            --m 1 \
            --temperature 0.0 \
            --top_p 1.0 \
            --seed 12345 \
            --resume \
            2>&1 | tee "${EXP}/logs/generate_m1.log"
    fi

    echo ""
    echo "  ✓ M=1 candidates: ${CANDS_M1}"
    echo ""

    # Quick oracle check
    echo "  Computing M=1 oracle (= EM since there's only 1 candidate):"
    ${PYTHON} "${TOOLS_DIR}/exp1_compute_oracle.py" \
        --evidence    "${EVIDENCE}" \
        --candidates  "${CANDS_M1}" \
        --gold        "${GOLD}" \
        --split dev --m 1 \
        --out_json    "${EXP}/metrics/oracle_M1.json" \
        --out_jsonl   "${EXP}/metrics/oracle_M1_perqid.jsonl" \
        --out_sha256  "${EXP}/metrics/oracle_M1.sha256" \
        --manifest    "${EXP}/manifest_m1.json" \
        2>&1 | tee "${EXP}/logs/oracle_m1.log"

    echo ""
    echo "  Next: bash $0 step3  (generate M=5, ~4-5 hours)"
}


# ══════════════════════════════════════════════════════════════════════
step3() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 3: Generate M=5 Sampling (for verification experiments) ━━━"
    echo "  Temperature: 0.7, top_p: 0.95 (same as MDR pipeline)"
    echo "  This produces the candidate pool for majority voting + verification"
    echo "  Estimated time: ~4-5 hours"
    echo ""

    # Verify vLLM is running
    if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "  ERROR: vLLM not running on port 8000."
        exit 1
    fi

    if [[ -f "${CANDS_M5}" ]] && [[ $(wc -l < "${CANDS_M5}") -ge 7405 ]]; then
        echo "  ✓ Already complete ($(wc -l < "${CANDS_M5}") lines) — skipping"
    else
        ${PYTHON} "${TOOLS_DIR}/distractor_generate.py" \
            --evidence      "${EVIDENCE}" \
            --gold          "${GOLD}" \
            --prompt_file   "${PROMPT_V2}" \
            --out_jsonl     "${CANDS_M5}" \
            --manifest      "${EXP}/manifest_m5.json" \
            --llm_base_url  http://127.0.0.1:8000/v1 \
            --llm_model_id  "${MODEL_7B}" \
            --m 5 \
            --temperature 0.7 \
            --top_p 0.95 \
            --seed 12345 \
            --resume \
            2>&1 | tee "${EXP}/logs/generate_m5.log"
    fi

    echo ""
    echo "  ✓ M=5 candidates: ${CANDS_M5}"

    # Oracle@5
    echo ""
    echo "  Computing oracle@5:"
    ${PYTHON} "${TOOLS_DIR}/exp1_compute_oracle.py" \
        --evidence    "${EVIDENCE}" \
        --candidates  "${CANDS_M5}" \
        --gold        "${GOLD}" \
        --split dev --m 5 \
        --out_json    "${EXP}/metrics/oracle_M5.json" \
        --out_jsonl   "${EXP}/metrics/oracle_M5_perqid.jsonl" \
        --out_sha256  "${EXP}/metrics/oracle_M5.sha256" \
        --manifest    "${EXP}/manifest_m5.json" \
        2>&1 | tee "${EXP}/logs/oracle_m5.log"

    echo ""
    echo "  Next: bash $0 step4  (kill vLLM to free GPUs for scoring)"
}


# ══════════════════════════════════════════════════════════════════════
step4() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 4: Kill vLLM Server ━━━"
    echo ""

    if pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; then
        echo "  ✓ vLLM server stopped"
    else
        echo "  vLLM was not running (or already stopped)"
    fi

    sleep 3
    echo "  GPUs freed for NLI/QA scoring"
    echo ""
    echo "  Next: bash $0 step5  (NLI hop scoring)"
}


# ══════════════════════════════════════════════════════════════════════
step5() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 5: NLI Hop Scoring on M=5 Candidates ━━━"
    echo "  Scores each candidate against hop1 and hop2 evidence separately"
    echo "  Estimated time: ~15 min on GPU"
    echo ""

    HOP_SCORES="${EXP}/preds/dev_hop_scores.jsonl"

    if [[ -f "${HOP_SCORES}" ]] && [[ $(wc -l < "${HOP_SCORES}") -ge 7405 ]]; then
        echo "  ✓ Already complete — skipping"
    else
        ${PYTHON} "${TOOLS_DIR}/exp2_q1_signal_independence.py" \
            --candidates  "${CANDS_M5}" \
            --nli_preds   "${EXP}/preds/dev_nli_preds.jsonl" \
            --evidence    "${EVIDENCE}" \
            --gold        "${GOLD}" \
            --model       "${NLI_MODEL}" \
            --out_hop_scores "${HOP_SCORES}" \
            --out_json    "${EXP}/metrics/q1_signal.json" \
            --out_jsonl   "${EXP}/metrics/q1_signal_perqid.jsonl" \
            --log         "${EXP}/logs/nli_hop_scoring.log" \
            --batch_size  64 \
            2>&1 | tee "${EXP}/logs/nli_hop_scoring_stdout.log"
    fi

    echo ""
    echo "  ✓ NLI hop scores: ${HOP_SCORES}"
    echo ""
    echo "  Next: bash $0 step6  (QA cross-encoder scoring)"
}


# ══════════════════════════════════════════════════════════════════════
step6() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 6: QA Cross-Encoder Hop Scoring ━━━"
    echo "  Scores each candidate using DeBERTa-v3-base-squad2"
    echo "  Estimated time: ~35 min on GPU"
    echo ""

    QA_SCORES="${EXP}/preds/dev_qa_hop_scores.jsonl"

    if [[ -f "${QA_SCORES}" ]] && [[ $(wc -l < "${QA_SCORES}") -ge 7405 ]]; then
        echo "  ✓ Already complete — skipping"
    else
        ${PYTHON} "${TOOLS_DIR}/exp_a1_qa_hop_score.py" \
            --candidates  "${CANDS_M5}" \
            --evidence    "${EVIDENCE}" \
            --gold        "${GOLD}" \
            --model       "${QA_MODEL}" \
            --out_jsonl   "${QA_SCORES}" \
            --log         "${EXP}/logs/qa_hop_scoring.log" \
            --batch_size  64 \
            2>&1 | tee "${EXP}/logs/qa_hop_scoring_stdout.log"
    fi

    echo ""
    echo "  ✓ QA scores: ${QA_SCORES}"
    echo ""
    echo "  Next: bash $0 step7  (lexical features)"
}


# ══════════════════════════════════════════════════════════════════════
step7() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 7: Lexical Grounding Features ━━━"
    echo "  CPU only, ~10 min"
    echo ""

    LEX_FEATS="${EXP}/preds/dev_lex_features.jsonl"

    if [[ -f "${LEX_FEATS}" ]] && [[ $(wc -l < "${LEX_FEATS}") -ge 7405 ]]; then
        echo "  ✓ Already complete — skipping"
    else
        ${PYTHON} "${TOOLS_DIR}/exp_a1_lex_features.py" \
            --candidates  "${CANDS_M5}" \
            --evidence    "${EVIDENCE}" \
            --gold        "${GOLD}" \
            --out_jsonl   "${LEX_FEATS}" \
            --log         "${EXP}/logs/lex_features.log" \
            2>&1 | tee "${EXP}/logs/lex_features_stdout.log"
    fi

    echo ""
    echo "  ✓ Lex features: ${LEX_FEATS}"
    echo ""
    echo "  Next: bash $0 step8  (oracle + M=1 baseline evaluation)"
}


# ══════════════════════════════════════════════════════════════════════
step8() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 8: Compute M=1 Baseline EM + Oracle Summary ━━━"
    echo ""

    # M=1 EM (this IS the baseline — one candidate = verifier picks it)
    ${PYTHON} -c "
import json, re, string, collections

def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred, gold):
    return int(normalize(pred) == normalize(gold))

gold_map = {str(ex['_id']): ex['answer']
            for ex in json.load(open('${GOLD}'))}

# M=1 greedy EM
m1_correct = 0
m1_preds = {}
for line in open('${CANDS_M1}'):
    rec = json.loads(line)
    qid = str(rec['qid'])
    cands = rec['candidates']
    ans = cands[0]['answer_text'] if cands else ''
    m1_preds[qid] = ans
    m1_correct += em(ans, gold_map.get(qid, ''))

m1_em = m1_correct / len(gold_map)
print(f'M=1 greedy EM: {m1_correct}/{len(gold_map)} = {m1_em:.4f}')

# Save M=1 preds in standard format (for bootstrap)
with open('${EXP}/results/m1_greedy_preds.jsonl', 'w') as f:
    for qid, pred in sorted(m1_preds.items()):
        f.write(json.dumps({'qid': qid, 'pred': pred}) + '\n')

# Oracle@5 summary
try:
    o5 = json.load(open('${EXP}/metrics/oracle_M5.json'))
    print(f'Oracle@5 EM:   {o5[\"overall\"][\"oracle_em\"]:.4f}')
except:
    print('Oracle@5: not yet computed')

# Oracle@1 summary
try:
    o1 = json.load(open('${EXP}/metrics/oracle_M1.json'))
    print(f'Oracle@1 EM:   {o1[\"overall\"][\"oracle_em\"]:.4f} (= M=1 greedy EM)')
except:
    pass

summary = {
    'm1_greedy_em': m1_em,
    'm1_correct': m1_correct,
    'n_questions': len(gold_map),
}
json.dump(summary, open('${EXP}/metrics/m1_summary.json', 'w'), indent=2)
print(f'\\nSaved: ${EXP}/metrics/m1_summary.json')
print(f'Saved: ${EXP}/results/m1_greedy_preds.jsonl')
"

    echo ""
    echo "  Next: bash $0 step9  (Phase 0 ablations + bootstrap)"
}


# ══════════════════════════════════════════════════════════════════════
step9() {
# ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "━━━ Step 9: Phase 0 Ablations + Bootstrap on Distractor Setting ━━━"
    echo "  Runs Z1 (majority vote), Z2 (surface), Z3 (chain-only), Z_full"
    echo "  Then bootstrap significance tests including M=1 comparison"
    echo ""

    # Run Phase 0 ablations
    ${PYTHON} "${TOOLS_DIR}/phase0_ablations.py" \
        --proj_root     "${PROJ_ROOT}" \
        --out_dir       "${EXP}/results" \
        --candidates    "${CANDS_M5}" \
        --hop_scores    "${EXP}/preds/dev_hop_scores.jsonl" \
        --qa_scores     "${EXP}/preds/dev_qa_hop_scores.jsonl" \
        --lex_features  "${EXP}/preds/dev_lex_features.jsonl" \
        --mono_preds    "${MONO_PREDS}" \
        --gold          "${GOLD}" \
        --n_folds 5 --seed 42 \
        2>&1 | tee "${EXP}/logs/phase0_ablations.log"

    echo ""

    # Run bootstrap WITH M=1 comparison (--distractor flag)
    ${PYTHON} "${TOOLS_DIR}/phase0_bootstrap.py" \
        --preds_dir     "${EXP}/results" \
        --gold          "${GOLD}" \
        --mono_preds    "${MONO_PREDS}" \
        --m1_preds      "${EXP}/results/m1_greedy_preds.jsonl" \
        --distractor \
        --out_dir       "${EXP}/results" \
        --n_bootstrap   10000 \
        --seed 42 \
        2>&1 | tee "${EXP}/logs/bootstrap.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DISTRACTOR-SETTING EXPERIMENT COMPLETE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Results: ${EXP}/results/"
    echo "  Key files:"
    echo "    phase0_results.json    — ablation EMs"
    echo "    bootstrap_results.json — significance tests"
    echo "    bootstrap_report.txt   — human-readable report"
    echo "    m1_greedy_preds.jsonl  — M=1 baseline predictions"
    echo ""
    echo "  PAPER TABLE (fill in from the above):"
    echo "    Row 1: M=1 greedy          → see m1_summary.json"
    echo "    Row 2: M=5 majority vote   → Z1 in phase0_results.json"
    echo "    Row 3: M=5 chain-only      → Z3 in phase0_results.json"
    echo "    Row 4: M=5 full verifier   → Z_full in phase0_results.json"
    echo "    Row 5: Oracle@5            → oracle_M5.json"
    echo ""
}


# ══════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════════════════

case "${STEP}" in
    step0) step0 ;;
    step1) step1 ;;
    step2) step2 ;;
    step3) step3 ;;
    step4) step4 ;;
    step5) step5 ;;
    step6) step6 ;;
    step7) step7 ;;
    step8) step8 ;;
    step9) step9 ;;
    all)
        step0
        echo ""; echo "Step 0 done. Starting step 1..."
        step1
        echo ""; echo "Verify vLLM is running, then press Enter..."
        read -r
        step2
        step3
        step4
        step5
        step6
        step7
        step8
        step9
        ;;
    *)
        echo "Usage: bash $0 {step0|step1|step2|step3|step4|step5|step6|step7|step8|step9|all}"
        echo ""
        echo "  step0  — Prepare distractor evidence file"
        echo "  step1  — Print vLLM startup instructions"
        echo "  step2  — Generate M=1 greedy (~1.5 hrs GPU)"
        echo "  step3  — Generate M=5 sampling (~4-5 hrs GPU)"
        echo "  step4  — Kill vLLM server"
        echo "  step5  — NLI hop scoring (~15 min GPU)"
        echo "  step6  — QA cross-encoder scoring (~35 min GPU)"
        echo "  step7  — Lexical features (~10 min CPU)"
        echo "  step8  — Compute M=1 baseline EM"
        echo "  step9  — Phase 0 ablations + bootstrap"
        echo "  all    — Run everything in sequence"
        ;;
esac