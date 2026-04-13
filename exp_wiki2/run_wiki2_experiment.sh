#!/usr/bin/env bash
# =============================================================================
#  run_wiki2_experiment.sh — Full 2WikiMultiHopQA Experiment Pipeline
#
#  Track A of the publication plan: second dataset to address the #1
#  rejection risk ("results limited to a single benchmark").
#
#  This script mirrors exp_distractor/run_distractor_experiment.sh exactly,
#  adapted for 2WikiMultiHopQA paths and question types.
#
#  All existing tools are reused UNCHANGED. Only the evidence preparation
#  (Step 0) and per-type analysis (Step 9b) are new.
#
# ─────────────────────────────────────────────────────────────────────────────
#  PRE-REGISTRATION: THREE CONTINGENCY FRAMINGS
#  Written BEFORE running any experiments (publication plan, Track A).
#  Classify your result IMMEDIATELY after Step 8 using these framings.
#
#  CASE A — Strong result (chain marginal ≥ 0.70pp, p < 0.01):
#    "The compositional verification principle generalizes across benchmarks.
#     On 2WikiMultiHopQA, per-hop features provide a +X.Xpp gain (p<0.001),
#     consistent with HotpotQA (+0.96pp). When compositional structure exists,
#     per-hop decomposition recovers discriminative signal that flat scoring
#     discards."
#    ACTION: Proceed directly to paper writing. This is your headline result.
#
#  CASE B — Moderate result (chain marginal +0.30–0.69pp, borderline sig):
#    "Per-hop features provide consistent directional improvement across both
#     benchmarks, though magnitude varies with structural clarity. Analysis
#     reveals the gain concentrates in question types where hop decomposition
#     is cleanest (bridge > comparison > inference > compositional), confirming
#     the effect depends on compositional structure quality."
#    ACTION: Run wiki2_per_type_analysis.py immediately. Run hop clarity
#    analysis (score entropy per candidate, correlation with per-question
#    chain gain). This turns the moderate result into a mechanistic finding.
#
#  CASE C — Weak or null result (chain marginal < 0.30pp or p > 0.10):
#    "The compositional verification principle shows strong gains when evidence
#     structure is clean (HotpotQA: +0.96pp) but attenuates when hop
#     decomposition is noisier (2WikiMultiHopQA: +X.Xpp). We measure hop
#     mapping quality and show that chain feature discriminability correlates
#     with decomposition clarity (r=X.XX). This confirms the conditional
#     claim: verification should respect compositional structure when it
#     exists, and the benefit scales with the quality of that structure."
#    ACTION: Run wiki2_per_type_analysis.py. Measure hop_score entropy per
#    question type. Compute Pearson(hop_clarity, per_question_chain_gain).
#    This is the scientific finding — not damage control.
#
#  NOTE: Do NOT decide case classification before seeing bootstrap results.
#  The chain marginal is Z_full EM − Z2 EM (NOT Z_full − Z1).
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS — adjust if your project root differs
# ─────────────────────────────────────────────────────────────────────────────

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# 2WikiMultiHopQA raw data (framolfese repack — uses "id" not "_id")
WIKI2_RAW="${PROJ_ROOT}/data/wiki2/raw/dev.json"
# Normalized copy with "_id" restored so all existing pipeline scripts work unchanged
WIKI2_GOLD="${PROJ_ROOT}/data/wiki2/raw/dev_normalized.json"

# Shared resources (unchanged from distractor setting)
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"
NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"
QA_MODEL="/var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2"
MODEL_ID="/var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct"

# Monolithic baseline (from MDR exp0c — used as fallback reference in bootstrap)
MONO_PREDS="${PROJ_ROOT}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"

# Experiment directory
EXP="${PROJ_ROOT}/exp_wiki2"

# Sub-directories (mirrors exp_distractor layout)
EVIDENCE_DIR="${EXP}/evidence"
CAND_DIR="${EXP}/candidates"
PREDS_DIR="${EXP}/preds"
METRICS_DIR="${EXP}/metrics"
RESULTS_DIR="${EXP}/results"
LOGS_DIR="${EXP}/logs"

# Key files
EVIDENCE="${EVIDENCE_DIR}/dev_wiki2_chains.jsonl"
CANDS_M1="${CAND_DIR}/dev_M1_greedy.jsonl"
CANDS_M5="${CAND_DIR}/dev_M5_sampling.jsonl"
NLI_PREDS="${PREDS_DIR}/dev_nli_preds.jsonl"
HOP_SCORES="${PREDS_DIR}/dev_hop_scores.jsonl"
QA_SCORES="${PREDS_DIR}/dev_qa_hop_scores.jsonl"
LEX_FEATS="${PREDS_DIR}/dev_lex_features.jsonl"
M1_PREDS_CONVERTED="${PREDS_DIR}/dev_M1_greedy_preds.jsonl"

# Expected question count for 2WikiMultiHopQA dev set (~12,576)
# Adjust if your dev split differs.
N_WIKI2=12576

# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Track A: 2WikiMultiHopQA — Chain-Aware Verification"
echo "  Generator: Qwen2.5-7B-Instruct  M=5  T=0.7  seed=12345"
echo "  Ablations: Z1/Z2/Z3/Z_full  Bootstrap: B=10,000"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

echo "Pre-flight checks ..."
MISSING=0
for F in "${WIKI2_RAW}" "${PROMPT_V2}" "${NLI_MODEL}" "${MODEL_ID}"; do
    if [[ ! -e "${F}" ]]; then
        echo "  ✗ MISSING: ${F}"
        MISSING=1
    else
        echo "  ✓ ${F}"
    fi
done
if [[ ${MISSING} -eq 1 ]]; then
    echo ""
    echo "ERROR: Missing required files. See above."
    echo "  Download 2WikiMultiHopQA dev set to: ${WIKI2_RAW}"
    echo "  URL: https://www.dropbox.com/s/ms2m13252h6xubs/data_ids.zip"
    echo "       (or from huggingface: 'wikimultihopqa' dataset)"
    exit 1
fi

mkdir -p "${EVIDENCE_DIR}" "${CAND_DIR}" "${PREDS_DIR}" \
         "${METRICS_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0a: Normalize field names
#  The framolfese repack renames "_id" → "id". All existing pipeline scripts
#  (phase0_ablations_v2.py, phase0_bootstrap.py, exp1_nli_baseline.py, etc.)
#  load gold with str(ex["_id"]). This one-liner restores "_id" in a copy so
#  zero existing scripts need modification.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 0a: Normalize field names (id → _id) ━━━"

if [[ -f "${WIKI2_GOLD}" ]]; then
    echo "  ✓ Already done — ${WIKI2_GOLD}"
else
    ${PYTHON} -c "
import json
data = json.load(open('${WIKI2_RAW}', encoding='utf-8'))
for ex in data:
    if '_id' not in ex and 'id' in ex:
        ex['_id'] = ex['id']
with open('${WIKI2_GOLD}', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
print(f'  Written {len(data):,} records with _id restored → ${WIKI2_GOLD}')
"
fi
echo ""

# =============================================================================
#  STEP 0: Evidence Preparation
#  Convert 2WikiMultiHopQA dev.json → pipeline evidence JSONL
#  Output schema is IDENTICAL to distractor_prepare_evidence.py so all
#  downstream tools work unchanged.
#  ~1 min CPU only
# =============================================================================
echo "━━━ Step 0: Evidence Preparation ━━━"

if [[ -f "${EVIDENCE}" ]] && [[ $(wc -l < "${EVIDENCE}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete ($(wc -l < "${EVIDENCE}") lines) — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/wiki2_prepare_evidence.py" \
        --gold         "${WIKI2_RAW}" \
        --out_evidence "${EVIDENCE}" \
        --out_stats    "${EVIDENCE_DIR}/prep_stats.json" \
        2>&1 | tee "${LOGS_DIR}/step0_evidence_prep.log"
fi

echo "  ✓ Evidence: ${EVIDENCE}"

# Check stats for warnings
if [[ -f "${EVIDENCE_DIR}/prep_stats.json" ]]; then
    GOLD0=$(python3 -c "import json; d=json.load(open('${EVIDENCE_DIR}/prep_stats.json')); print(d.get('n_gold_found_0',0))" 2>/dev/null || echo "?")
    if [[ "${GOLD0}" != "0" ]] && [[ "${GOLD0}" != "?" ]]; then
        echo "  ⚠  WARNING: ${GOLD0} questions with 0 gold paragraphs found!"
        echo "     Inspect prep_stats.json and fix wiki2_prepare_evidence.py if > 10"
    fi
fi

echo ""

# =============================================================================
#  MANDATORY: Hop Validation Gate (Day 2)
#  Run BEFORE any generation. Prints 50 examples and prompts for Y/N.
#  Decision: ≥40/50 proceed, 30-39 proceed with caveat, <30 STOP.
# =============================================================================
if [[ ! -f "${EVIDENCE_DIR}/hop_validation.json" ]]; then
    echo "━━━ MANDATORY: Hop Validation (run before Step 1) ━━━"
    echo ""
    echo "  This is the Day 2 gate from the publication plan."
    echo "  You MUST run this before generating candidates."
    echo ""
    echo "  Run interactively:"
    echo "    ${PYTHON} ${TOOLS_DIR}/wiki2_hop_validation.py \\"
    echo "        --evidence ${EVIDENCE} \\"
    echo "        --out_json ${EVIDENCE_DIR}/hop_validation.json"
    echo ""
    echo "  Or dry-run (prints all 50 without prompting):"
    echo "    ${PYTHON} ${TOOLS_DIR}/wiki2_hop_validation.py \\"
    echo "        --evidence ${EVIDENCE} \\"
    echo "        --out_json ${EVIDENCE_DIR}/hop_validation.json \\"
    echo "        --auto"
    echo ""
    echo "  Once validation is complete, re-run this script to continue."
    echo ""
    read -r -p "  Press Enter to run --auto dry-run now, or Ctrl+C to run manually: "
    ${PYTHON} "${TOOLS_DIR}/wiki2_hop_validation.py" \
        --evidence "${EVIDENCE}" \
        --out_json "${EVIDENCE_DIR}/hop_validation.json" \
        --auto \
        2>&1 | tee "${LOGS_DIR}/hop_validation.log"
    echo ""
    echo "  ⚠  Dry-run complete. Re-run without --auto for actual annotation."
    echo "     Then re-run this script."
    exit 0
else
    DECISION=$(python3 -c "
import json, sys
d = json.load(open('${EVIDENCE_DIR}/hop_validation.json'))
print(d.get('decision','?'))
print(d.get('n_clean','?'), 'of', d.get('n_total','?'))
" 2>/dev/null || echo "unknown ?")
    echo "  ✓ Hop validation already done: ${DECISION}"
    DECISION_VAL=$(python3 -c "import json; print(json.load(open('${EVIDENCE_DIR}/hop_validation.json')).get('decision','?'))" 2>/dev/null || echo "?")
    if [[ "${DECISION_VAL}" == "stop_fix_mapping" ]]; then
        echo ""
        echo "  ✗ GATE FAILED: hop_validation.json says 'stop_fix_mapping'."
        echo "     Fix wiki2_prepare_evidence.py hop ordering logic first."
        echo "     Then delete ${EVIDENCE} and ${EVIDENCE_DIR}/hop_validation.json"
        echo "     and re-run this script."
        exit 1
    fi
fi
echo ""

# =============================================================================
#  STEP 1: Generate M=1 Greedy Candidates
#  Same script as distractor_generate.py — reads all_paragraphs field
#  from the evidence JSONL and formats [paragraph N] blocks for the prompt.
#  ~2–3 hours (12K questions × 1 candidate)
#  REQUIRES: vLLM serving Qwen2.5-7B-Instruct on port 8000
# =============================================================================
echo "━━━ Step 1: Generate M=1 Greedy (T=0) ━━━"
echo "  REQUIRES: vLLM serving 7B on port 8000"
echo "  Estimated time: ~2–3 hours"
echo ""

if [[ -f "${CANDS_M1}" ]] && [[ $(wc -l < "${CANDS_M1}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete ($(wc -l < "${CANDS_M1}") lines) — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/distractor_generate.py" \
        --evidence        "${EVIDENCE}" \
        --gold            "${WIKI2_GOLD}" \
        --prompt_file     "${PROMPT_V2}" \
        --out_jsonl       "${CANDS_M1}" \
        --manifest        "${CAND_DIR}/manifest_m1.json" \
        --llm_base_url    http://127.0.0.1:8000/v1 \
        --llm_model_id    "${MODEL_ID}" \
        --m               1 \
        --temperature     0.0 \
        --seed            12345 \
        --resume \
        2>&1 | tee "${LOGS_DIR}/step1_generate_m1.log"
fi
echo "  ✓ M=1 candidates: ${CANDS_M1}"
echo ""

# =============================================================================
#  STEP 2: Generate M=5 Sampling Candidates
#  ~5–6 hours (12K questions × 5 candidates × T=0.7)
#  REQUIRES: vLLM still running
# =============================================================================
echo "━━━ Step 2: Generate M=5 Sampling (T=0.7) ━━━"
echo "  Estimated time: ~5–6 hours"
echo ""

if [[ -f "${CANDS_M5}" ]] && [[ $(wc -l < "${CANDS_M5}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete ($(wc -l < "${CANDS_M5}") lines) — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/distractor_generate.py" \
        --evidence        "${EVIDENCE}" \
        --gold            "${WIKI2_GOLD}" \
        --prompt_file     "${PROMPT_V2}" \
        --out_jsonl       "${CANDS_M5}" \
        --manifest        "${CAND_DIR}/manifest_m5.json" \
        --llm_base_url    http://127.0.0.1:8000/v1 \
        --llm_model_id    "${MODEL_ID}" \
        --m               5 \
        --temperature     0.7 \
        --top_p           0.95 \
        --seed            12345 \
        --resume \
        2>&1 | tee "${LOGS_DIR}/step2_generate_m5.log"
fi
echo "  ✓ M=5 candidates: ${CANDS_M5}"
echo ""

# =============================================================================
#  STEP 3: Oracle@1 and Oracle@5
#  CPU only, ~3 min
# =============================================================================
echo "━━━ Step 3: Oracle Computation ━━━"

# Oracle@5
ORACLE_M5_JSON="${METRICS_DIR}/oracle_M5.json"
ORACLE_M5_JSONL="${METRICS_DIR}/oracle_M5_perqid.jsonl"
if [[ -f "${ORACLE_M5_JSON}" ]]; then
    echo "  ✓ Oracle@5 already done — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp1_compute_oracle.py" \
        --evidence   "${EVIDENCE}" \
        --candidates "${CANDS_M5}" \
        --gold       "${WIKI2_GOLD}" \
        --split      dev \
        --m          5 \
        --out_json   "${ORACLE_M5_JSON}" \
        --out_jsonl  "${ORACLE_M5_JSONL}" \
        --out_sha256 "${METRICS_DIR}/oracle_M5.sha256" \
        --manifest   "${METRICS_DIR}/manifest_oracle.json" \
        2>&1 | tee "${LOGS_DIR}/step3_oracle_m5.log"
fi

# Oracle@1 (for M=1 greedy reference)
ORACLE_M1_JSON="${METRICS_DIR}/oracle_M1.json"
if [[ -f "${ORACLE_M1_JSON}" ]]; then
    echo "  ✓ Oracle@1 already done — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp1_compute_oracle.py" \
        --evidence   "${EVIDENCE}" \
        --candidates "${CANDS_M1}" \
        --gold       "${WIKI2_GOLD}" \
        --split      dev \
        --m          1 \
        --out_json   "${ORACLE_M1_JSON}" \
        --out_jsonl  "${METRICS_DIR}/oracle_M1_perqid.jsonl" \
        --out_sha256 "${METRICS_DIR}/oracle_M1.sha256" \
        --manifest   "${METRICS_DIR}/manifest_oracle.json" \
        2>&1 | tee "${LOGS_DIR}/step3_oracle_m1.log"
fi

# Print oracle summary
echo ""
echo "  ★ Oracle Summary:"
${PYTHON} -c "
import json, sys
try:
    m5 = json.load(open('${ORACLE_M5_JSON}'))
    m1 = json.load(open('${ORACLE_M1_JSON}'))
    o5 = m5.get('overall', {}).get('oracle_em', 0)
    o1 = m1.get('overall', {}).get('oracle_em', 0)
    print(f'     Oracle@1: {o1:.4f}  Oracle@5: {o5:.4f}')
    print(f'     Self-consistency ceiling (oracle@5 - oracle@1): {100*(o5-o1):.2f}pp')
except Exception as e:
    print(f'     (could not read oracle JSON: {e})')
" 2>/dev/null || true
echo ""

echo ""
echo "  ★★★  STOP vLLM NOW before NLI scoring  ★★★"
echo "  Run: pkill -f 'vllm.entrypoints' || pkill -f 'vllm serve'"
echo "  Wait until GPUs are fully free, then press Enter."
echo ""
read -r -p "  Press Enter when GPU is free ..."
echo ""

# =============================================================================
#  STEP 4: Flat NLI Scoring
#  GPU, ~5 min (NLI model is fast at 12K questions, batch_size=64)
# =============================================================================
echo "━━━ Step 4: Flat NLI Scoring ━━━"

if [[ -f "${NLI_PREDS}" ]] && [[ $(wc -l < "${NLI_PREDS}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp1_nli_baseline.py" \
        --candidates  "${CANDS_M5}" \
        --evidence    "${EVIDENCE}" \
        --gold        "${WIKI2_GOLD}" \
        --model       "${NLI_MODEL}" \
        --out_metrics "${METRICS_DIR}/nli_baseline.json" \
        --out_preds   "${NLI_PREDS}" \
        --batch_size  64 \
        2>&1 | tee "${LOGS_DIR}/step4_nli_baseline.log"
fi
echo "  ✓ Flat NLI: ${NLI_PREDS}"
echo ""

# =============================================================================
#  STEP 5: Hop-Level NLI Scoring
#  Reads chains[0].hops[0/1].text from evidence JSONL (same schema)
#  GPU, ~5 min
# =============================================================================
echo "━━━ Step 5: Hop-Level NLI Scoring ━━━"

if [[ -f "${HOP_SCORES}" ]] && [[ $(wc -l < "${HOP_SCORES}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp2_q1_signal_independence.py" \
        --candidates     "${CANDS_M5}" \
        --nli_preds      "${NLI_PREDS}" \
        --evidence       "${EVIDENCE}" \
        --gold           "${WIKI2_GOLD}" \
        --model          "${NLI_MODEL}" \
        --out_hop_scores "${HOP_SCORES}" \
        --out_json       "${METRICS_DIR}/q1_signal_independence.json" \
        --out_jsonl      "${METRICS_DIR}/q1_signal_independence_perqid.jsonl" \
        --log            "${LOGS_DIR}/step5_hop_nli.log" \
        --batch_size     64 \
        2>&1 | tee "${LOGS_DIR}/step5_hop_nli_stdout.log"
fi
echo "  ✓ Hop NLI: ${HOP_SCORES}"
echo ""

# =============================================================================
#  STEP 6: QA Cross-Encoder Hop Scoring (DeBERTa-v3-base-squad2)
#  GPU, ~30 min (QA model is slower than NLI)
# =============================================================================
echo "━━━ Step 6: QA Cross-Encoder Hop Scoring ━━━"
echo "  Model: DeBERTa-v3-base-squad2  (~30 min GPU)"
echo ""

if [[ -f "${QA_SCORES}" ]] && [[ $(wc -l < "${QA_SCORES}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp_a1_qa_hop_score.py" \
        --candidates "${CANDS_M5}" \
        --evidence   "${EVIDENCE}" \
        --model      "${QA_MODEL}" \
        --out_jsonl  "${QA_SCORES}" \
        --out_json   "${METRICS_DIR}/qa_hop_scores_summary.json" \
        --device     cuda \
        2>&1 | tee "${LOGS_DIR}/step6_qa_hop.log"
fi
echo "  ✓ QA scores: ${QA_SCORES}"
echo ""

# =============================================================================
#  STEP 7: Lexical Features (CPU only, ~10 sec)
# =============================================================================
echo "━━━ Step 7: Lexical Features ━━━"

if [[ -f "${LEX_FEATS}" ]] && [[ $(wc -l < "${LEX_FEATS}") -ge ${N_WIKI2} ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/exp_a1_lex_features.py" \
        --evidence   "${EVIDENCE}" \
        --candidates "${CANDS_M5}" \
        --out_jsonl  "${LEX_FEATS}" \
        --out_json   "${METRICS_DIR}/lex_features_summary.json" \
        2>&1 | tee "${LOGS_DIR}/step7_lex_features.log"
fi
echo "  ✓ Lex features: ${LEX_FEATS}"
echo ""

# =============================================================================
#  STEP 8: Phase 0 Ablations (Z1 / Z2 / Z3 / Z_full)
#  Two-stage XGBoost, 5-fold CV, CPU only, ~5 min
#
#  IMPORTANT: --greedy_preds prevents cross-setting contamination.
#  M=1 greedy preds are converted to the {qid, pred} format first if needed.
# =============================================================================
echo "━━━ Step 8: Phase 0 Ablations (Z1/Z2/Z3/Z_full) ━━━"
echo "  5-fold XGBoost CV, all 19 features  (~5 min CPU)"
echo ""

# Convert M=1 greedy candidates to simple pred format for the fallback
if [[ ! -f "${M1_PREDS_CONVERTED}" ]]; then
    echo "  Converting M1 candidates to pred format ..."
    ${PYTHON} -c "
import json, sys
out = open('${M1_PREDS_CONVERTED}', 'w')
with open('${CANDS_M1}') as f:
    for line in f:
        rec = json.loads(line.strip())
        qid = str(rec['qid'])
        cands = rec.get('candidates', [])
        if cands and isinstance(cands[0], dict):
            pred = cands[0].get('answer_text', '')
        elif cands:
            pred = str(cands[0])
        else:
            pred = ''
        out.write(json.dumps({'qid': qid, 'pred': pred}) + '\n')
out.close()
print(f'Converted M1 preds written to ${M1_PREDS_CONVERTED}')
" 2>&1
fi

if [[ -f "${RESULTS_DIR}/z_full_preds.jsonl" ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/phase0_ablations_v2.py" \
        --proj_root      "${PROJ_ROOT}" \
        --candidates     "${CANDS_M5}" \
        --hop_scores     "${HOP_SCORES}" \
        --qa_scores      "${QA_SCORES}" \
        --lex_features   "${LEX_FEATS}" \
        --greedy_preds   "${M1_PREDS_CONVERTED}" \
        --gold           "${WIKI2_GOLD}" \
        --out_dir        "${RESULTS_DIR}" \
        --n_folds        5 \
        --seed           42 \
        2>&1 | tee "${LOGS_DIR}/step8_ablations.log"
fi

echo "  ✓ Ablation results: ${RESULTS_DIR}/"
echo ""

# Print immediate result summary
echo "  ★ Ablation Summary:"
${PYTHON} -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
print()
systems = [
    ('Z1_majority',  'z1_majority_preds.jsonl'),
    ('Z2_surface',   'z2_surface_preds.jsonl'),
    ('Z3_chain',     'z3_chain_preds.jsonl'),
    ('Z_full',       'z_full_preds.jsonl'),
]

# Load gold
import re, string
def norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

with open('${WIKI2_GOLD}') as f:
    raw = f.read(1); f.seek(0)
    data = json.load(f) if raw.strip()=='[' else [json.loads(l) for l in f if l.strip()]
gold_map = {str(ex['_id']): ex['answer'] for ex in data}
n_total = len(gold_map)

rows = []
for name, fname in systems:
    path = os.path.join(results_dir, fname)
    if not os.path.isfile(path):
        continue
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            preds[str(rec['qid'])] = rec.get('pred','')
    n_correct = sum(int(norm(preds.get(q,''))==norm(gold_map[q])) for q in gold_map)
    em = n_correct / n_total
    rows.append((name, em))

for name, em in rows:
    print(f'    {name:12s}: {em:.4f}  ({em*100:.2f}%)')

if len(rows) >= 4:
    z2_em   = rows[1][1]
    zf_em   = rows[3][1]
    chain_pp = 100*(zf_em - z2_em)
    z1_em = rows[0][1]
    verif_pp = 100*(zf_em - z1_em)
    print()
    print(f'    Chain marginal (Z_full - Z2): {chain_pp:+.2f}pp')
    print(f'    Verifier vs self-consistency (Z_full - Z1): {verif_pp:+.2f}pp')
    # Case classification
    if chain_pp >= 0.70:
        case = 'A (STRONG)'
    elif chain_pp >= 0.30:
        case = 'B (MODERATE)'
    else:
        case = 'C (WEAK/NULL)'
    print(f'    → Preliminary case classification: CASE {case}')
    print(f'    → Confirm with bootstrap (Step 9) before finalising')
" 2>/dev/null || echo "  (could not compute summary)"

echo ""

# =============================================================================
#  STEP 9: Bootstrap Significance (B=10,000)
#  CPU only, ~2–3 min for 12K questions
# =============================================================================
echo "━━━ Step 9: Bootstrap Significance Tests ━━━"
echo "  B=10,000 paired bootstrap, seed=42  (~2–3 min CPU)"
echo ""

BOOTSTRAP_JSON="${RESULTS_DIR}/bootstrap_results.json"
if [[ -f "${BOOTSTRAP_JSON}" ]]; then
    echo "  ✓ Already complete — skipping"
else
    ${PYTHON} "${TOOLS_DIR}/phase0_bootstrap.py" \
        --preds_dir    "${RESULTS_DIR}" \
        --gold         "${WIKI2_GOLD}" \
        --mono_preds   "${MONO_PREDS}" \
        --m1_preds     "${M1_PREDS_CONVERTED}" \
        --distractor \
        --out_dir      "${RESULTS_DIR}" \
        --n_bootstrap  10000 \
        --seed         42 \
        2>&1 | tee "${LOGS_DIR}/step9_bootstrap.log"
fi

echo "  ✓ Bootstrap report: ${RESULTS_DIR}/bootstrap_report.txt"
echo "  ✓ Full results:     ${BOOTSTRAP_JSON}"
echo ""

# =============================================================================
#  STEP 9b: Per-Type Analysis
#  CPU only, ~30 sec
# =============================================================================
echo "━━━ Step 9b: Per-Question-Type Analysis ━━━"

PER_TYPE_JSON="${RESULTS_DIR}/per_type_analysis.json"
${PYTHON} "${TOOLS_DIR}/wiki2_per_type_analysis.py" \
    --gold      "${WIKI2_GOLD}" \
    --preds_dir "${RESULTS_DIR}" \
    --out_json  "${PER_TYPE_JSON}" \
    --m1_preds  "${M1_PREDS_CONVERTED}" \
    2>&1 | tee "${LOGS_DIR}/step9b_per_type.log"

echo "  ✓ Per-type analysis: ${PER_TYPE_JSON}"
echo ""

# =============================================================================
#  DONE
# =============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "  2WikiMultiHopQA experiment complete."
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Key outputs:"
echo "    Evidence:          ${EVIDENCE}"
echo "    Ablation results:  ${RESULTS_DIR}/"
echo "    Bootstrap report:  ${RESULTS_DIR}/bootstrap_report.txt"
echo "    Per-type analysis: ${PER_TYPE_JSON}"
echo ""
echo "  Next steps:"
echo "    1. Read bootstrap_report.txt — classify as Case A / B / C"
echo "    2. Read per_type_analysis.json — check type-specific chain marginals"
echo "    3. If Case B/C: run hop clarity analysis (score entropy per question)"
echo "    4. Update cross-dataset table in the paper draft"
echo ""
echo "  Cross-dataset table (fill in from bootstrap_results.json):"
echo "    Metric                       | HotpotQA (MDR) | HotpotQA (distractor) | 2WikiMultiHopQA"
echo "    M=1 greedy                   |     —          |       44.40%          |     ?"
echo "    Z1 (majority vote)           |   46.41%       |       46.54%          |     ?"
echo "    Z3 (chain-only)              |   46.63%       |       47.49%          |     ?"
echo "    Z_full                       |   46.98%       |       47.52%          |     ?"
echo "    Chain marginal (Z_full - Z2) |   +0.77pp***   |       +0.96pp***      |     ?"
echo ""