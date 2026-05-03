#!/usr/bin/env bash
# =============================================================================
# z3_on_diverse.sh — Run Z3 pipeline on the diverse candidate pool.
#
# Purpose:
#   Quantify how much of the SFAV vs Z3 gap comes from the diversity fix
#   (better candidates) vs the neural verifier (better scoring model).
#
#   By running Z3 on dev_M5_diverse.jsonl we get:
#     Z3 + old candidates (dev_M5_sampling) : EM = 0.4749  ← already done
#     Z3 + diverse candidates               : EM = ???     ← this script
#     SFAV + diverse candidates             : EM = 0.6119  ← already done
#
#   The gap breakdown becomes:
#     Diversity gain     = Z3_diverse - Z3_old
#     Neural verifier gain = SFAV_diverse - Z3_diverse
#
# What this script does (no generation needed — candidates already exist):
#   Step 1: NLI hop scoring on diverse candidates   (~15 min, 1 GPU)
#   Step 2: QA cross-encoder scoring on diverse     (~35 min, 1 GPU)
#   Step 3: Lexical features on diverse             (~10 min, CPU)
#   Step 4: Z3 XGBoost ablations on diverse         (~20 min, CPU)
#   Step 5: Bootstrap significance test             (~5 min,  CPU)
#
# Total: ~1.5 hours (all sequential, 1 GPU)
# Output: exp_distractor/results_diverse/
#
# Usage:
#   cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#   CUDA_VISIBLE_DEVICES=0 bash tools/z3_on_diverse.sh \
#       2>&1 | tee exp_distractor/z3_diverse.log
# =============================================================================

set -uo pipefail

PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
TOOLS="${PROJ_ROOT}/tools"

# ── Input files ───────────────────────────────────────────────────────────────
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
EVIDENCE="${PROJ_ROOT}/exp_distractor/evidence/dev_distractor_chains.jsonl"
CANDS_DIVERSE="${PROJ_ROOT}/exp_distractor/candidates/dev_M5_diverse.jsonl"
MONO_PREDS="${PROJ_ROOT}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"

NLI_MODEL="/var/tmp/u24sf51014/sro/models/nli-roberta-base"
QA_MODEL="/var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2"

# ── Output directories ────────────────────────────────────────────────────────
OUT="${PROJ_ROOT}/exp_distractor/results_diverse"
PREDS_DIR="${OUT}/preds"
mkdir -p "${OUT}" "${PREDS_DIR}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

cd "${PROJ_ROOT}"

echo "================================================================"
echo "  Z3 on Diverse Candidates"
echo "  $(date)"
echo "================================================================"
echo ""
echo "  Input candidates : ${CANDS_DIVERSE}"
echo "  N lines: $(wc -l < ${CANDS_DIVERSE})"
echo "  Output directory : ${OUT}"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
for f in "${GOLD}" "${EVIDENCE}" "${CANDS_DIVERSE}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"; exit 1
    fi
done
echo "  Preflight: all input files found ✓"
echo ""

# ── Step 1: NLI Hop Scoring ───────────────────────────────────────────────────
NLI_SCORES="${PREDS_DIR}/dev_hop_scores_diverse.jsonl"

echo "━━━ Step 1: NLI Hop Scoring (~15 min) ━━━"
if [[ -f "${NLI_SCORES}" ]] && [[ $(wc -l < "${NLI_SCORES}") -ge 7405 ]]; then
    echo "  ✓ Already done ($(wc -l < ${NLI_SCORES}) lines)"
else
    ${PYTHON} "${TOOLS}/exp2_q1_signal_independence.py" \
        --candidates    "${CANDS_DIVERSE}" \
        --nli_preds     "${OUT}/preds/dev_nli_preds_diverse.jsonl" \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --model         "${NLI_MODEL}" \
        --out_hop_scores "${NLI_SCORES}" \
        --out_json      "${OUT}/nli_signal.json" \
        --out_jsonl     "${OUT}/nli_signal_perqid.jsonl" \
        --log           "${OUT}/nli_scoring.log" \
        --batch_size    64 \
        2>&1 | tee "${OUT}/nli_scoring_stdout.log"
    echo "  ✓ NLI scores: ${NLI_SCORES}"
fi
echo ""

# ── Step 2: QA Cross-Encoder Scoring ─────────────────────────────────────────
QA_SCORES="${PREDS_DIR}/dev_qa_hop_scores_diverse.jsonl"

echo "━━━ Step 2: QA Cross-Encoder Scoring (~35 min) ━━━"
if [[ -f "${QA_SCORES}" ]] && [[ $(wc -l < "${QA_SCORES}") -ge 7405 ]]; then
    echo "  ✓ Already done ($(wc -l < ${QA_SCORES}) lines)"
else
    ${PYTHON} "${TOOLS}/exp_a1_qa_hop_score.py" \
        --candidates    "${CANDS_DIVERSE}" \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --model         "${QA_MODEL}" \
        --out_jsonl     "${QA_SCORES}" \
        --log           "${OUT}/qa_scoring.log" \
        --batch_size    64 \
        2>&1 | tee "${OUT}/qa_scoring_stdout.log"
    echo "  ✓ QA scores: ${QA_SCORES}"
fi
echo ""

# ── Step 3: Lexical Features ──────────────────────────────────────────────────
LEX_FEATS="${PREDS_DIR}/dev_lex_features_diverse.jsonl"

echo "━━━ Step 3: Lexical Features (~10 min, CPU) ━━━"
if [[ -f "${LEX_FEATS}" ]] && [[ $(wc -l < "${LEX_FEATS}") -ge 7405 ]]; then
    echo "  ✓ Already done ($(wc -l < ${LEX_FEATS}) lines)"
else
    ${PYTHON} "${TOOLS}/exp_a1_lex_features.py" \
        --candidates    "${CANDS_DIVERSE}" \
        --evidence      "${EVIDENCE}" \
        --gold          "${GOLD}" \
        --out_jsonl     "${LEX_FEATS}" \
        --log           "${OUT}/lex_features.log" \
        2>&1 | tee "${OUT}/lex_features_stdout.log"
    echo "  ✓ Lex features: ${LEX_FEATS}"
fi
echo ""

# ── Step 4: Z3 Ablations ──────────────────────────────────────────────────────
echo "━━━ Step 4: Z3 XGBoost Ablations (~20 min, CPU) ━━━"
${PYTHON} "${TOOLS}/phase0_ablations_v2.py" \
    --proj_root     "${PROJ_ROOT}" \
    --out_dir       "${OUT}" \
    --candidates    "${CANDS_DIVERSE}" \
    --hop_scores    "${NLI_SCORES}" \
    --qa_scores     "${QA_SCORES}" \
    --lex_features  "${LEX_FEATS}" \
    --mono_preds    "${MONO_PREDS}" \
    --gold          "${GOLD}" \
    --n_folds 5 --seed 42 \
    2>&1 | tee "${OUT}/ablations.log"
echo ""

# ── Step 5: Bootstrap ─────────────────────────────────────────────────────────
echo "━━━ Step 5: Bootstrap Significance Tests (~5 min, CPU) ━━━"
${PYTHON} "${TOOLS}/phase0_bootstrap.py" \
    --preds_dir     "${OUT}" \
    --gold          "${GOLD}" \
    --mono_preds    "${MONO_PREDS}" \
    --distractor \
    --out_dir       "${OUT}" \
    --n_bootstrap   10000 \
    --seed 42 \
    2>&1 | tee "${OUT}/bootstrap.log"
echo ""

# ── Final comparison ──────────────────────────────────────────────────────────
echo "================================================================"
echo "  RESULTS COMPARISON"
echo "================================================================"
echo ""

${PYTHON} - << 'PYEOF'
import json, os

# Old Z3 results (from original sampling candidates)
old_path = "exp_distractor/results_v2/bootstrap_results.json"
new_path = "exp_distractor/results_diverse/bootstrap_results.json"
sfav_em  = 0.6119   # from aggregate.json

print(f"  {'System':<40} {'EM':>7}  {'ΔEM vs Z3_old':>15}")
print(f"  {'─'*65}")

if os.path.exists(old_path):
    old = json.load(open(old_path))["system_cis"]
    z3_old = old.get("Z3_chain", {}).get("em", float("nan"))
    m1_old = old.get("M1_greedy", {}).get("em", float("nan"))
    print(f"  {'M1 greedy (old pool)':<40} {m1_old:>7.4f}  {'—':>15}")
    for k, label in [("Z1_majority","Z1 majority vote"), ("Z2_surface","Z2 surface"),
                     ("Z3_chain","Z3 chain (XGBoost)"), ("Z_full","Z_full")]:
        v = old.get(k, {}).get("em", float("nan"))
        print(f"  {label+' [old candidates]':<40} {v:>7.4f}  {'baseline' if k=='Z3_chain' else '':>15}")

if os.path.exists(new_path):
    new = json.load(open(new_path))["system_cis"]
    z3_old = json.load(open(old_path))["system_cis"]["Z3_chain"]["em"]
    print(f"  {'─'*65}")
    for k, label in [("Z1_majority","Z1 majority vote"), ("Z2_surface","Z2 surface"),
                     ("Z3_chain","Z3 chain (XGBoost)"), ("Z_full","Z_full")]:
        v = new.get(k, {}).get("em", float("nan"))
        delta = v - z3_old
        print(f"  {label+' [diverse candidates]':<40} {v:>7.4f}  {delta:>+14.4f}")

# SFAV
z3_diverse_em = new.get("Z3_chain", {}).get("em", float("nan")) if os.path.exists(new_path) else float("nan")
z3_old_em = json.load(open(old_path))["system_cis"]["Z3_chain"]["em"] if os.path.exists(old_path) else float("nan")

print(f"  {'─'*65}")
print(f"  {'SFAV [diverse candidates]':<40} {sfav_em:>7.4f}  {sfav_em-z3_old_em:>+14.4f}")
print()

if not (z3_diverse_em != z3_diverse_em):  # not nan
    diversity_gain = z3_diverse_em - z3_old_em
    neural_gain    = sfav_em - z3_diverse_em
    total_gain     = sfav_em - z3_old_em
    print(f"  Gap decomposition:")
    print(f"    Total gain (SFAV - Z3_old)       : {total_gain:+.4f} ({total_gain*100:+.2f} pp)")
    print(f"    Diversity fix contribution        : {diversity_gain:+.4f} ({diversity_gain*100:+.2f} pp)")
    print(f"    Neural verifier contribution      : {neural_gain:+.4f} ({neural_gain*100:+.2f} pp)")
    print(f"    Diversity share of total gain     : {100*diversity_gain/total_gain:.1f}%")
    print(f"    Neural verifier share             : {100*neural_gain/total_gain:.1f}%")
PYEOF

echo ""
echo "  Full results: exp_distractor/results_diverse/"
echo "  $(date)"
echo "================================================================"