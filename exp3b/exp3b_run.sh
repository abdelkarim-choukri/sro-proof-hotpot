#!/usr/bin/env bash
# ==========================================================================
#  exp3b_run.sh — Exp3b: M=10 Candidates with Flat Prompt
#
#  Goal:    Does generating 10 candidates (instead of 5) raise oracle@10?
#  Change:  --m 10  (everything else identical to exp1b)
#  Control: Same model, same flat prompt v2, same evidence format, same seed
#
#  Decision gate:
#    oracle@10 ≥ 0.45   →  meaningful headroom, worth running full pipeline
#    oracle@10 ≈ 0.41   →  model is near-saturated, jump to 7B
#    oracle@10 < 0.40   →  no help from more candidates
#
#  Estimated runtime: ~50 min generation (2x exp1b) + 2 min oracle
# ==========================================================================

set -euo pipefail

# ── paths ──
PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

TOOLS_DIR="${PROJ_ROOT}/tools"

# Inputs (shared — read-only)
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
PROMPT_V2="${PROJ_ROOT}/exp1/inputs/prompt_v2.txt"

# Exp3b outputs
EXP3B_DIR="${PROJ_ROOT}/exp3b"
CANDIDATES="${EXP3B_DIR}/candidates/dev_M10_candidates_flat.jsonl"
MANIFEST="${EXP3B_DIR}/manifest.json"

ORACLE_JSON="${EXP3B_DIR}/metrics/oracle_M10_dev.json"
ORACLE_JSONL="${EXP3B_DIR}/metrics/oracle_M10_dev_perqid.jsonl"
ORACLE_SHA="${EXP3B_DIR}/metrics/oracle_M10_dev.json.sha256"

# ── pre-flight ──
echo "=== Exp3b: M=10 Candidates (Flat Prompt) ==="
echo "Evidence:   ${EVIDENCE}"
echo "Gold:       ${GOLD}"
echo "Prompt:     ${PROMPT_V2}  (same as exp1b)"
echo "Output:     ${CANDIDATES}"
echo "Candidates: M=10  (exp1b was M=5)"
echo ""

for f in "${EVIDENCE}" "${GOLD}" "${PROMPT_V2}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing input file: $f"
        exit 1
    fi
done

mkdir -p "${EXP3B_DIR}/candidates" "${EXP3B_DIR}/metrics" "${EXP3B_DIR}/logs"

# ==========================================================================
#  STEP 1: Generate M=10 candidates
# ==========================================================================
echo ""
echo "━━━ Step 1: Generate M=10 candidates (flat prompt v2) ━━━"
echo "  Model:  qwen2.5-1.5b-instruct"
echo "  M=10, T=0.7, top_p=0.95, seed=12345"
echo "  Evidence format: flat (identical to exp1b)"
echo ""

${PYTHON} ${TOOLS_DIR}/exp3_generate_candidates.py \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --prompt_file     "${PROMPT_V2}" \
    --prompt_version  v2 \
    --evidence_format flat \
    --out_jsonl       "${CANDIDATES}" \
    --manifest        "${MANIFEST}" \
    --llm_base_url    http://127.0.0.1:8000/v1 \
    --llm_model_id    qwen2.5-1.5b-instruct \
    --split dev \
    --m 10 \
    --seed 12345 \
    --resume \
    2>&1 | tee "${EXP3B_DIR}/logs/exp3b_generate.log"

echo ""
echo "  ✓ Candidates written to: ${CANDIDATES}"

# ==========================================================================
#  STEP 2: Compute oracle@10
# ==========================================================================
echo ""
echo "━━━ Step 2: Compute oracle@10 ━━━"
echo ""

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence    "${EVIDENCE}" \
    --candidates  "${CANDIDATES}" \
    --gold        "${GOLD}" \
    --split       dev \
    --m           10 \
    --out_json    "${ORACLE_JSON}" \
    --out_jsonl   "${ORACLE_JSONL}" \
    --out_sha256  "${ORACLE_SHA}" \
    --manifest    "${MANIFEST}" \
    2>&1 | tee "${EXP3B_DIR}/logs/exp3b_oracle.log"

echo ""
echo "  ✓ Oracle results: ${ORACLE_JSON}"

# ==========================================================================
#  STEP 3: Decision gate
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DECISION GATE: M=10 vs M=5 (same flat prompt)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Exp1b baseline (M=5, flat prompt v2):"
echo "    oracle_em = 0.3982    oracle_f1 = 0.5350"
echo "    feasible oracle_em = 0.4372"
echo ""
echo "  Exp3b result (M=10, flat prompt v2):"

${PYTHON} -c "
import json
r = json.load(open('${ORACLE_JSON}'))
ov = r['overall']
fs = r.get('subset_docrecall1', {})
print(f\"    oracle_em = {ov['oracle_em']:.4f}    oracle_f1 = {ov['oracle_f1']:.4f}\")
if fs:
    print(f\"    feasible oracle_em = {fs['oracle_em']:.4f}\")

delta_em = ov['oracle_em'] - 0.3982
delta_f1 = ov['oracle_f1'] - 0.5350

print()
print(f\"  Delta vs M=5:  EM {delta_em:+.4f}   F1 {delta_f1:+.4f}\")
print()

if ov['oracle_em'] >= 0.45:
    print(f\"  → MEANINGFUL HEADROOM: oracle@10 = {ov['oracle_em']:.4f} (≥0.45)\")
    print(f\"    More candidates help. Run full pipeline on M=10 candidates.\")
    print(f\"    The verifier gap will widen — more correct candidates to pick from.\")
elif ov['oracle_em'] >= 0.41:
    print(f\"  → MODEST GAIN: oracle@10 = {ov['oracle_em']:.4f} (0.41–0.45)\")
    print(f\"    Some headroom from more candidates but model is near saturation.\")
    print(f\"    Consider: run pipeline on M=10 AND test 7B on a 300-question sample.\")
elif ov['oracle_em'] > 0.3982:
    print(f\"  → MARGINAL: oracle@10 = {ov['oracle_em']:.4f} (<0.41)\")
    print(f\"    More candidates barely help. The 1.5B model is the bottleneck.\")
    print(f\"    Next: test Qwen 7B on 300-question sample.\")
else:
    print(f\"  → NO GAIN: oracle@10 = {ov['oracle_em']:.4f} (≤ M=5 baseline)\")
    print(f\"    Unexpected. Check for bugs. Model may be fully saturated.\")
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp3b complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"