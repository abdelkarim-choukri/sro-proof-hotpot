#!/usr/bin/env bash
# ==========================================================================
#  exp3_run.sh — Exp3: Chain-Aware Prompting Experiment
#
#  Goal:    Does labelling hop structure in the prompt improve oracle@5?
#  Change:  prompt_v3_chain_aware + evidence_text_chain_aware()
#  Control: Everything else identical to exp1b (same model, M, temp, seed)
#
#  Decision gate:
#    oracle@5(v3) > oracle@5(v2=0.3982) by ≥ 1.5pp  →  proceed to full pipeline
#    oracle@5(v3) ≈ oracle@5(v2) within 1pp           →  prompt change is negligible
#    oracle@5(v3) < oracle@5(v2)                       →  chain labels hurt, revert
#
#  Estimated runtime: ~25 min generation + 2 min oracle (same as exp1b)
# ==========================================================================

set -euo pipefail

# ── paths (adjust PROJ_ROOT to your project root) ──
PROJ_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

EXP3_DIR="${PROJ_ROOT}/exp3"
TOOLS_DIR="${PROJ_ROOT}/tools"

# Inputs (shared with exp1b — read-only)
EVIDENCE="${PROJ_ROOT}/exp1b/evidence/dev_K100_chains.jsonl"
GOLD="${PROJ_ROOT}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

# Exp3-specific
PROMPT_V3="${EXP3_DIR}/inputs/prompt_v3_chain_aware.txt"
CANDIDATES="${EXP3_DIR}/candidates/dev_M5_candidates_chain_aware.jsonl"
MANIFEST="${EXP3_DIR}/manifest.json"

# Oracle outputs
ORACLE_JSON="${EXP3_DIR}/metrics/oracle_M5_dev.json"
ORACLE_JSONL="${EXP3_DIR}/metrics/oracle_M5_dev_perqid.jsonl"
ORACLE_SHA="${EXP3_DIR}/metrics/oracle_M5_dev.json.sha256"

# ── pre-flight checks ──
echo "=== Exp3: Chain-Aware Prompting ==="
echo "Evidence:   ${EVIDENCE}"
echo "Gold:       ${GOLD}"
echo "Prompt:     ${PROMPT_V3}"
echo "Output:     ${CANDIDATES}"
echo ""

for f in "${EVIDENCE}" "${GOLD}" "${PROMPT_V3}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing input file: $f"
        exit 1
    fi
done

mkdir -p "${EXP3_DIR}/candidates" "${EXP3_DIR}/metrics" "${EXP3_DIR}/logs"

# ==========================================================================
#  STEP 1: Generate M=5 candidates with chain-aware prompt
# ==========================================================================
echo ""
echo "━━━ Step 1: Generate candidates (chain-aware prompt) ━━━"
echo "  Model:  qwen2.5-1.5b-instruct"
echo "  M=5, T=0.7, top_p=0.95, seed=12345"
echo "  Evidence format: chain_aware"
echo ""

${PYTHON} ${TOOLS_DIR}/exp3_generate_candidates.py \
    --evidence        "${EVIDENCE}" \
    --gold            "${GOLD}" \
    --prompt_file     "${PROMPT_V3}" \
    --prompt_version  v3_chain_aware \
    --evidence_format chain_aware \
    --out_jsonl       "${CANDIDATES}" \
    --manifest        "${MANIFEST}" \
    --llm_base_url    http://127.0.0.1:8000/v1 \
    --llm_model_id    qwen2.5-1.5b-instruct \
    --split dev \
    --m 5 \
    --seed 12345 \
    --resume \
    2>&1 | tee "${EXP3_DIR}/logs/exp3_generate.log"

echo ""
echo "  ✓ Candidates written to: ${CANDIDATES}"

# ==========================================================================
#  STEP 2: Compute oracle@5
# ==========================================================================
echo ""
echo "━━━ Step 2: Compute oracle@5 ━━━"
echo ""

${PYTHON} ${TOOLS_DIR}/exp1_compute_oracle.py \
    --evidence    "${EVIDENCE}" \
    --candidates  "${CANDIDATES}" \
    --gold        "${GOLD}" \
    --split       dev \
    --m           5 \
    --out_json    "${ORACLE_JSON}" \
    --out_jsonl   "${ORACLE_JSONL}" \
    --out_sha256  "${ORACLE_SHA}" \
    --manifest    "${MANIFEST}" \
    2>&1 | tee "${EXP3_DIR}/logs/exp3_oracle.log"

echo ""
echo "  ✓ Oracle results: ${ORACLE_JSON}"

# ==========================================================================
#  STEP 3: Compare against exp1b baseline
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DECISION GATE: Chain-Aware Prompt vs Flat Prompt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Exp1b baseline (flat prompt v2):"
echo "    oracle_em = 0.3982    oracle_f1 = 0.5350"
echo "    feasible oracle_em = 0.4372"
echo ""
echo "  Exp3 result (chain-aware prompt v3):"

${PYTHON} -c "
import json
r = json.load(open('${ORACLE_JSON}'))
ov = r['overall']
fs = r.get('subset_docrecall1', {})
print(f\"    oracle_em = {ov['oracle_em']:.4f}    oracle_f1 = {ov['oracle_f1']:.4f}\")
if fs:
    print(f\"    feasible oracle_em = {fs['oracle_em']:.4f}\")

# Decision
delta_em = ov['oracle_em'] - 0.3982
print()
if delta_em >= 0.015:
    print(f\"  → PROCEED: +{delta_em:.4f} EM (≥1.5pp) — chain-aware prompting helps.\")
    print(f\"    Next: run full pipeline (NLI + XGBoost) on exp3 candidates.\")
elif delta_em >= -0.01:
    print(f\"  → NEGLIGIBLE: {delta_em:+.4f} EM — prompt change alone is not enough.\")
    print(f\"    Next: try M=10 candidates (Step 2 in the plan).\")
else:
    print(f\"  → REGRESSED: {delta_em:+.4f} EM — chain labels hurt. Revert.\")
    print(f\"    Next: skip to M=10 with flat prompt instead.\")
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Exp3 complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"