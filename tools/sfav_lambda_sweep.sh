#!/usr/bin/env bash
# =============================================================================
# sfav_lambda_sweep.sh — λ sensitivity sweep for the SFAV experiment.
#
# Motivation:
#   At λ=0.3, l_sup decays 57× from epoch 1 to epoch 8 (0.145 → 0.0025).
#   By epoch 5 the aux head is effectively silenced and the encoder reverts
#   to collapse. One run (s42_f2) showed Pearson=0.52 and anchor_delta=0.097
#   — genuine structural differentiation — before reverting at epoch 4+.
#   Higher λ should sustain the structural pressure longer.
#
# Design:
#   Fixed: seed=42, fold=2 (most structurally sensitive run at λ=0.3)
#   λ values: 1.0, 3.0
#   Comparison baseline: SFAV_lam0.30_s42_f2 already complete (Pearson=0.52)
#
# What to look for:
#   - Does l_sup decay slower? (epoch_log.jsonl)
#   - Does Pearson stay low (< 0.80)? (metrics.json)
#   - Does EM stay competitive (> 0.60)? (metrics.json)
#   - Does anchor_delta increase? (metrics.json)
#
# Runtime: 2 runs × ~5 hours each / 2 GPUs = ~5 hours total
#
# Usage:
#   cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#   tmux new -s lam_sweep
#   bash tools/sfav_lambda_sweep.sh 2>&1 | tee exp_sfav/lambda_sweep.log
#   Ctrl+B D
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
ENCODER="microsoft/deberta-v3-base"

GOLD="data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
CANDIDATES="exp_distractor/candidates/dev_M5_diverse.jsonl"
CHAINS="exp_distractor/evidence/dev_distractor_chains.jsonl"
OUT_DIR="exp_sfav/runs"

SEED=42
FOLD=2
N_EPOCHS=8
BATCH=16
POS_WEIGHT=2.0

# λ values to sweep — 1.0 and 3.0
LAMS=(1.0 3.0)

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  SFAV λ Sweep"
echo "  seed=${SEED}  fold=${FOLD}"
echo "  λ values: ${LAMS[*]}"
echo "  $(date)"
echo "================================================================"

# Show baseline for comparison
echo ""
echo "  Baseline (λ=0.30, s42_f2):"
if [ -f "${OUT_DIR}/SFAV_lam0.30_s42_f2/metrics.json" ]; then
    ${PYTHON} -c "
import json
m = json.load(open('${OUT_DIR}/SFAV_lam0.30_s42_f2/metrics.json'))
print(f'    EM={m[\"em\"]:.4f}  Pearson={m[\"pearson_flat_minhop\"]:.4f}  CKA={m[\"cka_post\"]:.4f}  anchor_delta={m[\"anchor_delta\"]}')
print(f'    l_sup trajectory: ', end='')
for e in m['epoch_log']:
    print(f'{e[\"avg_sup_loss\"]:.4f}', end=' ')
print()
"
else
    echo "    Not found"
fi
echo ""

# ── Launch both λ runs in parallel ────────────────────────────────────────────
GPU0_PID=""
GPU1_PID=""

for i in "${!LAMS[@]}"; do
    LAM="${LAMS[$i]}"
    GPU=$i   # GPU 0 for lam1.0, GPU 1 for lam3.0

    TAG=$(printf "SFAV_lam%.2f_s%s_f%s" "${LAM}" "${SEED}" "${FOLD}")
    DONE_PATH="${OUT_DIR}/${TAG}/metrics.json"

    if [ -f "${DONE_PATH}" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP (already done): ${TAG}"
        continue
    fi

    mkdir -p "${OUT_DIR}/${TAG}"
    LOG_PATH="${OUT_DIR}/${TAG}/train.log"

    echo "[$(date '+%H:%M:%S')] LAUNCH  ${TAG}  GPU=${GPU}"

    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=${GPU} \
    HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    ${PYTHON} tools/sfav_train_one_run.py \
        --arch SFAV \
        --lam ${LAM} \
        --pos_weight ${POS_WEIGHT} \
        --seed ${SEED} --fold ${FOLD} \
        --gold        ${GOLD} \
        --candidates  ${CANDIDATES} \
        --chains      ${CHAINS} \
        --encoder     ${ENCODER} \
        --out_dir     ${OUT_DIR} \
        --n_folds 5 \
        --n_epochs    ${N_EPOCHS} \
        --batch_candidates ${BATCH} \
        --gpu 0 \
    > "${LOG_PATH}" 2>&1 &

    PID=$!
    if [ "$GPU" == "0" ]; then GPU0_PID=$PID; else GPU1_PID=$PID; fi

    sleep 10
done

# ── Wait for both ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Waiting for both λ runs to complete ..."
[ -n "${GPU0_PID}" ] && wait "${GPU0_PID}" || true
[ -n "${GPU1_PID}" ] && wait "${GPU1_PID}" || true

# ── Results comparison ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  LAMBDA SWEEP RESULTS"
echo "================================================================"
echo ""

${PYTHON} - << 'PYEOF'
import json, os, math

runs_dir = "exp_sfav/runs"
seed, fold = 42, 2
lams = [0.30, 1.0, 3.0]

print(f"  {'λ':>6}  {'EM':>7}  {'Pearson':>9}  {'CKA':>8}  {'anchor_Δ':>10}  {'l_sup_e1':>10}  {'l_sup_e8':>10}")
print(f"  {'─'*75}")

for lam in lams:
    tag = f"SFAV_lam{lam:.2f}_s{seed}_f{fold}"
    path = os.path.join(runs_dir, tag, "metrics.json")
    if not os.path.exists(path):
        print(f"  {lam:>6.2f}  {'MISSING':>7}")
        continue
    m = json.load(open(path))
    el = m.get("epoch_log", [])
    l_sup_e1 = el[0]["avg_sup_loss"] if el else float("nan")
    l_sup_e8 = el[-1]["avg_sup_loss"] if el else float("nan")
    print(
        f"  {lam:>6.2f}  "
        f"{m['em']:>7.4f}  "
        f"{m.get('pearson_flat_minhop', float('nan')):>9.4f}  "
        f"{m.get('cka_post', float('nan')):>8.4f}  "
        f"{m.get('anchor_delta', float('nan')):>10.4f}  "
        f"{l_sup_e1:>10.4f}  "
        f"{l_sup_e8:>10.4f}"
    )

print()
print("  Interpretation guide:")
print("  - Pearson < 0.80 → structural differentiation (good for SFAV claim)")
print("  - l_sup_e8 > 0.02 → aux head still active at end of training")
print("  - EM > 0.60 → structural gain doesn't hurt task performance")
print("  - anchor_delta > 0.05 → hop-2 contribution is meaningful")
PYEOF

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"