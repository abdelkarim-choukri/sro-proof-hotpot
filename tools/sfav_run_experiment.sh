#!/usr/bin/env bash
# =============================================================================
# sfav_run_experiment.sh — Full SFAV experiment launcher.
#
# Runs 30 training jobs across 2 GPUs:
#   2 architectures (A, SFAV) × 3 seeds (42, 123, 456) × 5 folds = 30 runs
#
# Parallelism: 2 jobs at a time — one per GPU. Each job uses one A100.
# Session-aware: skips completed runs (metrics.json exists). Safe to re-run
# after a server allocation ends — resumes from where it left off.
#
# Prerequisites:
#   1. Generation complete:
#      exp_distractor/candidates/dev_M5_diverse.jsonl  (7405 lines)
#   2. vLLM shut down (frees both GPUs for training)
#   3. This script called from project root:
#      cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#      tmux new -s sfav_exp
#      bash tools/sfav_run_experiment.sh 2>&1 | tee exp_sfav/launcher.log
#
# Runtime estimate:
#   ~90 min per run × 30 runs / 2 parallel = ~22.5 hours total
#   With 2× A100: plan for 1 overnight run + 1 daytime run across 2 sessions.
#
# Output:
#   exp_sfav/runs/{arch}_lam0.30_s{seed}_f{fold}/  (SFAV)
#   exp_sfav/runs/A_s{seed}_f{fold}/               (baseline)
#   exp_sfav/launcher.log
# =============================================================================

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
ENCODER="microsoft/deberta-v3-base"

GOLD="data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
CANDIDATES="exp_distractor/candidates/dev_M5_diverse.jsonl"
CHAINS="exp_distractor/evidence/dev_distractor_chains.jsonl"
OUT_DIR="exp_sfav/runs"

ARCHS=("A" "SFAV")
SEEDS=(42 123 456)
N_FOLDS=5
N_EPOCHS=8
BATCH=16
LAM=0.3          # λ for SFAV supporting-fact head
POS_WEIGHT=2.0   # ~50% positive rate measured in label validation → no heavy imbalance

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1   # model is cached — don't hit network during training

# ── Preflight checks ──────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  SFAV Experiment Launcher"
echo "  $(date)"
echo "================================================================"

# Check candidates file exists and has expected size
if [[ ! -f "${CANDIDATES}" ]]; then
    echo "ERROR: Candidates file not found: ${CANDIDATES}"
    echo "       Run sfav_generate_diverse.py first."
    exit 1
fi
N_CANDS=$(wc -l < "${CANDIDATES}")
if [[ ${N_CANDS} -lt 7400 ]]; then
    echo "ERROR: Candidates file only has ${N_CANDS} lines (expected 7405)."
    echo "       Generation may be incomplete."
    exit 1
fi
echo "  Candidates : ${CANDIDATES}  (${N_CANDS} lines ✓)"

# Check chains file
if [[ ! -f "${CHAINS}" ]]; then
    echo "ERROR: Chains file not found: ${CHAINS}"; exit 1
fi
echo "  Chains     : ${CHAINS} ✓"

# Check gold file
if [[ ! -f "${GOLD}" ]]; then
    echo "ERROR: Gold file not found: ${GOLD}"; exit 1
fi
echo "  Gold       : ${GOLD} ✓"

mkdir -p "${OUT_DIR}"
mkdir -p exp_sfav

# ── Build job list ────────────────────────────────────────────────────────────
# Each job is a string: "arch seed fold"
JOBS=()
for ARCH in "${ARCHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for (( FOLD=0; FOLD<N_FOLDS; FOLD++ )); do
            JOBS+=("${ARCH} ${SEED} ${FOLD}")
        done
    done
done

TOTAL=${#JOBS[@]}
echo "  Total jobs : ${TOTAL} (${#ARCHS[@]} arch × ${#SEEDS[@]} seeds × ${N_FOLDS} folds)"
echo ""

# Count already-completed runs
N_DONE=0
for JOB in "${JOBS[@]}"; do
    read -r ARCH SEED FOLD <<< "${JOB}"
    if [[ "${ARCH}" == "SFAV" ]]; then
        TAG=$(printf "SFAV_lam%.2f_s%s_f%s" "${LAM}" "${SEED}" "${FOLD}")
    else
        TAG="A_s${SEED}_f${FOLD}"
    fi
    if [[ -f "${OUT_DIR}/${TAG}/metrics.json" ]]; then
        N_DONE=$((N_DONE + 1))
    fi
done
echo "  Already done: ${N_DONE}/${TOTAL}"
echo "  To run      : $((TOTAL - N_DONE))"
echo ""

# ── GPU parallelism helpers ───────────────────────────────────────────────────
GPU0_PID=""
GPU1_PID=""

wait_for_slot() {
    # Wait until at least one GPU slot is free
    while true; do
        # Check GPU 0
        if [[ -n "${GPU0_PID}" ]]; then
            if ! kill -0 "${GPU0_PID}" 2>/dev/null; then
                wait "${GPU0_PID}" || true
                GPU0_PID=""
            fi
        fi
        # Check GPU 1
        if [[ -n "${GPU1_PID}" ]]; then
            if ! kill -0 "${GPU1_PID}" 2>/dev/null; then
                wait "${GPU1_PID}" || true
                GPU1_PID=""
            fi
        fi
        # Return which slot is free
        if [[ -z "${GPU0_PID}" ]]; then
            echo "0"; return
        fi
        if [[ -z "${GPU1_PID}" ]]; then
            echo "1"; return
        fi
        sleep 30
    done
}

wait_all() {
    if [[ -n "${GPU0_PID}" ]]; then
        wait "${GPU0_PID}" || true
        GPU0_PID=""
    fi
    if [[ -n "${GPU1_PID}" ]]; then
        wait "${GPU1_PID}" || true
        GPU1_PID=""
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
N_LAUNCHED=0
N_SKIPPED=0
T_START=$(date +%s)

for JOB in "${JOBS[@]}"; do
    read -r ARCH SEED FOLD <<< "${JOB}"

    # Build run tag and done path
    if [[ "${ARCH}" == "SFAV" ]]; then
        TAG=$(printf "SFAV_lam%.2f_s%s_f%s" "${LAM}" "${SEED}" "${FOLD}")
        ARCH_FLAG="--arch SFAV --lam ${LAM} --pos_weight ${POS_WEIGHT}"
    else
        TAG="A_s${SEED}_f${FOLD}"
        ARCH_FLAG="--arch A"
    fi
    DONE_PATH="${OUT_DIR}/${TAG}/metrics.json"

    # Skip if already complete
    if [[ -f "${DONE_PATH}" ]]; then
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    # Get a free GPU slot
    GPU=$(wait_for_slot)

    LOG_PATH="${OUT_DIR}/${TAG}/train.log"
    mkdir -p "${OUT_DIR}/${TAG}"

    echo "[$(date '+%H:%M:%S')] LAUNCH  ${TAG}  GPU=${GPU}"

    CMD="CUDA_VISIBLE_DEVICES=${GPU} HF_ENDPOINT=${HF_ENDPOINT} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
        ${PYTHON} tools/sfav_train_one_run.py \
            ${ARCH_FLAG} \
            --seed ${SEED} --fold ${FOLD} \
            --gold        ${GOLD} \
            --candidates  ${CANDIDATES} \
            --chains      ${CHAINS} \
            --encoder     ${ENCODER} \
            --out_dir     ${OUT_DIR} \
            --n_folds     ${N_FOLDS} \
            --n_epochs    ${N_EPOCHS} \
            --batch_candidates ${BATCH} \
            --gpu         0"

    # Launch in background, redirect to per-run log
    eval "PYTHONUNBUFFERED=1 ${CMD}" > "${LOG_PATH}" 2>&1 &
    JOB_PID=$!
    N_LAUNCHED=$((N_LAUNCHED + 1))

    if [[ "${GPU}" == "0" ]]; then
        GPU0_PID="${JOB_PID}"
    else
        GPU1_PID="${JOB_PID}"
    fi

    # Brief pause to stagger GPU memory allocation
    sleep 10
done

# Wait for last two jobs
echo ""
echo "[$(date '+%H:%M:%S')] Waiting for final jobs to complete ..."
wait_all

# ── Summary ───────────────────────────────────────────────────────────────────
T_END=$(date +%s)
ELAPSED=$(( (T_END - T_START) / 60 ))

echo ""
echo "================================================================"
echo "  LAUNCHER DONE  $(date)"
echo "  Launched : ${N_LAUNCHED}"
echo "  Skipped  : ${N_SKIPPED}"
echo "  Elapsed  : ${ELAPSED} min"
echo "================================================================"

# Count final completed runs
N_FINAL=0
for JOB in "${JOBS[@]}"; do
    read -r ARCH SEED FOLD <<< "${JOB}"
    if [[ "${ARCH}" == "SFAV" ]]; then
        TAG=$(printf "SFAV_lam%.2f_s%s_f%s" "${LAM}" "${SEED}" "${FOLD}")
    else
        TAG="A_s${SEED}_f${FOLD}"
    fi
    if [[ -f "${OUT_DIR}/${TAG}/metrics.json" ]]; then
        N_FINAL=$((N_FINAL + 1))
    fi
done
echo "  Completed runs: ${N_FINAL}/${TOTAL}"

if [[ ${N_FINAL} -lt ${TOTAL} ]]; then
    echo ""
    echo "  Incomplete runs:"
    for JOB in "${JOBS[@]}"; do
        read -r ARCH SEED FOLD <<< "${JOB}"
        if [[ "${ARCH}" == "SFAV" ]]; then
            TAG=$(printf "SFAV_lam%.2f_s%s_f%s" "${LAM}" "${SEED}" "${FOLD}")
        else
            TAG="A_s${SEED}_f${FOLD}"
        fi
        if [[ ! -f "${OUT_DIR}/${TAG}/metrics.json" ]]; then
            echo "    MISSING: ${TAG}"
        fi
    done
fi

echo ""
echo "  Next step: run sfav_aggregate.py to collect all metrics."
echo "================================================================"