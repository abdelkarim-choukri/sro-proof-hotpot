#!/usr/bin/env bash
# run_experiment.sh — Session-aware launcher for the full 3-arm experiment.
#
# Runs all 45 (arch × seed × fold) combinations, 2 at a time across 2 GPUs.
# Checks for existing metrics.json before launching; skips completed runs.
# Safe to re-run after a session ends — resumes from where it left off.
#
# Usage:
#   cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#   bash tools/run_experiment.sh 2>&1 | tee exp_crosshop/launcher.log
#
# Requirements:
#   - tmux must be running (so script survives session timeouts)
#   - The sanity run must have passed
#   - All three tools/ files deployed:
#       tools/train_one_run.py
#       tools/crosshop_model.py
#       tools/crosshop_data.py

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

GOLD="data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
CANDIDATES="exp5b/candidates/dev_M5_7b_hightemp.jsonl"
CHAINS="exp0c/evidence/dev_K200_chains.jsonl"
WIKI2_GOLD="data/wiki2/raw/dev.json"
WIKI2_CANDIDATES="exp_wiki2/candidates/dev_M5_sampling.jsonl"
WIKI2_CHAINS="exp_wiki2/evidence/dev_wiki2_chains.jsonl"
ENCODER="microsoft/deberta-v3-base"
OUT_DIR="exp_crosshop/runs"

ARCHS=("A" "B" "C")
SEEDS=(42 123 456)
N_FOLDS=5
N_EPOCHS=8
BATCH=16

# Detect whether 2Wiki files exist; skip if any missing
SKIP_WIKI2_FLAG=""
for F in "$WIKI2_GOLD" "$WIKI2_CANDIDATES" "$WIKI2_CHAINS"; do
    if [[ ! -f "${PROJECT_ROOT}/${F}" ]]; then
        echo "[launcher] WARNING: missing 2Wiki file: ${F} — will pass --skip_wiki2"
        WIKI2_FLAG="--skip_wiki2"
        [[ "${fold}" == "0" ]] && WIKI2_FLAG=""
        break
    fi
done

# ─────────────────────────────────────────────────────────────────────
# BUILD COMBINATION LIST
# ─────────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}" exp_crosshop/logs

# Collect all 45 run tags that are NOT yet complete
PENDING=()
for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for ((fold=0; fold<N_FOLDS; fold++)); do
            tag="${arch}_s${seed}_f${fold}"
            done_file="${OUT_DIR}/${tag}/metrics.json"
            if [[ ! -f "${done_file}" ]]; then
                PENDING+=("${arch}|${seed}|${fold}")
            fi
        done
    done
done

TOTAL=45
DONE=$((TOTAL - ${#PENDING[@]}))
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Cross-Hop Attention Experiment — Full 3-Arm Run"
echo "  Total combinations : ${TOTAL}   Done : ${DONE}   Pending : ${#PENDING[@]}"
echo "══════════════════════════════════════════════════════════════════"
echo ""

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "[launcher] All 45 runs complete. Run aggregate_results.py next."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# PARALLEL LAUNCHER (2 GPUs)
# ─────────────────────────────────────────────────────────────────────

# We run pairs: GPU 0 and GPU 1 simultaneously.
# Each run takes ~11-12 hours → a pair takes ~11-12 hours wall-clock.
# With 2 GPUs and 45 runs, total = ceil(45/2) * 12h ≈ 2.3 wall-clock days.

IDX=0
TOTAL_PENDING=${#PENDING[@]}

while [[ $IDX -lt $TOTAL_PENDING ]]; do
    # Pick up to 2 runs for this slot
    SLOT=()
    for GPU in 0 1; do
        if [[ $IDX -lt $TOTAL_PENDING ]]; then
            SLOT+=("${GPU}|${PENDING[$IDX]}")
            IDX=$((IDX + 1))
        fi
    done

    PIDS=()
    for entry in "${SLOT[@]}"; do
        gpu="${entry%%|*}"
        rest="${entry#*|}"
        arch="${rest%%|*}"
        rest2="${rest#*|}"
        seed="${rest2%%|*}"
        fold="${rest2##*|}"
        tag="${arch}_s${seed}_f${fold}"
        log="exp_crosshop/logs/${tag}.log"

        echo "[launcher] Starting arch=${arch} seed=${seed} fold=${fold}  GPU=${gpu}  log=${log}"

        CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" tools/train_one_run.py \
            --arch "${arch}" \
            --seed "${seed}" \
            --fold "${fold}" \
            --gold        "${GOLD}" \
            --candidates  "${CANDIDATES}" \
            --chains      "${CHAINS}" \
            --wiki2_gold       "${WIKI2_GOLD}" \
            --wiki2_candidates "${WIKI2_CANDIDATES}" \
            --wiki2_chains     "${WIKI2_CHAINS}" \
            ${SKIP_WIKI2_FLAG} \
            --encoder     "${ENCODER}" \
            --out_dir     "${OUT_DIR}" \
            --n_folds  "${N_FOLDS}" \
            --n_epochs "${N_EPOCHS}" \
            --batch_candidates "${BATCH}" \
            --gpu "${gpu}" \
            > "${log}" 2>&1 &

        PIDS+=($!)
    done

    # Wait for both to finish before starting the next pair
    for pid in "${PIDS[@]}"; do
        if wait "${pid}"; then
            echo "[launcher] PID ${pid} finished OK"
        else
            echo "[launcher] PID ${pid} FAILED — check logs"
        fi
    done

    # Progress report
    DONE_NOW=$(find "${OUT_DIR}" -name "metrics.json" | wc -l)
    echo ""
    echo "[launcher] Progress: ${DONE_NOW}/${TOTAL} complete  (slot ended)"
    echo ""
done

echo "══════════════════════════════════════════════════════════════════"
echo "  All runs complete. Running aggregation ..."
echo "══════════════════════════════════════════════════════════════════"

"${PYTHON}" tools/aggregate_results.py \
    --runs_dir "${OUT_DIR}" \
    --out_json exp_crosshop/verdict.json