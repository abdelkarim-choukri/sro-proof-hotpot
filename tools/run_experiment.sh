#!/usr/bin/env bash
# run_experiment.sh — Session-aware launcher for the full 3-arm experiment.
#
# Runs all 45 (arch × seed × fold) combinations, 2 at a time across 2 GPUs.
# Checks for existing metrics.json before launching; skips completed runs.
# Safe to re-run after a session ends — resumes from where it left off.
#
# CHECKPOINT BEHAVIOUR (important for 24h server allocations):
#   Each train_one_run.py saves checkpoint_latest.pt after every epoch.
#   When the server kills the job, in-progress runs lose at most one epoch.
#   On the next session, re-run this script — it skips completed runs
#   (metrics.json exists) and resumes interrupted ones from their checkpoint.
#
# 2WIKI NOTE:
#   2Wiki zero-shot eval only runs on fold=0 for each (arch, seed).
#   That is 9 runs instead of 45, saving ~24h with no scientific loss
#   (all folds of the same seed produce identical zero-shot scores because
#   the model never sees 2Wiki during training).
#
# Usage:
#   cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
#   tmux new-session -d -s crosshop_full
#   tmux send-keys -t crosshop_full \
#     "bash tools/run_experiment.sh 2>&1 | tee exp_crosshop/launcher.log" C-m
#   tmux attach -t crosshop_full   # Ctrl+B D to detach without killing
#
# Requirements:
#   tools/train_one_run.py
#   tools/crosshop_model.py
#   tools/crosshop_data.py

# -e removed intentionally: individual run failures are handled by
# the "if wait" loop below and must not abort the entire launcher.
set -uo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these if paths differ
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot"
PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"

# HotpotQA inputs (all confirmed present from sanity run)
GOLD="data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
CANDIDATES="exp5b/candidates/dev_M5_7b_hightemp.jsonl"
CHAINS="exp0c/evidence/dev_K200_chains.jsonl"

# 2Wiki inputs — dev_normalized.json is the correct filename (not dev.json)
WIKI2_GOLD="data/wiki2/raw/dev_normalized.json"
WIKI2_CANDIDATES="exp_wiki2/candidates/dev_M5_sampling.jsonl"
WIKI2_CHAINS="exp_wiki2/evidence/dev_wiki2_chains.jsonl"

ENCODER="microsoft/deberta-v3-base"
OUT_DIR="exp_crosshop/runs"

ARCHS=("A" "B")
SEEDS=(42 123 456)
N_FOLDS=5
N_EPOCHS=8
BATCH=16

# ─────────────────────────────────────────────────────────────────────
# 2WIKI FILE CHECK — done once at startup, NOT inside the loop
# ─────────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
WIKI2_AVAILABLE=true
for F in "${WIKI2_GOLD}" "${WIKI2_CANDIDATES}" "${WIKI2_CHAINS}"; do
    if [[ ! -f "${F}" ]]; then
        echo "[launcher] WARNING: missing 2Wiki file: ${F}"
        echo "[launcher]          2Wiki zero-shot eval will be skipped for all runs."
        WIKI2_AVAILABLE=false
        break
    fi
done

if [[ "${WIKI2_AVAILABLE}" == "true" ]]; then
    echo "[launcher] 2Wiki files confirmed present — fold=0 runs will include zero-shot eval."
fi

# ─────────────────────────────────────────────────────────────────────
# PRE-FLIGHT: verify HotpotQA inputs exist before queuing anything
# ─────────────────────────────────────────────────────────────────────

MISSING=0
for F in "${GOLD}" "${CANDIDATES}" "${CHAINS}"; do
    if [[ ! -f "${F}" ]]; then
        echo "[launcher] ERROR: required HotpotQA file missing: ${F}"
        MISSING=1
    fi
done
if [[ ${MISSING} -eq 1 ]]; then
    echo "[launcher] Aborting — fix missing files before launching."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# BUILD PENDING LIST
# ─────────────────────────────────────────────────────────────────────

mkdir -p "${OUT_DIR}" exp_crosshop/logs

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

TOTAL=30
DONE_COUNT=$((TOTAL - ${#PENDING[@]}))

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Cross-Hop Attention Experiment — Full 3-Arm Run"
echo "  Architectures : A (baseline)  B (lightweight)  C (standard)"
echo "  Seeds         : 42  123  456"
echo "  Folds         : ${N_FOLDS}   Epochs : ${N_EPOCHS}   Batch : ${BATCH}"
echo "  Total runs    : ${TOTAL}   Done : ${DONE_COUNT}   Pending : ${#PENDING[@]}"
echo "══════════════════════════════════════════════════════════════════"
echo ""

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "[launcher] All ${TOTAL} runs already complete."
    echo "[launcher] Running aggregation ..."
    "${PYTHON}" tools/aggregate_results.py \
        --runs_dir "${OUT_DIR}" \
        --out_json exp_crosshop/verdict.json
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# PARALLEL LAUNCHER — 2 GPUs, sequential pairs
# ─────────────────────────────────────────────────────────────────────
# Timing: ~80 min training + ~10 min HotpotQA eval + ~40 min 2Wiki eval
# (fold=0 only) = ~130 min per fold=0 run, ~90 min for fold>0 runs.
# Pairs complete in ~130 min worst case. 23 pairs × 2.2h ≈ 50h total,
# ~25h wall-clock on 2 GPUs — across ~3 server sessions of 8-9h each.
# Each session: re-run this script, it picks up exactly where it stopped.

IDX=0
TOTAL_PENDING=${#PENDING[@]}

while [[ $IDX -lt $TOTAL_PENDING ]]; do

    # Pick up to 2 runs for this slot (one per GPU)
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

        # NEW — skip 2Wiki on ALL folds for now
        WIKI2_FLAG="--skip_wiki2"

        echo "[launcher] Starting  arch=${arch}  seed=${seed}  fold=${fold}  GPU=${gpu}"
        echo "[launcher]   log : ${log}"
        echo "[launcher]   2Wiki: $( [[ -z "${WIKI2_FLAG}" ]] && echo 'YES (fold=0)' || echo 'skipped' )"

        CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" tools/train_one_run.py \
            --arch        "${arch}" \
            --seed        "${seed}" \
            --fold        "${fold}" \
            --gold        "${GOLD}" \
            --candidates  "${CANDIDATES}" \
            --chains      "${CHAINS}" \
            --wiki2_gold        "${WIKI2_GOLD}" \
            --wiki2_candidates  "${WIKI2_CANDIDATES}" \
            --wiki2_chains      "${WIKI2_CHAINS}" \
            ${WIKI2_FLAG} \
            --encoder     "${ENCODER}" \
            --out_dir     "${OUT_DIR}" \
            --n_folds     "${N_FOLDS}" \
            --n_epochs    "${N_EPOCHS}" \
            --batch_candidates "${BATCH}" \
            --gpu         0 \
            > "${log}" 2>&1 &

        PIDS+=($!)
    done

    # Wait for both GPUs to finish before starting the next pair
    for pid in "${PIDS[@]}"; do
        if wait "${pid}"; then
            echo "[launcher] PID ${pid} — OK"
        else
            echo "[launcher] PID ${pid} — FAILED (check log above for details)"
            echo "[launcher]   Re-running this script will retry the failed run."
        fi
    done

    DONE_NOW=$(find "${OUT_DIR}" -name "metrics.json" 2>/dev/null | wc -l)
    echo ""
    echo "[launcher] Progress: ${DONE_NOW}/${TOTAL} complete"
    echo ""

done

# ─────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────

FINAL_DONE=$(find "${OUT_DIR}" -name "metrics.json" 2>/dev/null | wc -l)

echo "══════════════════════════════════════════════════════════════════"
echo "  Launcher finished.  ${FINAL_DONE}/${TOTAL} runs complete."
echo "══════════════════════════════════════════════════════════════════"

if [[ ${FINAL_DONE} -eq ${TOTAL} ]]; then
    echo ""
    echo "[launcher] All runs complete — running aggregation ..."
    "${PYTHON}" tools/aggregate_results.py \
        --runs_dir "${OUT_DIR}" \
        --out_json exp_crosshop/verdict.json
    echo ""
    echo "[launcher] Verdict saved to: exp_crosshop/verdict.json"
else
    REMAINING=$((TOTAL - FINAL_DONE))
    echo ""
    echo "[launcher] ${REMAINING} run(s) did not complete in this session."
    echo "[launcher] Re-run this script in the next session to continue."
    echo "[launcher] Already-complete runs will be skipped automatically."
fi