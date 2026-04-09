#!/usr/bin/env bash
# ==========================================================================
#  exp_phaseA/A1.1/download_deberta_squad2.sh
#  Downloads deepset/deberta-v3-base-squad2 via hf-mirror.com
#
#  Run once before run_qa_hop_score.sh
#  Target: /var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2
# ==========================================================================

set -euo pipefail

PYTHON="/var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3"
MODEL_ID="deepset/deberta-v3-base-squad2"
LOCAL_DIR="/var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2"

echo "================================================================"
echo "  Downloading: ${MODEL_ID}"
echo "  Target:      ${LOCAL_DIR}"
echo "  Mirror:      https://hf-mirror.com"
echo "================================================================"
echo ""

HF_ENDPOINT=https://hf-mirror.com ${PYTHON} -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_ID}',
    local_dir='${LOCAL_DIR}',
    local_dir_use_symlinks=False,
)
print('Done')
"

echo ""
echo "Verifying files:"
ls -lh "${LOCAL_DIR}/"
echo ""
echo "================================================================"
echo "  Download complete."
echo "  Model path for --model flag:"
echo "  ${LOCAL_DIR}"
echo "================================================================"