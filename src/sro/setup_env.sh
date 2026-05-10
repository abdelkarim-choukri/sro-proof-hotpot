#!/usr/bin/env bash
# scripts/sro/setup_env.sh — install SRO deps via Chinese mirrors.
set -euo pipefail

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR="https://hf-mirror.com"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Installing SRO dependencies via Tsinghua mirror..."
pip install -r requirements_sro.txt -i "${PIP_MIRROR}"

echo ""
echo "Configuring NLTK punkt for sentence splitting..."
python3 - <<'PY'
import os
import nltk
# Use a project-local NLTK data dir to avoid touching global state
nltk_dir = os.path.join(os.environ.get("REPO_ROOT", "."), ".nltk_data")
os.makedirs(nltk_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_dir, quiet=True)
print(f"NLTK data → {nltk_dir}")
PY

echo ""
echo "HuggingFace mirror: ${HF_MIRROR}"
echo "Always export HF_ENDPOINT before any HF download:"
echo ""
echo "    export HF_ENDPOINT=${HF_MIRROR}"
echo ""
echo "Adding to ~/.bashrc (idempotent)..."
LINE="export HF_ENDPOINT=${HF_MIRROR}"
grep -qxF "${LINE}" "${HOME}/.bashrc" 2>/dev/null || echo "${LINE}" >> "${HOME}/.bashrc"

# Also write a project-local activate snippet that any script can `source`
cat > scripts/sro/activate_env.sh <<EOF
# Source this file in any SRO script before HF or pip calls.
export HF_ENDPOINT="${HF_MIRROR}"
export NLTK_DATA="\${NLTK_DATA:-\$(pwd)/.nltk_data}"
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"
EOF
chmod +x scripts/sro/activate_env.sh

echo ""
echo "Setup complete."
echo "  • pip mirror:    ${PIP_MIRROR}"
echo "  • HF mirror:     ${HF_MIRROR}"
echo "  • In any new shell, run:  source scripts/sro/activate_env.sh"