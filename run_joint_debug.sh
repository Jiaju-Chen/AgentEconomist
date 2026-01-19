#!/usr/bin/env bash

set -euo pipefail

# Get script directory and project root (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ecosim >/dev/null
fi

export MODEL_PATH="${PROJECT_ROOT}/model/all-MiniLM-L6-v2"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agentsociety_ecosim:${PYTHONPATH:-}"

python agentsociety_ecosim/simulation/joint_debug_test.py "$@"
