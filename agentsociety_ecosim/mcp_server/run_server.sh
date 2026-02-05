#!/usr/bin/env bash

# Get script directory and project root (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate economist >/dev/null
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/all-MiniLM-L6-v2}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agentsociety_ecosim:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/agentsociety_ecosim/mcp_server/server_fastmcp.py" --transport streamable-http --port 8000