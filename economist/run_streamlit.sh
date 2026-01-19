#!/usr/bin/env bash
# 启动 Streamlit Web 界面

set -euo pipefail

# 获取脚本所在目录（economist 目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 获取项目根目录（agentsociety-ecosim 目录）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ecosim >/dev/null
fi

# 设置 PYTHONPATH（使用相对路径）
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:${PYTHONPATH:-}"

# 加载 .env 文件（如果存在）
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# 检查环境变量
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "⚠️  警告: 未设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY" >&2
    echo "请先设置环境变量或创建 .env 文件" >&2
    echo "继续启动 Streamlit，但 Agent 可能无法正常工作..." >&2
fi

# 启动 Streamlit
streamlit run streamlit_app.py --server.port 8503 --server.address 0.0.0.0




