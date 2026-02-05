#!/bin/bash

# Navigate to the agent directory
cd "$(dirname "$0")/../agent" || exit 1

# 使用 economist 环境，不创建新的虚拟环境
# 检查 economist 环境是否存在
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
ECONOMIST_ENV="$CONDA_BASE/envs/economist"

if [ -d "$ECONOMIST_ENV" ]; then
    echo "✅ Using existing economist conda environment"
    # 不需要安装，因为 AgentEconomist 的依赖已经在 economist 中
    exit 0
else
    echo "⚠️  economist environment not found, falling back to uv"
    echo "   请先创建 economist 环境: conda create -n economist python=3.11 -y"
    uv sync
fi