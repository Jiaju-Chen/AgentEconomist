#!/bin/bash

# Navigate to the agent directory
cd "$(dirname "$0")/../agent" || exit 1

# 使用 ecosim 环境，不创建新的虚拟环境
# 检查 ecosim 环境是否存在
if [ -d "/root/miniconda3/envs/ecosim" ]; then
    echo "✅ Using existing ecosim conda environment"
    # 不需要安装，因为 AgentEconomist 的依赖已经在 ecosim 中
    exit 0
else
    echo "❌ ecosim environment not found, falling back to uv"
    uv sync
fi