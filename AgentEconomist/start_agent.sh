#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ecosim
  echo "✅ 已激活 conda 环境: ecosim"
else
  echo "⚠️  未找到 conda，使用当前 Python 环境"
fi

# 检查 .env 文件
if [[ ! -f .env ]]; then
  echo "❌ 错误: .env 文件不存在"
  echo "   请复制 env.example 并配置 API Key"
  exit 1
fi

# 设置项目根目录环境变量
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "📍 项目根目录: $PROJECT_ROOT"
echo "🚀 启动 LangGraph Agent Server..."
echo ""

# 启动 LangGraph Server
exec langgraph dev --port 8123 --host 0.0.0.0
