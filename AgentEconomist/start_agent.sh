#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate economist
  echo "✅ 已激活 conda 环境: economist"
else
  echo "⚠️  未找到 conda，使用当前 Python 环境"
fi

# 检查 .env 文件（项目根目录，统一配置）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "❌ 错误: .env 文件不存在"
  echo "   请复制 .env.example 到项目根目录并配置 API Key"
  echo "   项目根目录: $PROJECT_ROOT"
  exit 1
fi

# 设置项目根目录环境变量
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "📍 项目根目录: $PROJECT_ROOT"
echo "🚀 启动 LangGraph Agent Server..."
echo ""

# 启动 LangGraph Server
# 注意：如果 langgraph dev 不可用，使用 start_server.py
if command -v langgraph >/dev/null 2>&1; then
    # 0.0.0.0 只能用于监听(bind)，浏览器连接时不允许用它作为域名
    # 本机开发默认用 127.0.0.1；如需局域网访问可 export LANGGRAPH_HOST=0.0.0.0 或真实 IP
    exec langgraph dev --port "${LANGGRAPH_PORT:-8123}" --host "${LANGGRAPH_HOST:-127.0.0.1}"
else
    echo "⚠️  langgraph CLI 不可用，使用 Python 启动脚本"
    exec python "$SCRIPT_DIR/start_server.py"
fi
