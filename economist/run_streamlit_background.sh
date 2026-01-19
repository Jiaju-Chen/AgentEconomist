#!/usr/bin/env bash
# 后台启动 Streamlit Web 界面

set -euo pipefail

# 获取脚本所在目录（economist 目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 获取项目根目录（agentsociety-ecosim 目录）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 检查是否已经在运行
if pgrep -f "streamlit run streamlit_app.py" > /dev/null; then
    echo "⚠️  Streamlit 已经在运行中"
    echo "PID: $(pgrep -f 'streamlit run streamlit_app.py')"
    exit 1
fi

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/streamlit_$(date +%Y%m%d_%H%M%S).log"

# 使用 nohup 后台运行
nohup bash "$SCRIPT_DIR/run_streamlit.sh" > "$LOG_FILE" 2>&1 &

# 获取进程 PID
PID=$!
echo "✅ Streamlit 已在后台启动"
echo "PID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "停止服务: kill $PID"
echo "查看进程: ps aux | grep streamlit"





