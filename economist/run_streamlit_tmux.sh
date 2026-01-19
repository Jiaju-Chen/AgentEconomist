#!/usr/bin/env bash
# 使用 tmux 后台启动 Streamlit Web 界面

set -euo pipefail

# 获取脚本所在目录（economist 目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 获取项目根目录（agentsociety-ecosim 目录）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# tmux session 名称
SESSION_NAME="streamlit-app"

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 检查 tmux 是否安装
if ! command -v tmux >/dev/null 2>&1; then
    echo "❌ 错误: 未安装 tmux"
    echo "请先安装: sudo apt-get install tmux 或 sudo yum install tmux"
    exit 1
fi

# 检查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' 已存在"
    echo ""
    echo "可用操作："
    echo "  查看: tmux attach -t $SESSION_NAME"
    echo "  停止: tmux kill-session -t $SESSION_NAME"
    echo "  重启: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 || ss -tlnp | grep -q ':8501 '; then
    echo "⚠️  警告: 端口 8501 已被占用"
    echo "请先停止占用该端口的进程，或修改脚本中的端口号"
    exit 1
fi

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/streamlit_$(date +%Y%m%d_%H%M%S).log"

# 创建新的 tmux session 并在其中运行 Streamlit
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" \
    "bash -c 'exec bash $SCRIPT_DIR/run_streamlit.sh 2>&1 | tee $LOG_FILE'"

# 等待一下确保启动成功
sleep 2

# 检查 session 是否运行
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ Streamlit 已在 tmux session 中启动"
    echo ""
    echo "Session 名称: $SESSION_NAME"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "常用命令："
    echo "  查看实时输出: tmux attach -t $SESSION_NAME"
    echo "  退出查看（不停止）: 按 Ctrl+B，然后按 D"
    echo "  停止服务: tmux kill-session -t $SESSION_NAME"
    echo "  查看日志: tail -f $LOG_FILE"
    echo "  查看所有 session: tmux ls"
else
    echo "❌ 启动失败，请检查日志: $LOG_FILE"
    exit 1
fi





