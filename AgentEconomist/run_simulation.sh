#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录（AgentEconomist 目录）
AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$AGENT_DIR/.." && pwd)"

# 实际的仿真脚本在项目根目录
SIM_SCRIPT="$PROJECT_ROOT/agentsociety_ecosim/run_sim_with_yaml.sh"

# 确保仿真脚本可执行
if [[ ! -x "$SIM_SCRIPT" ]]; then
  chmod +x "$SIM_SCRIPT"
fi

# 获取配置文件路径（第一个参数）
CONFIG_FILE="${1:-}"

if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
  # 提取目录和文件名（不含扩展名）
  CONFIG_DIR=$(dirname "$CONFIG_FILE")
  CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
  LOG_FILE="$CONFIG_DIR/${CONFIG_BASENAME}.log"
  
  # 创建目录（如果不存在）
  mkdir -p "$CONFIG_DIR"
  
  # 运行仿真并输出到日志文件
  "$SIM_SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
else
  # 如果没有有效的配置文件，正常运行
  exec "$SIM_SCRIPT" "$@"
fi
