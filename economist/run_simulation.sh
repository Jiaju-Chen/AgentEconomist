#!/usr/bin/env bash
set -euo pipefail
# 获取脚本所在目录（economist 目录），然后获取项目根目录
ECONOMIST_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$ECONOMIST_DIR/.." && pwd)"
SIM_SCRIPT="$SCRIPT_DIR/run_sim_with_yaml.sh"

if [[ ! -x "$SIM_SCRIPT" ]]; then
  chmod +x "$SIM_SCRIPT"
fi

# Get the config file path (first argument)
CONFIG_FILE="${1:-}"

if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
  # Extract directory and basename (without extension) from config file
  CONFIG_DIR=$(dirname "$CONFIG_FILE")
  CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
  LOG_FILE="$CONFIG_DIR/${CONFIG_BASENAME}.log"
  
  # Create directory if it doesn't exist
  mkdir -p "$CONFIG_DIR"
  
  # Run simulation and output to log file
  "$SIM_SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
else
  # If no valid config file, just run normally
exec "$SIM_SCRIPT" "$@"
fi
