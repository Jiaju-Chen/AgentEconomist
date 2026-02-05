#!/usr/bin/env bash
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"
set -euo pipefail

# Get script directory and project root (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 脚本现在在 agentsociety_ecosim/ 目录，需要回到上一级
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/default.yaml"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate economist >/dev/null
fi

export MODEL_PATH="${PROJECT_ROOT}/model/all-MiniLM-L6-v2"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agentsociety_ecosim:${PYTHONPATH:-}"
export ECOSIM_PROJECT_ROOT="$PROJECT_ROOT"

CONFIG_INPUT="${1:-$DEFAULT_CONFIG}"

if [[ ! -f "$CONFIG_INPUT" ]]; then
  echo "❌ 配置文件不存在: $CONFIG_INPUT" >&2
  exit 1
fi

CONFIG_FILE="$(python - "$CONFIG_INPUT" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"

export ECOSIM_CONFIG_FILE="$CONFIG_FILE"

python - <<'PY'
import asyncio
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ["ECOSIM_PROJECT_ROOT"]).resolve()
CONFIG_PATH = Path(os.environ["ECOSIM_CONFIG_FILE"]).resolve()

import sys  # noqa: E402
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentsociety_ecosim.simulation.joint_debug_test import EconomicSimulation, SimulationConfig

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    raw_config = yaml.safe_load(file) or {}

fields = SimulationConfig.__dataclass_fields__.keys()
overrides = {}

def collect_leaves(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                collect_leaves(value)
            else:
                if key in fields:
                    overrides[key] = value

collect_leaves(raw_config)

# 确保创新模块相关字段在 overrides 中可用
innovation_fields = ("enable_innovation_module", "innovation_gamma", "policy_encourage_innovation", "innovation_lambda", "innovation_concavity_beta")
innovation_config = raw_config.get("innovation", {})
for key in innovation_fields:
    if key in innovation_config and key not in overrides:
        overrides[key] = innovation_config[key]

config = SimulationConfig(**overrides)
simulation = EconomicSimulation(config)

async def run():
    if not await simulation.setup_simulation_environment():
        raise RuntimeError("仿真环境设置失败")
    await simulation.run_simulation()
    report = await simulation.generate_simulation_report()
    await simulation.save_simulation_report(report)
    await simulation.cleanup_resources()

asyncio.run(run())
PY