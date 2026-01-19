# YAML配置文件使用指南

## 1. 参数管理器YAML功能

### 功能概述

参数管理器完全支持YAML文件的生成和加载：

1. ✅ **生成YAML文件**: `save_current_config_to_yaml()` - 将当前配置保存为YAML
2. ✅ **加载YAML文件**: `load_yaml_config()` - 从YAML文件加载配置
3. ✅ **应用YAML配置**: `apply_yaml_config()` - 加载并应用到当前config对象
4. ✅ **列出YAML配置**: `list_yaml_configs()` - 列出所有可用的YAML配置文件

### 保存当前配置为YAML

```python
from parameter_manager import ParameterManager
from simulation.joint_debug_test import SimulationConfig

config = SimulationConfig()
param_manager = ParameterManager.get_instance(config=config)

# 保存当前配置
result = param_manager.save_current_config_to_yaml(
    config_name="my_config",
    description="我的自定义配置"
)

# 结果:
# {
#   "success": True,
#   "message": "Config saved to ...",
#   "path": "/path/to/config/my_config.yaml"
# }
```

### 加载YAML配置文件

```python
# 加载YAML配置（不应用）
params = param_manager.load_yaml_config("my_config")

# 加载并应用YAML配置
result = param_manager.apply_yaml_config("my_config", validate=True)

# 结果:
# {
#   "success": True,
#   "loaded_parameters": 50,
#   "applied_parameters": 50,
#   "errors": []
# }
```

### MCP工具支持

通过MCP服务器可以使用以下工具：

1. **save_current_config_to_yaml**: 保存当前配置为YAML
2. **load_yaml_config**: 加载并应用YAML配置
3. **list_yaml_configs**: 列出所有YAML配置文件

## 2. YAML文件格式

### 文件结构

YAML配置文件按类别组织参数：

```yaml
# 配置描述（作为注释）
# 生成时间: 2024-11-22

# ==================== 税收政策 ====================
tax_policy:
  income_tax_rate: 0.45
  vat_rate: 0.20
  corporate_tax_rate: 0.42

# ==================== 劳动力市场 ====================
labor_market:
  dismissal_rate: 0.1
  enable_dismissal: true
  unemployment_threshold: 0.4

# ==================== 生产参数 ====================
production:
  labor_productivity_factor: 100.0
  labor_elasticity: 0.7

# ==================== 系统规模 ====================
system_scale:
  num_households: 100
  num_iterations: 12
  random_state: 42

# ... 更多类别
```

### 现有配置文件

在 `mcp_server/config/` 目录下已存在以下配置文件：

1. **default.yaml**: 默认配置
2. **high_tax_scenario.yaml**: 高税收场景
3. **crisis_scenario.yaml**: 危机场景
4. **low_tax_scenario.yaml**: 低税收场景

## 3. 模型运行时的YAML支持

### 当前状态

⚠️ **当前模型运行脚本 (`joint_debug_test.py`) 还不支持直接从YAML文件加载参数**

### 当前模型初始化方式

模型目前通过以下方式初始化：

```python
# joint_debug_test.py
@dataclass
class SimulationConfig:
    """仿真配置类"""
    num_households: int = 100
    num_iterations: int = 12
    income_tax_rate: float = 0.45
    # ... 更多参数

# 创建配置对象（使用默认值）
config = SimulationConfig()

# 运行仿真
simulator = EconomicSimulator(config)
simulator.run()
```

### 如何让模型支持YAML配置

有两种方式让模型支持从YAML文件加载配置：

#### 方式1: 通过MCP服务器设置参数后运行

1. 启动MCP服务器
2. 通过MCP工具加载YAML配置：
   ```
   load_yaml_config("my_config")
   ```
3. 参数已经应用到config对象
4. 运行模型（使用修改后的config对象）

#### 方式2: 修改模型脚本支持YAML文件（推荐）

在模型运行脚本中添加YAML支持：

```python
# 在 joint_debug_test.py 中添加
import yaml
from pathlib import Path
import argparse

def load_config_from_yaml(yaml_file: str) -> SimulationConfig:
    """从YAML文件加载配置"""
    from agentsociety_ecosim.mcp_server.parameter_manager import ParameterManager
    
    # 创建默认配置
    config = SimulationConfig()
    
    # 初始化参数管理器
    param_manager = ParameterManager.get_instance(config=config)
    
    # 加载并应用YAML配置
    config_name = Path(yaml_file).stem  # 获取文件名（不含扩展名）
    result = param_manager.apply_yaml_config(config_name, validate=True)
    
    if not result["success"]:
        raise ValueError(f"Failed to load config: {result['errors']}")
    
    return config

# 修改主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    args = parser.parse_args()
    
    if args.config:
        # 从YAML文件加载配置
        config = load_config_from_yaml(args.config)
    else:
        # 使用默认配置
        config = SimulationConfig()
    
    # 运行仿真
    simulator = EconomicSimulator(config)
    simulator.run()

if __name__ == "__main__":
    main()
```

### 使用方式

```bash
# 使用YAML配置文件运行模型
python joint_debug_test.py --config config/my_config.yaml

# 或使用默认配置
python joint_debug_test.py
```

## 4. 推荐的工作流

### 自动化实验工作流

对于自动化实验，推荐使用以下流程：

```
1. Agent分析问题
   → analyze_question("How will AI agents reshape the labor market?")

2. 生成配置
   → generate_config_from_question(question)
   → 返回推荐的参数配置

3. 设置参数
   → batch_set_parameters(recommended_parameters)

4. 保存配置为YAML
   → save_current_config_to_yaml(config_name, description)

5. 加载配置（可选，验证）
   → load_yaml_config(config_name)

6. 运行模型（使用当前config对象）
   → start_simulation()
   → 或手动运行: python joint_debug_test.py

7. 捕捉实验
   → capture_experiment(experiment_name, "completed")

8. 分析实验
   → analyze_experiment(experiment_name)
```

### 手动使用YAML配置

如果模型脚本支持YAML，可以直接：

```bash
# 1. 通过MCP工具保存配置
# 调用 save_current_config_to_yaml("labor_productivity_policy")

# 2. 直接使用YAML文件运行模型
python joint_debug_test.py --config config/labor_productivity_policy.yaml
```

## 5. 总结

### ✅ 已支持的功能

1. **参数管理器完全支持YAML**:
   - ✅ 生成YAML文件
   - ✅ 加载YAML文件
   - ✅ 应用YAML配置
   - ✅ 列出YAML配置

2. **MCP工具支持**:
   - ✅ `save_current_config_to_yaml`
   - ✅ `load_yaml_config`
   - ✅ `list_yaml_configs`

### ⚠️ 需要改进的功能

1. **模型脚本支持YAML**:
   - ⚠️ 当前模型运行脚本还不支持直接从YAML文件加载
   - 💡 建议：添加命令行参数支持YAML文件

2. **改进建议**:
   - 修改 `joint_debug_test.py` 添加 `--config` 参数
   - 或者通过MCP设置参数后运行（参数已应用到config对象）

### 当前可用方案

即使模型脚本不支持YAML文件，仍然可以通过以下方式使用YAML配置：

1. **通过MCP工具**:
   - 使用 `load_yaml_config()` 加载配置
   - 参数已应用到config对象
   - 通过 `start_simulation()` 运行（如果支持）
   - 或手动运行模型（使用修改后的config对象）

2. **手动方式**:
   - 通过MCP工具设置参数
   - 保存为YAML文件（用于记录）
   - 运行模型（使用已设置的参数）

