# 自动化实验工作流指南

## 概述

系统现在支持根据问题自动生成实验配置、启动实验、分析结果的完整工作流。Agent可以通过调用工具链自动完成整个实验流程。

## 支持的问题类型

系统可以自动识别以下问题类型：

1. **创新政策（Innovation）**: "How do innovation-promoting policies shape economic performance?"
   - 自动调整创新相关参数
   - 提高创新概率和改进幅度
   
2. **再分配政策（Redistribution）**: "How will a universal basic income policy affect people's lives?"
   - 自动调整再分配策略参数
   - 修改redistribution相关权重
   
3. **劳动力市场（Labor Productivity）**: "How will AI agents reshape the labor market?"
   - 自动提高生产效率参数
   - 调整劳动生产率和弹性
   
4. **关税政策（Tariff）**: "How will a breaking news event such as the Liberation Day tariff affect the stock market?"
   - 自动调整税收和市场参数
   - 模拟政策冲击

## 自动化工作流步骤

### 步骤1: 分析问题
**工具**: `analyze_question(question: str)`

识别问题类型并返回推荐的参数配置。

```python
result = analyze_question("How will AI agents reshape the labor market?")
# 返回: 问题类型、关键词、推荐参数
```

### 步骤2: 生成配置
**工具**: `generate_config_from_question(question: str)`

根据问题自动生成YAML配置文件和参数设置。

```python
config = generate_config_from_question("How will AI agents reshape the labor market?")
# 返回: 配置名称、参数列表、指导说明
```

### 步骤3: 保存配置
**工具**: `save_current_config_to_yaml(config_name: str, description: str)`

将推荐的参数保存为YAML文件。

```python
# 首先设置参数
batch_set_parameters(config["parameters"])
# 然后保存配置
save_current_config_to_yaml(config["config_name"], config["description"])
```

### 步骤4: 加载配置
**工具**: `load_yaml_config(config_name: str)`

加载并应用YAML配置。

```python
load_yaml_config(config["config_name"])
```

### 步骤5: 启动仿真
**工具**: `start_simulation()`

启动仿真实验。

```python
start_simulation()
```

### 步骤6: 监控状态
**工具**: `get_simulation_status()`

监控仿真运行状态，直到完成。

```python
status = get_simulation_status()
# 检查 status["running"] == False
```

### 步骤7: 捕捉实验
**工具**: `capture_experiment(experiment_name: str, status: str)`

捕捉实验目录（从仿真输出获取实验名称，格式：`exp_100h_12m_YYYYMMDD_HHMMSS`）。

```python
capture_experiment("exp_100h_12m_20251122_101530", "completed")
```

### 步骤8: 分析实验
**工具**: `analyze_experiment(experiment_name: str, include_innovation: bool)`

分析实验数据，获取指标。

```python
analysis = analyze_experiment("exp_100h_12m_20251122_101530", include_innovation=True)
# 返回: 宏观指标、微观指标、创新指标（如果适用）
```

### 步骤9: 获取结果
**工具**: `get_analysis_result(experiment_name: str)`

获取完整的分析结果。

```python
results = get_analysis_result("exp_100h_12m_20251122_101530")
```

## 完整工作流工具

**工具**: `get_experiment_workflow(question: str)`

获取完整的实验工作流指导，包含所有步骤的详细说明。

```python
workflow = get_experiment_workflow("How will AI agents reshape the labor market?")
# 返回: 完整的步骤列表、工具调用、参数设置
```

## 各问题类型的配置指导

### 创新政策（Innovation）
- **参数调整**:
  - `enable_innovation`: True
  - `innovation_probability`: 0.15（提高创新概率）
  - `innovation_*_improvement`: 0.15（提高创新改进幅度）
- **建议**: 启用竞争市场模式（`enable_competitive_market`）
- **分析指标**: 创新事件数、创新与市场占有率相关性、GDP增长、企业生产率

### 再分配政策（Redistribution）
- **参数调整**:
  - `redistribution_strategy`: "progressive" 或 "aggressive"
  - `redistribution_poverty_weight`: 0.5（增加贫困权重）
  - `redistribution_unemployment_weight`: 0.3
  - `redistribution_family_size_weight`: 0.2
- **建议**: 可调整个人所得税率（`income_tax_rate`）为再分配提供资金来源
- **分析指标**: 消费者总支出、总属性值、GDP、收入分布（基尼系数）、家庭储蓄率

### 劳动力市场（Labor Productivity）
- **参数调整**:
  - `labor_productivity_factor`: 150.0-200.0（默认100）
  - `labor_elasticity`: 0.8-0.9（提高劳动弹性）
  - `dismissal_rate`: 0.05（降低裁员率，AI提高效率可能降低裁员需求）
  - `enable_dynamic_job_posting`: True
- **分析指标**: 劳动生产率、失业率、GDP、企业利润率、市场占有率

### 关税政策（Tariff）
- **参数调整**:
  - `vat_rate`: 0.25（提高增值税率）
  - `corporate_tax_rate`: 0.45
  - `enable_price_adjustment`: True
  - `price_adjustment_rate`: 0.15（提高价格调整速率）
- **分析指标**: GDP、总支出、价格变化、企业利润、市场占有率变化

## Agent使用示例

Agent可以使用以下流程自动完成实验：

1. **分析问题** → `analyze_question(question)`
2. **获取工作流** → `get_experiment_workflow(question)` 
3. **生成配置** → `generate_config_from_question(question)`
4. **设置参数** → `batch_set_parameters(parameters)`
5. **保存配置** → `save_current_config_to_yaml(config_name, description)`
6. **加载配置** → `load_yaml_config(config_name)`
7. **启动仿真** → `start_simulation()`
8. **监控状态** → `get_simulation_status()`（循环直到完成）
9. **捕捉实验** → `capture_experiment(experiment_name, "completed")`
10. **分析实验** → `analyze_experiment(experiment_name, include_innovation)`
11. **获取结果** → `get_analysis_result(experiment_name)`
12. **得出结论** → 基于分析结果得出结论

## 注意事项

1. **实验名称格式**: 仿真会自动生成实验目录，格式为 `exp_{households}h_{months}m_{timestamp}`
2. **参数验证**: 所有参数设置都会自动验证合法性
3. **分析指标**: 根据问题类型选择不同的分析指标
4. **错误处理**: 每个步骤都包含错误处理，失败时会返回错误信息

## 扩展性

系统设计具有良好的扩展性：
- 可以轻松添加新的问题类型识别规则
- 可以为每种问题类型定制参数配置
- 可以添加新的分析指标

