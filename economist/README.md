# Design Agent - 经济系统设计智能体

## 📖 概述

Design Agent 是一个基于 AgentScope 框架构建的智能经济系统设计代理，专门用于通过受控实验进行科学研究。它能够：

- 🔍 **自动检索学术文献**：基于研究问题检索相关学术论文
- 🧪 **设计受控实验**：创建控制组和实验组配置
- 🚀 **运行经济仿真**：执行大规模经济系统仿真
- 📊 **分析实验结果**：提取关键经济指标并生成对比报告
- 💡 **得出结论**：基于统计差异得出科学结论

## 🚀 快速开始

### 前置要求

1. **Python 环境**：Python 3.8 或更高版本
2. **依赖管理**：Poetry（推荐）或 pip
3. **API Key**：OpenAI API Key 或 DashScope API Key
4. **Ray**：分布式计算框架（用于仿真）
5. **Qdrant**：向量数据库（用于知识库检索，可选）

### 安装步骤

#### 方式 1：使用 Poetry（推荐）

```bash
# 1. 安装 Poetry（如果未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 2. 进入项目根目录
cd /root/project/agentsociety-ecosim

# 3. 安装依赖
poetry install

# 4. 激活虚拟环境
poetry shell
```

#### 方式 2：使用 pip

```bash
# 1. 进入项目根目录
cd /root/project/agentsociety-ecosim

# 2. 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt  # 如果有 requirements.txt
# 或手动安装主要依赖：
pip install agentscope pyyaml ray qdrant-client
```

### 配置环境变量

创建 `.env` 文件（在 `economist/` 目录下）：

```bash
# 方式 1：使用 OpenAI API
export OPENAI_API_KEY="your-openai-api-key"

# 方式 2：使用 DashScope API（阿里云）
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# 如果使用 DeepSeek（在代码中已配置）
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export BASE_URL="http://your-api-endpoint/v1/"
```

或者直接在命令行设置：

```bash
export OPENAI_API_KEY="your-api-key"
```

### 启动 Design Agent

**推荐方式：使用 Streamlit Web 界面**

```bash
# 进入 economist 目录
cd economist

# 启动 Streamlit Web 界面（推荐）
chmod +x run_streamlit.sh
./run_streamlit.sh
```

启动后，在浏览器中访问 `http://localhost:8501` 即可使用 Design Agent。

**命令行方式（可选）**：

如果需要使用命令行界面：

```bash
# 方式 1：使用启动脚本
chmod +x run_design_agent.sh
./run_design_agent.sh

# 方式 2：直接运行 Python
python design_agent.py
```

### 首次使用

启动后，Agent 会：
1. 显示欢迎信息
2. 等待你的研究问题
3. 自动检索相关学术文献
4. 提出假设并设计实验

**示例对话**：
```
用户: 鼓励研发的政策能否提高 GDP？

Agent: 
### Evidence from academic literature
[显示相关论文...]

Proposed hypothesis:
鼓励研发的政策会增加创新活动，从而提高生产力和 GDP。

Does this hypothesis sound reasonable? Should I proceed with designing the experiment?
```

## 📁 目录结构

### 新的两级目录结构

```
economist/experiments/
└── 20251219_120530/                          # 顶层：会话时间戳（YYYYMMDD_HHMMSS）
    ├── chat_history.json                     # 整个会话的对话历史
    │
    ├── rd_policy_gdp_study/                  # 实验1子目录（基于研究意图自动生成）
    │   ├── manifest.yaml                     # 实验清单
    │   ├── control_rdshare_010.yaml          # 控制组配置
    │   ├── control_rdshare_010.log           # 控制组运行日志
    │   ├── treatment_rdshare_020.yaml        # 实验组配置
    │   └── treatment_rdshare_020.log         # 实验组运行日志
    │
    └── tax_policy_employment/                  # 实验2子目录（同一会话的另一个实验）
        ├── manifest.yaml
        ├── control_tax_low.yaml
        └── treatment_tax_high.yaml
```

### 目录说明

- **顶层目录（时间戳）**：
  - 格式：`YYYYMMDD_HHMMSS`（如 `20251219_120530`）
  - 自动创建，基于会话开始时间
  - 包含 `chat_history.json`（整个会话的对话记录）

- **子目录（实验目录）**：
  - 基于研究意图自动生成名称（小写+下划线）
  - 每个实验有独立的子目录
  - 包含：`manifest.yaml`、配置文件（`.yaml`）、日志文件（`.log`）

- **输出目录**：
  - 仿真结果保存在：`/root/project/agentsociety-ecosim/output/{experiment_name}/`
  - 包含：`simulation_report_*.json`、统计数据等

## 🛠️ 核心功能

### 1. 实验管理

**Manifest 系统**：每个实验都有一个 `manifest.yaml` 文件，自动跟踪：
  - 研究问题和假设
  - 实验配置和参数变更
- 运行状态和时间记录（北京时间 UTC+8）
  - 关键指标和比较结果
  - 实验结论和洞察

### 2. 配置管理

- 从模板文件创建新配置
- 修改现有参数（点号路径，如 `innovation.policy_encourage_innovation`）
- 自动生成描述性文件名
- 支持自定义文件名

### 3. 仿真执行

- 运行经济仿真实验
- 实时日志记录（每个运行独立的 `.log` 文件）
- 自动更新 manifest 状态
- 超时保护（默认 1 小时）

### 4. 结果分析

- 从仿真报告中提取关键经济指标
- 自动记录到 manifest
- 支持多维度指标比较

### 5. 实验比较

- 对比控制组和实验组
- 计算差异和百分比变化
- 生成比较报告和结论

## 📚 工具函数详解

### `init_experiment_manifest`

创建或更新实验清单文件。

**参数**：
- `experiment_dir` (可选): 实验子目录名称，如果不提供则自动生成
- `experiment_name`: 实验名称（Title Case，如 "R&D Policy and GDP Study"）
- `research_question`: 研究问题
- `hypothesis`: 假设
- `description`: 实验描述
- `tags`: 标签列表（如 `["innovation", "GDP", "policy"]`）

**行为**：
- 自动创建或查找顶层时间戳目录
- 基于研究意图自动生成子目录名称
- 如果 manifest 已存在，会更新而不是覆盖

### `create_yaml_from_template`

从模板创建新的 YAML 配置文件。

**参数**：
- `source_file`: 源模板（默认 `"default.yaml"`）
- `parameter_changes`: 参数字典，格式：`{"category.param_name": value}`
- `manifest_path`: manifest 文件路径（用于自动更新）
- `config_label`: 配置标签（如 `"control"` 或 `"treatment"`）
- `custom_filename`: 自定义文件名（可选）

**示例**：
```python
create_yaml_from_template(
    source_file="default.yaml",
    parameter_changes={
        "innovation.innovation_research_share": 0.1,
        "system_scale.num_iterations": 12
    },
    manifest_path="experiments/20251219_120530/rd_policy_gdp_study/manifest.yaml",
    config_label="control"
)
```

### `run_simulation`

运行经济仿真实验。

**参数**：
- `config_file`: YAML 配置文件路径
- `manifest_path`: manifest 文件路径（用于更新状态）
- `run_label`: 运行标签（如 `"control"` 或 `"treatment"`）
- `timeout`: 超时时间（秒，默认 3600）

**功能**：
- 自动更新 manifest 运行状态
- 生成日志文件：`{yaml_basename}.log`
- 记录运行时间（北京时间）

### `read_simulation_report`

读取并提取仿真报告中的关键指标。

**参数**：
- `experiment_name`: 实验名称（用于查找报告）
- 或 `report_file`: 直接指定报告文件路径
- `manifest_path`: manifest 文件路径（用于更新指标）
- `run_label`: 运行标签

**提取的指标**：
- 就业统计（劳动力利用率、失业率）
- 收入和支出（平均月收入、平均月支出、储蓄率）
- 财富分布（平均财富、中位数财富、基尼系数）
- 经济趋势（方向、变化率）

### `compare_experiments`

比较两个实验的关键指标。

**参数**：
- `experiment1_name`: 第一个实验名称（控制组）
- `experiment2_name`: 第二个实验名称（实验组）
- `manifest_path`: manifest 文件路径（用于保存比较结果）

**返回**：
- 数值差异
- 百分比变化
- 并排对比

## 🔄 典型工作流程

### 完整研究流程

```
1. 用户提出研究问题
   ↓
2. Agent 检索学术文献（query_knowledge_base）
   ↓
3. Agent 提出假设，等待确认
   ↓
4. Agent 设计实验（get_available_parameters + 文献指导）
   ↓
5. Agent 创建 manifest（init_experiment_manifest）
   ↓
6. Agent 创建配置文件（create_yaml_from_template）
   - control: 基线配置
   - treatment: 修改目标参数
   ↓
7. Agent 运行仿真（run_simulation）
   - 先运行 control
   - 再运行 treatment
   ↓
8. Agent 分析结果（read_simulation_report）
   ↓
9. Agent 比较实验（compare_experiments）
   ↓
10. Agent 得出结论并更新 manifest
```

### 示例：研究 R&D 政策对 GDP 的影响

```python
# 1. Agent 自动创建 manifest（基于研究意图）
# 目录结构：
# experiments/20251219_120530/
#   ├── chat_history.json
#   └── rd_policy_gdp_study/
#       └── manifest.yaml

# 2. Agent 创建控制组配置
create_yaml_from_template(
    source_file="default.yaml",
    parameter_changes={
        "innovation.innovation_research_share": 0.1,
        "innovation.policy_encourage_innovation": True
    },
    manifest_path="experiments/20251219_120530/rd_policy_gdp_study/manifest.yaml",
    config_label="control"
)

# 3. Agent 创建实验组配置
create_yaml_from_template(
    source_file="default.yaml",
    parameter_changes={
        "innovation.innovation_research_share": 0.2,  # 提高研发投入
        "innovation.policy_encourage_innovation": True
    },
    manifest_path="experiments/20251219_120530/rd_policy_gdp_study/manifest.yaml",
    config_label="treatment"
)

# 4. Agent 运行仿真
run_simulation(
    config_file="experiments/20251219_120530/rd_policy_gdp_study/control_rdshare_010.yaml",
    manifest_path="experiments/20251219_120530/rd_policy_gdp_study/manifest.yaml",
    run_label="control"
)

run_simulation(
    config_file="experiments/20251219_120530/rd_policy_gdp_study/treatment_rdshare_020.yaml",
    manifest_path="experiments/20251219_120530/rd_policy_gdp_study/manifest.yaml",
    run_label="treatment"
)

# 5. Agent 分析并比较
read_simulation_report(experiment_name="exp_5h_5m_...", run_label="control", ...)
read_simulation_report(experiment_name="exp_5h_5m_...", run_label="treatment", ...)
compare_experiments(experiment1_name="...", experiment2_name="...", ...)
```

## 📋 Manifest 文件结构

```yaml
experiment_info:
  name: R&D Policy and GDP Study
  description: Two-arm controlled simulation...
  created_date: '2025-12-19'
  author: DesignAgent
  tags: [innovation, GDP, productivity, policy]
  directory: /root/project/.../experiments/20251219_120530/rd_policy_gdp_study

metadata:
  research_question: Does encouraging R&D increase GDP?
  hypothesis: Increasing research share will raise innovation and GDP
  expected_outcome: Treatment shows higher GDP...
  status: completed  # planned/running/analysis_pending/completed/failed
  runtime:
    start_time: '2025-12-19T12:11:46+08:00'
    end_time: '2025-12-19T12:45:23+08:00'
    duration_seconds: 2017

experiment_intervention:
  intervention_type: policy_intensity_adjustment
  intervention_parameters: {}

configurations:
  control:
    path: .../control_rdshare_010.yaml
    parameters_changed:
      innovation.innovation_research_share: 0.1
  treatment:
    path: .../treatment_rdshare_020.yaml
    parameters_changed:
      innovation.innovation_research_share: 0.2

runs:
  control:
    config_path: .../control_rdshare_010.yaml
    status: completed
    start_time: '2025-12-19T12:11:46+08:00'
    end_time: '2025-12-19T12:30:15+08:00'
    duration_seconds: 1109
    report_file: .../simulation_report_*.json
    log_file: .../control_rdshare_010.log
    key_metrics:
      employment: {...}
      income_expenditure: {...}
      wealth: {...}
  treatment:
    # 同上

results_summary:
  comparison_status: completed
  conclusion: Treatment group showed 15% higher GDP...
  insights:
    - Higher R&D share leads to increased innovation frequency
    - GDP growth is gradual, requiring extended simulation periods
  comparison: {...}
```

## 🔧 参数系统

### 参数查询工具

在设计实验前，先查询可用参数：

```python
# 获取所有参数
get_available_parameters(category="all")

# 获取特定类别参数
get_available_parameters(category="tax_policy")
get_available_parameters(category="innovation")

# 获取单个参数详情
get_parameter_info(parameter_name="innovation.innovation_research_share")
```

### 常用参数类别

- **tax_policy**: 税收政策（所得税率、增值税率、企业所得税率）
- **production**: 生产参数（劳动力生产率、生产能力）
- **labor_market**: 劳动力市场（解雇率、工资调整）
- **market**: 市场参数（价格调整、竞争强度）
- **system_scale**: 系统规模（家庭数、迭代次数、随机种子）
- **innovation**: 创新模块（研发投入比例、创新强度、政策支持）

### 参数路径格式

使用点号路径访问嵌套参数：
- `innovation.innovation_research_share` - 研发投入比例
- `tax_policy.income_tax_rate` - 个人所得税率
- `system_scale.num_households` - 家庭数量
- `system_scale.num_iterations` - 迭代次数（月数）

支持列表索引：
- `some.list[0].field` - 访问列表第一个元素的字段

## 📊 知识库检索

Design Agent 集成了学术论文知识库，支持：

- **语义检索**：基于 SPECTER2 模型进行语义搜索
- **期刊过滤**：按期刊筛选（Nature, Science, 等）
- **年份范围**：指定发表年份范围
- **相似论文查找**：基于标题和摘要查找相似论文

**使用示例**：
```python
query_knowledge_base(
    query="R&D tax credits innovation outcomes GDP growth",
    top_k=10,
    year_start=2020,
    year_end=2024
)
```

## ⚙️ 配置说明

### 默认配置文件

默认使用 `/root/project/agentsociety-ecosim/default.yaml` 作为模板。

### 重要参数

**系统规模**：
- `system_scale.num_households`: 家庭数量（影响仿真规模）
- `system_scale.num_iterations`: 迭代次数（月数）
- `system_scale.random_state`: 随机种子（确保可复现）

**创新模块**：
- `innovation.enable_innovation_module`: 是否启用创新系统
- `innovation.policy_encourage_innovation`: 政策是否鼓励创新
- `innovation.innovation_research_share`: 研发投入比例（0.0-1.0）

**税收政策**：
- `tax_policy.income_tax_rate`: 个人所得税率（默认 0.225）
- `tax_policy.vat_rate`: 增值税率（默认 0.08）
- `tax_policy.corporate_tax_rate`: 企业所得税率（默认 0.21）

## 🐛 故障排除

### 问题 1：API Key 未设置

**错误信息**：
```
Missing OPENAI_API_KEY (or DASHSCOPE_API_KEY) environment variable.
```

**解决方法**：
```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 或创建 .env 文件
echo "OPENAI_API_KEY=your-api-key" > economist/.env
```

### 问题 2：参数修改失败

**错误信息**：
```
Parameter 'xxx' does not exist in source file.
```

**解决方法**：
1. 使用 `get_available_parameters()` 查询可用参数
2. 检查参数路径是否正确（使用点号格式）
3. 确认参数在 `default.yaml` 中存在

### 问题 3：仿真运行失败

**可能原因**：
- 配置文件路径错误
- 仿真脚本权限问题
- Ray 未正确初始化

**解决方法**：
```bash
# 检查脚本权限
chmod +x economist/run_simulation.sh
chmod +x run_sim_with_yaml.sh

# 检查 Ray 状态
python -c "import ray; ray.init()"
```

### 问题 4：知识库检索失败

**错误信息**：
```
Error: cannot access local variable 'logger'...
```

**状态**：已修复（logger 作用域问题）

**如果仍有问题**：
- 检查 Qdrant 服务是否运行：`curl http://localhost:6333/collections`
- 确认知识库数据已索引

### 问题 5：目录结构混乱

**症状**：实验文件分散在不同目录

**解决方法**：
- 新版本已实现两级目录结构
- 旧实验可以手动迁移到新结构
- 新实验会自动使用新结构

## 📝 最佳实践

### 1. 实验设计

- **明确研究问题**：在开始前明确要研究的问题
- **查阅文献**：利用知识库检索功能了解相关研究
- **控制变量**：只修改目标参数，保持其他参数不变
- **设置随机种子**：确保可复现性

### 2. 参数选择

- **使用参数查询工具**：设计前先查询可用参数
- **参考默认值**：从 `default.yaml` 了解参数范围
- **渐进式调整**：不要一次性大幅修改参数

### 3. 实验执行

- **小规模测试**：先用少量家庭和迭代次数测试
- **监控日志**：关注 `.log` 文件中的错误信息
- **检查 manifest**：确认运行状态正确更新

### 4. 结果分析

- **多维度比较**：不仅看 GDP，还要看就业、收入分布等
- **统计显著性**：关注百分比变化，而不仅仅是绝对数值
- **时间趋势**：分析指标随时间的变化趋势

## 🔗 相关文档

- **目录结构说明**：`/root/project/agentsociety-ecosim/DIRECTORY_STRUCTURE_EXAMPLE.md`
- **经济系统核心**：`/root/project/agentsociety-ecosim/agentsociety_ecosim/`
- **数据库系统**：`/root/project/agentsociety-ecosim/database/`
- **MCP 服务器**：`/root/project/agentsociety-ecosim/agentsociety_ecosim/mcp_server/`

## 📜 开发历史

- **v1.0**：基本的 YAML 修改功能
- **v2.0**：添加实验管理、仿真执行、结果分析功能
- **v3.0**：引入 Manifest 系统，统一时间记录（北京时间）
- **v4.0**：两级目录结构，支持多实验会话，知识库集成

## 💡 提示

- **交互式工作流**：Agent 采用逐步确认的方式，不要一次性执行所有步骤
- **重用现有实验**：如果继续研究相同主题，Agent 会重用现有目录
- **自动命名**：子目录名称基于研究意图自动生成，也可以手动指定
- **日志查看**：每个运行的日志文件与配置文件在同一目录，便于排查问题

---

**需要帮助？** 查看代码注释或联系开发团队。
