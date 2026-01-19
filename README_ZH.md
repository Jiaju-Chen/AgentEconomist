# AgentSociety Economic Simulation

一个基于 AgentScope 框架的智能经济系统仿真平台，支持通过受控实验进行经济政策研究。

## 🎯 项目概述

AgentSociety Economic Simulation 是一个集成了 AI Agent、经济仿真和知识库检索的综合研究平台。它能够：

- 🤖 **智能实验设计**：基于 Design Agent 自动设计受控实验
- 📚 **学术文献检索**：使用 SPECTER2 + Qdrant 检索相关学术论文
- 🏭 **经济系统仿真**：运行大规模多智能体经济仿真
- 📊 **结果分析**：自动提取关键指标并生成对比报告
- 🔬 **科学研究**：支持政策影响、创新效应等经济研究

## 📁 项目结构

```
agentsociety-ecosim/
├── economist/                    # Design Agent 模块
│   ├── design_agent.py          # 核心 Agent 实现
│   ├── tool_manager.py          # 工具管理器
│   ├── experiment_analyzer.py   # 实验分析工具
│   ├── analyze_firm_detail.py  # 企业详情分析
│   ├── streamlit_app.py         # Streamlit Web 应用
│   ├── streamlit_agent_wrapper.py  # Streamlit Agent 包装器
│   ├── default.yaml             # 默认配置模板
│   ├── run_streamlit.sh         # Streamlit 启动脚本（推荐）
│   ├── run_design_agent.sh      # CLI 启动脚本（可选）
│   ├── run_simulation.sh        # 仿真运行脚本
│   ├── requirements_ui.txt      # UI 依赖文件
│   ├── .env.example             # 环境变量模板
│   ├── experiments/             # 实验目录（自动生成）
│   └── README.md                # Design Agent 详细文档
│
├── agentsociety_ecosim/          # 经济仿真核心模块
│   ├── agent/                   # 智能体（家庭、企业、银行、政府）
│   ├── center/                  # 经济中心（市场、就业、资产）
│   ├── simulation/              # 仿真引擎
│   ├── consumer_modeling/       # 消费者行为建模
│   ├── mcp_server/              # MCP 参数管理服务器
│   └── SETUP.md                 # 环境配置指南
│
├── database/                    # 知识库模块
│   ├── knowledge_base/          # SPECTER2 + Qdrant 检索系统
│   │   ├── config.py            # 配置管理
│   │   ├── embeddings.py        # SPECTER2 Embedding 封装
│   │   ├── document_loader.py  # 论文 JSON 加载器
│   │   ├── vector_store.py      # Qdrant 向量存储
│   │   ├── retriever.py         # 语义检索器
│   │   ├── indexer.py           # 索引构建器
│   │   └── tool.py             # Agent 工具封装
│   ├── scripts/                 # 索引构建脚本
│   │   └── build_index.py      # 构建索引脚本
│   ├── requirements.txt         # 数据库模块依赖
│   ├── DEEP_SEARCH_PROMPT.md   # 深度搜索提示文档
│   └── README.md               # 知识库使用文档
│
├── model/                       # 预训练模型（需单独下载）
│   └── all-MiniLM-L6-v2/        # Embedding 模型
│
├── default.yaml                 # 默认仿真配置
├── pyproject.toml              # Poetry 项目配置
└── README.md                    # 本文件
```

## 🚀 快速开始

### 前置要求

- **Python**: 3.9 或更高版本
- **Conda**: 用于环境管理（推荐）
- **Docker**: 用于运行 Qdrant 向量数据库（可选，但推荐）
- **Ray**: 分布式计算框架（用于大规模仿真）

### 1. 克隆项目

```bash
git clone <repository-url>
cd agentsociety-ecosim
```

### 2. 环境配置

#### 方式 1：使用 Conda（推荐）

```bash
# 创建并激活 conda 环境
conda create -n ecosim python=3.10
conda activate ecosim

# 安装依赖（见下方）
```

#### 方式 2：使用 Poetry

```bash
# 安装 Poetry（如果未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 安装项目依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 3. 安装依赖

项目使用 Poetry 管理依赖。如果使用 Conda，可以手动安装：

```bash
# 核心依赖
pip install agentscope pyyaml ray qdrant-client transformers torch

# 数据库模块依赖
cd database
pip install -r requirements.txt

# MCP 服务器依赖
cd ../agentsociety_ecosim/mcp_server
pip install -r requirements.txt

# Streamlit UI（可选）
cd ../../economist
pip install -r requirements_ui.txt
```

### 4. 配置环境变量

创建 `economist/.env` 文件（参考 `economist/.env.example`）：

```bash
# API Keys（至少需要设置一个）
OPENAI_API_KEY=your-openai-api-key-here
# 或
DASHSCOPE_API_KEY=your-dashscope-api-key-here
# 或
DEEPSEEK_API_KEY=your-deepseek-api-key-here
BASE_URL=http://your-api-endpoint/v1/

# Qdrant 配置（知识库）
KB_QDRANT_MODE=remote
KB_QDRANT_HOST=localhost
KB_QDRANT_PORT=6333

# 模型路径（可选，默认使用相对路径）
MODEL_PATH=model/all-MiniLM-L6-v2
```

### 5. 启动 Qdrant（知识库）

```bash
# 使用 Docker 启动 Qdrant（推荐）
docker run -d --name qdrant --network host qdrant/qdrant:latest

# 验证服务
curl http://localhost:6333/health
```

### 6. 构建知识库索引（可选）

如果需要使用知识库检索功能：

```bash
cd database
python scripts/build_index.py --incremental
```

### 7. 启动 Design Agent（Streamlit Web 界面）

**推荐方式：使用 Streamlit Web 界面**

```bash
cd economist
chmod +x run_streamlit.sh
./run_streamlit.sh
```

启动后，在浏览器中访问 `http://localhost:8501` 即可使用 Design Agent。

**命令行方式（可选）**：

如果需要使用命令行界面：

```bash
cd economist
chmod +x run_design_agent.sh
./run_design_agent.sh
```

## 📖 使用指南

### Design Agent 工作流程

1. **提出研究问题**：例如"鼓励研发的政策能否提高 GDP？"
2. **检索学术文献**：Agent 自动检索相关论文
3. **设计实验**：创建控制组和实验组配置
4. **运行仿真**：执行经济系统仿真
5. **分析结果**：自动提取关键指标并对比
6. **得出结论**：基于统计差异得出科学结论

详细使用说明请参考：[economist/README.md](economist/README.md)

### 直接运行仿真

```bash
# 使用默认配置
./run_sim_with_yaml.sh

# 使用自定义配置
./run_sim_with_yaml.sh path/to/config.yaml
```

### Streamlit Web 界面（推荐）

Streamlit Web 界面是使用 Design Agent 的推荐方式，提供了友好的交互界面：

```bash
cd economist
chmod +x run_streamlit.sh
./run_streamlit.sh
```

启动后访问 `http://localhost:8501` 即可开始使用。

## 🔧 配置说明

### 仿真参数配置

主要配置文件：`default.yaml`

主要参数类别：
- **税收政策**：个人所得税率、增值税率、企业所得税率
- **劳动力市场**：裁员率、失业率阈值、动态招聘
- **生产参数**：劳动生产率、劳动弹性、利润转化比例
- **创新模块**：创新政策、研发投入比例
- **系统规模**：家庭数量、企业数量、仿真月数

详细参数说明请参考配置文件中的注释。

### 环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `OPENAI_API_KEY` | OpenAI API Key | 是（三选一）|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | 是（三选一）|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 是（三选一）|
| `KB_QDRANT_MODE` | Qdrant 模式（remote/local） | 否 |
| `KB_QDRANT_HOST` | Qdrant 主机地址 | 否 |
| `KB_QDRANT_PORT` | Qdrant 端口 | 否 |
| `MODEL_PATH` | Embedding 模型路径 | 否 |

## 📚 主要模块

### 1. Design Agent (`economist/`)

基于 AgentScope 的智能实验设计代理，支持：
- 自动检索学术文献
- 设计受控实验
- 运行仿真并分析结果
- 生成实验报告

### 2. 经济仿真 (`agentsociety_ecosim/`)

多智能体经济系统仿真，包括：
- **Agent**：家庭、企业、银行、政府
- **Center**：商品市场、劳动力市场、资产市场
- **Simulation**：仿真引擎和结果分析

### 3. 知识库 (`database/`)

基于 SPECTER2 + Qdrant 的学术论文检索系统：
- 支持 13,000+ 篇学术论文
- 语义检索和元数据过滤
- 支持增量索引

## 🛠️ 开发指南

### 代码结构

- **Python 代码**：遵循 PEP 8 风格指南
- **配置文件**：使用 YAML 格式
- **文档**：Markdown 格式

### 运行测试

```bash
# 运行仿真测试
cd agentsociety_ecosim/simulation
python joint_debug_test.py

# 测试知识库检索
cd database
python scripts/test_retrieval.py
```

### 贡献代码

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 常见问题

### Q: Qdrant 连接失败怎么办？

A: 确保 Qdrant 服务正在运行：
```bash
docker ps | grep qdrant
curl http://localhost:6333/health
```

如果未运行，启动 Qdrant：
```bash
docker run -d --name qdrant --network host qdrant/qdrant:latest
```

### Q: 如何获取 Embedding 模型？

A: 模型文件较大，需要单独下载。可以：
1. 从 HuggingFace 下载 `all-MiniLM-L6-v2`
2. 放置到 `model/all-MiniLM-L6-v2/` 目录
3. 或设置 `MODEL_PATH` 环境变量指向模型路径

### Q: 如何配置 API Key？

A: 创建 `economist/.env` 文件，添加：
```bash
OPENAI_API_KEY=your-key-here
```

### Q: 实验数据保存在哪里？

A: 实验配置和结果保存在 `economist/experiments/` 目录下，按时间戳和实验意图组织。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 🙏 致谢

- [AgentScope](https://github.com/modelscope/agentscope) - Agent 框架
- [SPECTER2](https://github.com/allenai/specter2) - 学术论文 Embedding 模型
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [Ray](https://www.ray.io/) - 分布式计算框架

## 📮 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**：2025-01-XX
