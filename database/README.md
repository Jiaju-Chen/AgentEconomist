
基于 **SPECTER2 + Qdrant** 的学术论文语义检索系统，专为 Agent Economist 设计。

## 🎯 特性

- **SPECTER2 Embedding**: 使用 Allen AI 专为学术论文设计的 Embedding 模型
- **Qdrant 向量存储**: 高性能向量数据库，支持本地和远程部署
- **分层索引**: 支持论文摘要和章节级别的检索
- **元数据过滤**: 按期刊、年份、文档类型筛选
- **Agent 集成**: 封装为 AgentScope 和 MCP 工具
- **增量索引**: 支持新论文的增量添加

## 📁 项目结构

```
database/
├── knowledge_base/             # 核心模块
│   ├── __init__.py            # 模块导出
│   ├── config.py              # 配置管理（包含 chunking.only_intro_sections 等开关）
│   ├── embeddings.py          # SPECTER2 Embedding 封装
│   ├── document_loader.py     # 论文 JSON 加载器（只产出摘要 + 引言等块）
│   ├── vector_store.py        # Qdrant 向量存储
│   ├── retriever.py           # 语义检索器
│   ├── indexer.py             # 索引构建器
│   └── tool.py                # Agent 工具封装
├── scripts/                    # 脚本
│   ├── build_index.py         # 构建索引
│   └── test_retrieval.py      # 测试检索
├── Crawl_Results/             # 论文数据目录（默认 data_dir）
│   ├── Articles/              # 自然科学/综合类期刊（Nature 系列等）
│   └── Articles-Social/       # 社会科学类期刊（AER, AJS 等）
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /root/project/agentsociety-ecosim/database
pip install -r requirements.txt
```

### 2. 启动 Qdrant 服务（推荐使用 Docker）

```bash
# 使用 Docker 启动 Qdrant（推荐，避免本地文件锁冲突）
docker run -d --name qdrant --network host qdrant/qdrant:latest

# 验证服务
curl http://localhost:6333/health
```

> **注意**: 使用 Docker 部署的 Qdrant 可以避免本地文件锁冲突，支持并发访问。  
> 如果使用本地模式，请确保没有其他进程同时访问 `qdrant_data` 目录。

### 3. 构建索引

```bash
# 完整索引（使用远程 Qdrant 服务器）
python scripts/build_index.py

# 增量索引（仅索引新论文，跳过已存在的）
python scripts/build_index.py --incremental

# 测试模式（仅索引 100 篇）
python scripts/build_index.py --limit 100

# 仅索引特定期刊
python scripts/build_index.py --journals "Nature Human Behaviour"

# 仅索引近 5 年论文
python scripts/build_index.py --year-start 2020 --year-end 2024
```

> 默认配置下，`build_index.py` 使用远程 Qdrant 服务器（`localhost:6333`），  
> `KnowledgeBaseConfig.data_dir` 指向整个 `Crawl_Results` 目录，  
> 会同时索引 `Articles/` 和 `Articles1219/` 下的所有 `article.json`。

### 3. 测试检索

```bash
# 交互式测试
python scripts/test_retrieval.py

# 单次查询
python scripts/test_retrieval.py --query "urban air pollution health impact"
```

## 💡 使用方法

### Python API

```python
from knowledge_base import KnowledgeBaseConfig, PaperRetriever

# 初始化
config = KnowledgeBaseConfig()
retriever = PaperRetriever(config)

# 语义检索
results = retriever.search(
    query="climate change economic policy",
    top_k=5,
    year_range=(2020, 2024),
)

# 打印结果
print(results.format_markdown())

# 查找相似论文
similar = retriever.search_similar_papers(
    paper_title="The impact of carbon tax on economic growth",
    paper_abstract="This study investigates...",
)
```

### Agent 工具

```python
from knowledge_base.tool import query_knowledge_base

# 查询知识库
result = query_knowledge_base(
    query="urban air pollution health impact",
    top_k=5,
    journals="Nature",
    year_start=2020,
)

print(result)
```

### AgentScope 集成

```python
from agentscope.tool import Toolkit
from knowledge_base.tool import create_agentscope_tool

# 创建工具并注册
toolkit = Toolkit()
toolkit.register_tool_function(create_agentscope_tool())
```

### MCP 集成

```python
from fastmcp import FastMCP
from knowledge_base.tool import register_mcp_tools

mcp = FastMCP("knowledge-base")
register_mcp_tools(mcp)
```

## ⚙️ 配置

### 环境变量（推荐）

```bash
# 数据目录
export KB_DATA_DIR=/path/to/articles

# Qdrant 配置（推荐使用远程模式）
export KB_QDRANT_MODE=remote  # local | remote | memory
export KB_QDRANT_HOST=localhost
export KB_QDRANT_PORT=6333

# 本地模式（如果使用本地文件存储）
# export KB_QDRANT_MODE=local
# export KB_QDRANT_PATH=./qdrant_data

# Embedding 设备
export KB_EMBEDDING_DEVICE=auto  # auto | cpu | cuda
```

### 配置文件

```python
from knowledge_base import KnowledgeBaseConfig

config = KnowledgeBaseConfig(
    data_dir="/path/to/articles",
)

# Qdrant 配置（推荐使用远程模式）
config.qdrant.mode = "remote"  # 使用 Docker 部署的 Qdrant 服务器
config.qdrant.host = "localhost"
config.qdrant.port = 6333

# 本地模式（如果使用本地文件存储）
# config.qdrant.mode = "local"
# config.qdrant.local_path = "./qdrant_data"

# Embedding 配置
config.embedding.device = "cuda"
config.embedding.batch_size = 64

# 分块配置（推荐：只索引 摘要 + 引言）
config.chunking.strategy = "section"        # section | paragraph | fixed
config.chunking.index_abstract = True       # 开启摘要索引
config.chunking.index_sections = True       # 开启章节索引
config.chunking.only_intro_sections = True  # 只保留标题为 Introduction 的引言章节
```

### 使用层面配置

在使用知识库的代码中（如 `economist/design_agent.py`），`query_knowledge_base` 等函数会自动使用环境变量配置。  
如果未设置环境变量，默认使用**本地模式**。建议设置环境变量使用远程模式：

**方式一：在 `.env` 文件中设置（推荐）**

在 `economist/.env` 文件中添加：
```bash
KB_QDRANT_MODE=remote
KB_QDRANT_HOST=localhost
KB_QDRANT_PORT=6333
```

**方式二：在启动脚本中设置**

在 `economist/run_design_agent.sh` 中添加：
```bash
export KB_QDRANT_MODE=remote
export KB_QDRANT_HOST=localhost
export KB_QDRANT_PORT=6333
```

**方式三：在代码中显式配置**

```python
from knowledge_base import KnowledgeBaseConfig, PaperRetriever

config = KnowledgeBaseConfig()
config.qdrant.mode = "remote"
config.qdrant.host = "localhost"
config.qdrant.port = 6333

retriever = PaperRetriever(config)
```

> **注意**: 使用层面**不需要修改代码**，只需要设置环境变量即可。  
> `query_knowledge_base` 等函数会自动从环境变量读取配置。

## 📊 支持的数据源

### 自然科学/综合类期刊（`Articles/`）
- Nature
- Nature Communications
- Nature Human Behaviour
- Nature Climate Change
- Nature Sustainability
- Nature Machine Intelligence
- Nature Computational Science
- Nature Cities
- Nature Ecology & Evolution
- Scientific Reports
- Scientific Data
- ... 等 15+ 个期刊

### 社会科学期刊（`Articles1219/`）
- American Economic Review (AER)
- American Journal of Sociology (AJS)
- Academy of Management Journal (AMJ)
- Academy of Management Review (AMR)
- American Political Science Review (APSR)
- American Sociological Review (ASR)
- Econometrica
- Journal of Political Economy (JPE)
- Quarterly Journal of Economics (QJE)
- Review of Economic Studies (Restud)

## 🔧 技术栈

| 组件 | 技术 |
|------|------|
| Embedding | SPECTER2 (allenai/specter2) |
| 向量数据库 | Qdrant |
| 深度学习 | PyTorch + Transformers |
| 回退模型 | sentence-transformers (MiniLM) |

## 📝 API 参考

### `query_knowledge_base`

```python
def query_knowledge_base(
    query: str,              # 查询文本
    top_k: int = 5,          # 返回数量
    journals: str = None,    # 期刊过滤
    year_start: int = None,  # 起始年份
    year_end: int = None,    # 结束年份
    doc_type: str = None,    # 文档类型
) -> Dict[str, Any]:
    ...
```

### `find_similar_papers`

```python
def find_similar_papers(
    title: str,              # 论文标题
    abstract: str = "",      # 论文摘要
    top_k: int = 5,          # 返回数量
) -> Dict[str, Any]:
    ...
```

### `get_paper_details`

```python
def get_paper_details(
    paper_id: str,           # 论文 ID
) -> Dict[str, Any]:
    ...
```

## ⚠️ 注意事项

1. **Qdrant 服务**: 推荐使用 Docker 部署的 Qdrant 服务器（`localhost:6333`），避免本地文件锁冲突
2. **首次加载模型**: SPECTER2 模型约 400MB，首次运行需要下载
3. **GPU 内存**: 使用 GPU 时建议至少 4GB 显存
4. **索引时间**: 完整索引数万篇论文可能需要数小时（当前约 13,000 篇论文，索引时间约 1-2 分钟）
5. **磁盘空间**: Qdrant 索引约占用 2-5GB 磁盘空间（当前约 24,000 个文档点）
6. **增量索引**: 使用 `--incremental` 参数可以跳过已索引的论文，大幅提升更新速度

## 📄 License

MIT License


