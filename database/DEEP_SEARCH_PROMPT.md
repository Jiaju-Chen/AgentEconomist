
## 任务背景

我需要为一个**基于 LLM 的经济研究智能体 (Agent Economist)** 添加学术论文知识库功能。目前已有一个结构化的论文数据库，需要找到合适的向量数据库和 RAG 框架来实现语义检索。

---

## 一、现有数据库结构

### 1.1 数据规模
- **期刊数量**: 15+ 个 Nature 系列期刊
- **时间跨度**: 2000-2025 年
- **文章数量**: 数万篇学术论文
- **总数据量**: 估计 10-50GB JSON 文件

### 1.2 数据存储格式

``` 
/database/Crawl_Results/
├── Articles/               # 自然科学类期刊（Nature 系列等）
│   ├── Nature/
│   │   ├── 2024-article/
│   │   │   └── {article_id}/
│   │   │       └── article.json
│   │   └── ...
│   ├── Nature Human Behaviour/
│   ├── Nature Communications/
│   ├── Nature Climate Change/
│   ├── Scientific Reports/
│   └── ... (15+ 期刊)
├── Articles-Social/        # 社会科学类期刊（AER, AJS 等）
│   ├── aer/
│   │   ├── 2015_vol105_no02/
│   │   │   └── 10_1257_aer_20080841/
│   │   │       └── article.json
│   │   └── ...
│   ├── ajs/
│   │   └── 2015_vol121_no02/
│   │       └── 10_1086_225469/
│   │           └── article.json
│   └── ...（可持续扩展新的社会科学期刊）
└── crawl_info_filter.json  # 元数据索引
``` 

### 1.3 单篇文章 JSON 结构

```json
{
    "id": "s41562-024-01817-8",
    "journal": "Nature Human Behaviour",
    "type": "article",
    "title": "论文标题",
    "publish_time": "2024-01-15",
    "open_access": true,
    "Abstract": "论文摘要文本...",
    "Sections": [
        {
            "title": "Introduction",
            "text": "章节正文内容...",
            "cites": ["ref-CR1", "ref-CR2"],
            "figures": []
        },
        {
            "title": "Methods",
            "text": "方法描述..."
        }
        // ... 更多章节
    ],
    "Qwen3JudgeField": {
        "judgment": "Yes",
        "key_topics": ["urban economics", "policy", "sustainability"]
    },
    "Extract_CAMP": {
        "Context": "研究背景",
        "A (Independent Variable)": "自变量",
        "B (Dependent Variable)": "因变量",
        "Mechanism": "机制",
        "Pattern": "模式"
    }
}
```

### 1.4 数据特点

| 特点 | 描述 |
|------|------|
| **结构化** | JSON 格式，有清晰的字段定义 |
| **多层级** | 文章 → 章节 → 段落，需要支持层级检索；当前索引策略聚焦 **摘要 + 引言 (Introduction)** |
| **元数据丰富** | 期刊、时间、主题分类、摘要提取字段 |
| **长文本** | 单篇文章可能有数万字，需要合理分块 |
| **增量更新** | 持续有新论文加入 |

---

## 二、现有系统架构

### 2.1 Agent 框架

```python
# 基于 AgentScope 的 ReActAgent
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse

agent = ReActAgent(
    name="DesignAgent",
    sys_prompt="...",
    model=OpenAIChatModel(...),
    toolkit=toolkit,  # 工具注册
)
```

### 2.2 MCP (Model Context Protocol) 集成

系统已集成 MCP 服务器，支持：
- `mcp_ai-economist_*` 系列工具（参数管理、仿真控制）
- FastMCP 服务器架构
- 工具动态注册

### 2.3 工具注册模式

```python
def build_design_agent(api_key: str):
    toolkit = Toolkit()
    
    def register_tool(fn: Callable):
        toolkit.register_tool_function(fn)
    
    # 已有工具
    register_tool(get_available_parameters)
    register_tool(run_simulation)
    register_tool(analyze_experiment_directory)
    
    # 需要新增：知识库查询工具
    register_tool(query_knowledge_base)  # 待实现
```

---

## 三、需求描述

### 3.1 核心功能需求

1. **语义检索**: 根据自然语言查询，返回最相关的论文/章节（当前版本优先使用“摘要 + 引言” 级别的文档块）
2. **元数据过滤**: 支持按期刊、年份、主题筛选
3. **分块检索**: 支持文章级、章节级、段落级检索
4. **混合检索**: 结合关键词匹配和语义相似度

### 3.2 集成需求

1. **Python API**: 提供简单的 Python 接口供 Agent 调用
2. **工具封装**: 封装为 AgentScope 的 ToolResponse 格式
3. **MCP 兼容**: 可选地作为 MCP 工具注册
4. **流式支持**: 可选地支持流式返回结果

### 3.3 性能需求

| 指标 | 要求 |
|------|------|
| 首次索引时间 | < 1 小时（可接受离线处理） |
| 查询延迟 | < 2 秒 |
| 内存占用 | < 8GB（开发环境） |
| 增量更新 | 支持新论文动态添加 |

### 3.4 部署约束

- **运行环境**: Linux 服务器，Python 3.10+
- **本地优先**: 优先考虑本地部署方案（避免云依赖）
- **轻量级**: 避免过于复杂的架构（如 Kubernetes）
- **Embedding 模型**: 可使用本地模型（如 sentence-transformers）或 API

---

## 四、技术选型方向（请 Deep Search 评估）

### 4.1 向量数据库选项

请评估以下方案的**优缺点、适用场景、Python API 易用性**：

1. **Qdrant** - 本地 + 云，Rust 实现
2. **Milvus / Milvus Lite** - 云原生向量数据库
3. **ChromaDB** - 轻量级，纯 Python
4. **FAISS** - Facebook 开源，高性能
5. **LanceDB** - 嵌入式向量数据库，支持多模态
6. **Weaviate** - GraphQL 接口，支持混合搜索
7. **pgvector** - PostgreSQL 扩展

### 4.2 RAG 框架选项

请评估以下 RAG 框架的**功能完整性、与向量库集成度、文档质量**：

1. **LangChain** - 最流行，生态丰富
2. **LlamaIndex** - 专注文档索引，结构化数据支持好
3. **Haystack** - 端到端 NLP 管道
4. **自定义实现** - 直接使用 sentence-transformers + 向量库

### 4.3 Embedding 模型选项

请评估以下模型的**多语言支持、学术文本效果、资源占用**：

1. **all-MiniLM-L6-v2** - 轻量级，通用
2. **bge-large-zh-v1.5** - 中文优化
3. **e5-large-v2** - 微软，学术文本效果好
4. **text-embedding-3-small** - OpenAI API
5. **Cohere embed-v3** - 多语言支持好
6. **SPECTER2** - 专门针对学术论文

### 4.4 特殊需求

1. **JSON 结构保留**: 检索结果需要保留原始 JSON 字段（如 journal, publish_time）
2. **层级索引**: 支持文章 → 章节 → 段落的层级检索
3. **引用追踪**: 可选地支持引用关系图谱
4. **主题聚类**: 可选地支持论文主题聚类

---

## 五、期望输出

请提供：

1. **推荐方案**：最适合我场景的技术栈组合（向量库 + RAG框架 + Embedding模型）
2. **对比表格**：不同方案的详细对比
3. **实现示例**：推荐方案的核心代码示例
4. **注意事项**：该方案的局限性和需要注意的坑
5. **替代方案**：如果推荐方案不满足需求的备选

---

## 六、补充信息

### 6.1 现有代码参考

```python
# 工具返回格式（需要兼容）
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

def query_knowledge_base(query: str, top_k: int = 5) -> ToolResponse:
    """查询知识库工具"""
    # ... 实现检索逻辑
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"<status>success</status><results>{json.dumps(results)}</results>"
            )
        ]
    )
```

### 6.2 预期使用场景

```
用户: "关于城市空气污染对公共健康影响的研究有哪些？"

Agent 调用: query_knowledge_base(
    query="urban air pollution public health impact",
    filters={"journal": "Nature*", "year_range": [2020, 2025]},
    top_k=5
)

返回: 相关论文列表，包含标题、摘要、关键发现
```

### 6.3 技术栈约束

- Python 3.10+
- 已安装: torch, transformers, sentence-transformers
- 可安装: 任意 PyPI 包
- 服务器: 16GB RAM, 无 GPU（或可选 GPU）

---

**请基于以上信息，进行 Deep Search 并给出详细的技术选型建议。**


