"""
Agent 工具封装

将知识库检索功能封装为可被 AgentScope Agent 调用的工具。
"""

import json
import logging
import threading
from typing import Optional, List, Any, Dict

from .config import KnowledgeBaseConfig
from .retriever import PaperRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# 全局检索器实例（单例模式）
_retriever: Optional[PaperRetriever] = None
_retriever_lock = threading.Lock()  # 线程锁，确保并发安全


def get_retriever(config: Optional[KnowledgeBaseConfig] = None) -> PaperRetriever:
    """
    获取检索器实例（单例，线程安全）
    
    Args:
        config: 知识库配置
    
    Returns:
        PaperRetriever 实例
    """
    global _retriever
    
    # 快速路径：如果已初始化，直接返回
    if _retriever is not None:
        return _retriever
    
    # 使用锁确保只有一个线程创建实例
    with _retriever_lock:
        # 双重检查：可能在等待锁时其他线程已完成初始化
        if _retriever is not None:
            return _retriever
        
        # 使用配置（支持环境变量覆盖），默认指向全局 qdrant_data
        config = config or KnowledgeBaseConfig()
        # 如果显式启用 fallback，则自动调整向量维度以匹配 MiniLM
        if config.embedding.use_fallback:
            config.qdrant.vector_size = 384
        
        logger.info("Creating PaperRetriever singleton instance (thread-safe)")
        _retriever = PaperRetriever(config)
        
    return _retriever


def _is_chinese(text: str) -> bool:
    """检测文本是否包含中文字符"""
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def _translate_to_english(text: str) -> str:
    """
    将中文查询翻译为英文（简单实现）
    注意：这是一个简单的占位实现，实际应该使用翻译 API
    """
    # 如果包含中文，返回提示信息
    if _is_chinese(text):
        # 这里应该调用翻译服务，但为了不增加依赖，先返回原文本并记录警告
        logger.warning(f"Query contains Chinese characters: {text}. SPECTER2 embedding model works best with English queries.")
        # 返回原文本，但建议 Agent 使用英文查询
        return text
    return text


def query_knowledge_base(
    query: str,
    top_k: int = 5,
    journals: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    doc_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    查询学术论文知识库
    
    在学术论文库中进行语义检索，返回与查询最相关的论文内容。
    支持按期刊、年份、文档类型进行过滤。
    
    **重要**: SPECTER2 模型对英文查询效果最佳，如果查询包含中文，建议先翻译为英文。
    
    Args:
        query: 自然语言查询，描述你想了解的研究主题或问题（**建议使用英文**）
        top_k: 返回结果数量，默认为 5，最大 20
        journals: 期刊过滤（可选），支持的期刊包括：
            - Nature, Nature Communications, Nature Human Behaviour
            - Nature Climate Change, Nature Sustainability
            - Scientific Reports, 等
        year_start: 起始年份（可选），如 2020
        year_end: 结束年份（可选），如 2024
        doc_type: 文档类型过滤（可选）：
            - "paper": 整篇论文级别文档（当前索引策略：摘要 + 引言 合并为一个块）
    
    Returns:
        检索结果字典，包含：
        - status: 状态 ("success" 或 "error")
        - query: 原始查询
        - total_found: 找到的结果数量
        - results: 结果列表，每项包含：
            - title: 论文标题
            - journal: 期刊名称
            - publish_time: 发表时间
            - score: 相似度分数
            - text: 匹配的文本内容
            - section_title: 章节标题（如果是章节）
            - key_topics: 关键主题列表
            - pdf_link: 原文链接
    
    Examples:
        # 简单查询
        query_knowledge_base("urban air pollution health impact")
        
        # 带过滤条件的查询
        query_knowledge_base(
            query="climate change economic policy",
            top_k=10,
            journals="Nature Climate Change",
            year_start=2020,
            year_end=2024
        )
        
        # 只搜索摘要
        query_knowledge_base(
            query="machine learning in economics",
            doc_type="abstract"
        )
    """
    try:
        retriever = get_retriever()
        
        # 参数验证
        top_k = min(max(1, top_k), 20)
        
        # 检查查询语言：如果包含中文，记录警告
        if _is_chinese(query):
            logger.warning(
                f"Query contains Chinese characters: '{query}'. "
                "SPECTER2 embedding model is optimized for English. "
                "For best results, please translate the query to English before calling this function."
            )
            # 返回错误提示，要求使用英文查询
            return {
                "status": "error",
                "error": (
                    "Query contains Chinese characters. SPECTER2 embedding model requires English queries. "
                    "Please translate your query to English before searching. "
                    f"Original query: {query}"
                ),
                "query": query,
                "total_found": 0,
                "results": [],
            }
        
        # 构建年份范围
        year_range = None
        if year_start or year_end:
            year_range = (
                year_start or 2000,
                year_end or 2025,
            )
        
        # 期刊列表
        journal_list = [journals] if journals else None
        
        # 执行检索
        result = retriever.search(
            query=query,
            top_k=top_k,
            journals=journal_list,
            year_range=year_range,
            doc_type=doc_type,
        )
        
        # 格式化结果
        formatted_results = []
        for r in result.results:
            doc = r.document
            formatted_results.append({
                "title": doc.title,
                "journal": doc.journal,
                "publish_time": doc.publish_time,
                "score": round(r.score, 3),
                "doc_type": doc.doc_type,
                "section_title": doc.section_title,
                "text": doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text,
                "key_topics": doc.key_topics[:5] if doc.key_topics else [],
                "pdf_link": doc.pdf_link,
                "paper_id": doc.paper_id,
            })
        
        return {
            "status": "success",
            "query": query,
            "total_found": result.total_found,
            "results": formatted_results,
        }
        
    except Exception as e:
        logger.error(f"Knowledge base query failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def get_paper_details(paper_id: str) -> Dict[str, Any]:
    """
    获取论文详细信息
    
    根据论文 ID 获取完整的论文内容和上下文信息，
    包括所有章节内容和相似论文推荐。
    
    Args:
        paper_id: 论文 ID（从检索结果中获取）
    
    Returns:
        论文详情字典，包含：
        - paper_id: 论文 ID
        - title: 标题
        - journal: 期刊
        - publish_time: 发表时间
        - abstract: 摘要
        - sections: 章节列表
        - key_topics: 关键主题
        - similar_papers: 相似论文列表
    
    Example:
        get_paper_details("s41562-024-01817-8")
    """
    try:
        retriever = get_retriever()
        context = retriever.get_paper_context(
            paper_id=paper_id,
            include_similar=True,
            similar_count=3,
        )
        
        if "error" in context:
            return {
                "status": "error",
                "error": context["error"],
            }
        
        return {
            "status": "success",
            **context,
        }
        
    except Exception as e:
        logger.error(f"Failed to get paper details: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def find_similar_papers(
    title: str,
    abstract: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    查找相似论文
    
    给定一篇论文的标题和摘要，查找知识库中最相似的论文。
    适用于文献综述、相关工作查找等场景。
    
    Args:
        title: 论文标题
        abstract: 论文摘要（可选，提供后检索更精确）
        top_k: 返回结果数量
    
    Returns:
        相似论文列表
    
    Example:
        find_similar_papers(
            title="The impact of urban air pollution on public health",
            abstract="This study investigates...",
            top_k=5
        )
    """
    try:
        retriever = get_retriever()
        result = retriever.search_similar_papers(
            paper_title=title,
            paper_abstract=abstract,
            top_k=top_k,
        )
        
        return {
            "status": "success",
            "query_title": title,
            "total_found": result.total_found,
            "similar_papers": [
                {
                    "title": r.document.title,
                    "journal": r.document.journal,
                    "publish_time": r.document.publish_time,
                    "similarity": round(r.score, 3),
                    "abstract": r.document.abstract[:300] + "..." 
                        if r.document.abstract and len(r.document.abstract) > 300 
                        else r.document.abstract,
                    "pdf_link": r.document.pdf_link,
                }
                for r in result.results
            ],
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar papers: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def get_knowledge_base_stats() -> Dict[str, Any]:
    """
    获取知识库统计信息
    
    返回知识库的基本统计数据，包括索引的文档数量、
    支持的期刊列表等。
    
    Returns:
        统计信息字典
    """
    try:
        retriever = get_retriever()
        stats = retriever.get_stats()
        
        return {
            "status": "success",
            "collection_name": stats.get("name"),
            "total_documents": stats.get("points_count", 0),
            "vector_size": stats.get("vector_size"),
            "status": stats.get("status"),
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# AgentScope 工具封装（兼容 design_agent.py 的格式）
# ============================================================================

def create_agentscope_tool():
    """
    创建 AgentScope 兼容的工具函数
    
    返回可以直接注册到 AgentScope Toolkit 的工具函数。
    
    Usage:
        from knowledge_base.tool import create_agentscope_tool
        from agentscope.tool import Toolkit
        
        toolkit = Toolkit()
        toolkit.register_tool_function(create_agentscope_tool())
    """
    try:
        from agentscope.tool import ToolResponse
        from agentscope.message import TextBlock
        
        def query_academic_knowledge(
            query: str,
            top_k: int = 5,
            journals: str = "",
            year_start: int = 0,
            year_end: int = 0,
        ) -> ToolResponse:
            """
            查询学术论文知识库，获取与研究问题相关的学术文献。
            
            这个工具可以帮助你：
            1. 了解某个研究领域的背景知识
            2. 查找支持你研究假设的文献证据
            3. 发现相关的研究方法和理论框架
            
            Args:
                query: 自然语言查询，描述你想了解的研究主题
                top_k: 返回结果数量（1-20）
                journals: 期刊过滤（可选）
                year_start: 起始年份（可选，0表示不过滤）
                year_end: 结束年份（可选，0表示不过滤）
            
            Returns:
                包含相关论文信息的响应
            """
            # 调用核心检索函数
            result = query_knowledge_base(
                query=query,
                top_k=top_k,
                journals=journals if journals else None,
                year_start=year_start if year_start > 0 else None,
                year_end=year_end if year_end > 0 else None,
            )
            
            # 格式化为 XML 风格的响应
            if result["status"] == "success":
                results_json = json.dumps(result["results"], ensure_ascii=False, indent=2)
                response_text = (
                    f"<status>success</status>"
                    f"<query>{query}</query>"
                    f"<total_found>{result['total_found']}</total_found>"
                    f"<results>{results_json}</results>"
                )
            else:
                response_text = (
                    f"<status>error</status>"
                    f"<error>{result.get('error', 'Unknown error')}</error>"
                )
            
            return ToolResponse(
                content=[TextBlock(type="text", text=response_text)]
            )
        
        return query_academic_knowledge
        
    except ImportError:
        logger.warning("AgentScope not installed, returning plain function")
        return query_knowledge_base


# ============================================================================
# MCP 工具封装（用于 MCP 服务器集成）
# ============================================================================

def register_mcp_tools(mcp_server):
    """
    注册 MCP 工具
    
    将知识库工具注册到 MCP 服务器。
    
    Args:
        mcp_server: FastMCP 服务器实例
    
    Usage:
        from fastmcp import FastMCP
        from knowledge_base.tool import register_mcp_tools
        
        mcp = FastMCP("knowledge-base")
        register_mcp_tools(mcp)
    """
    
    @mcp_server.tool()
    def search_papers(
        query: str,
        top_k: int = 5,
        journal: str = "",
        year_start: int = 0,
        year_end: int = 0,
    ) -> str:
        """
        搜索学术论文知识库
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            journal: 期刊过滤
            year_start: 起始年份
            year_end: 结束年份
        """
        result = query_knowledge_base(
            query=query,
            top_k=top_k,
            journals=journal if journal else None,
            year_start=year_start if year_start > 0 else None,
            year_end=year_end if year_end > 0 else None,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def get_paper_info(paper_id: str) -> str:
        """
        获取论文详细信息
        
        Args:
            paper_id: 论文 ID
        """
        result = get_paper_details(paper_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def similar_papers(
        title: str,
        abstract: str = "",
        top_k: int = 5,
    ) -> str:
        """
        查找相似论文
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            top_k: 返回数量
        """
        result = find_similar_papers(title, abstract, top_k)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def knowledge_base_info() -> str:
        """获取知识库统计信息"""
        result = get_knowledge_base_stats()
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    logger.info("Registered MCP tools for knowledge base")
