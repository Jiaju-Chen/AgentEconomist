"""
知识库查询工具

提供学术论文知识库的查询功能，并自动保存查询结果到 manifest。
"""

import sys
import json
from pathlib import Path
from typing import Optional

from ..core.manifest import add_knowledge_base_items
from ..utils.path import get_project_root
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error


def query_knowledge_base(
    query: str,
    top_k: int = 20,
    journals: str = "",
    year_start: int = 0,
    year_end: int = 0,
    doc_type: str = "",
    manifest_path: Optional[str] = None,
) -> str:
    """
    查询学术论文知识库。
    
    CRITICAL: query 必须是英文！
    
    Args:
        query: 查询文本（必须是英文）
        top_k: 返回结果数量
        journals: 期刊过滤
        year_start: 起始年份
        year_end: 结束年份
        doc_type: 文档类型
        manifest_path: manifest.yaml 路径（可选，如果提供则自动保存查询结果）
    
    Returns:
        格式化字符串，包含查询结果
    """
    try:
        # 导入知识库工具
        project_root = get_project_root()
        database_path = project_root / "database"
        sys.path.insert(0, str(database_path))
        
        from knowledge_base.tool import query_knowledge_base as _query_kb
        
        result = _query_kb(
            query=query,
            top_k=top_k,
            journals=journals if journals else None,
            year_start=year_start if year_start > 0 else None,
            year_end=year_end if year_end > 0 else None,
            doc_type=None  # Always search all document types
        )
        
        if result.get("status") == "success":
            results = result.get("results", [])
            
            # 如果提供了 manifest_path，保存查询结果到 manifest
            knowledge_items = []
            if manifest_path and results:
                try:
                    # 转换结果格式：从查询结果提取 title, source, url
                    for item in results:
                        # 提取标题
                        title = item.get("title", "")
                        if not title:
                            continue
                        
                        # 提取来源（期刊 + 年份）
                        journal = item.get("journal", "")
                        publish_time = item.get("publish_time", "")
                        source_parts = []
                        if journal:
                            source_parts.append(journal)
                        if publish_time:
                            # 提取年份
                            year = str(publish_time).split("-")[0] if "-" in str(publish_time) else str(publish_time)[:4]
                            if year and year.isdigit():
                                source_parts.append(year)
                        source = ", ".join(source_parts) if source_parts else "Unknown"
                        
                        # 提取 URL
                        url = item.get("pdf_link") or item.get("url") or None
                        
                        knowledge_items.append({
                            "title": title,
                            "source": source,
                            "url": url
                        })
                    
                    # 保存到 manifest
                    if knowledge_items:
                        add_knowledge_base_items(manifest_path, knowledge_items, merge=True)
                except Exception as e:
                    # 保存失败不影响查询结果返回
                    print(f"Warning: Failed to save knowledge_base to manifest: {e}")
            
            # 格式化返回结果
            results_json = json.dumps(results, ensure_ascii=False, indent=2)
            
            response_parts = {
                "query": query,
                "total_found": str(result.get('total_found', 0)),
                "results": results_json
            }
            
            if manifest_path:
                response_parts["manifest_path"] = manifest_path
                if knowledge_items:
                    response_parts["knowledge_base_saved"] = f"{len(knowledge_items)} items saved"
            
            return format_tool_output(
                "success",
                f"Found {result.get('total_found', 0)} papers",
                **response_parts
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            response_parts = {"error_detail": error_msg}
            if manifest_path:
                response_parts["manifest_path"] = manifest_path
            return format_tool_output("error", error_msg, **response_parts)
            
    except Exception as e:
        return handle_tool_error("query_knowledge_base", e)
