"""
响应处理模块

提供工具响应的提取和错误处理功能。
"""

import os
import re
from typing import Optional


def extract_manifest_path(text: str) -> Optional[str]:
    """
    从工具输出中提取 manifest_path
    
    支持的格式：
    - <manifest_path>path</manifest_path>
    - <manifest_updated>path</manifest_updated>
    
    Args:
        text: 工具输出文本
    
    Returns:
        提取的 manifest_path，如果未找到则返回 None
    """
    if not text:
        return None
    
    # Try <manifest_path> tag
    m = re.search(r"<manifest_path>(.*?)</manifest_path>", text, re.DOTALL)
    if m:
        return m.group(1).strip() or None
    
    # Try <manifest_updated> tag
    m = re.search(r"<manifest_updated>(.*?)</manifest_updated>", text, re.DOTALL)
    if m:
        return m.group(1).strip() or None
    
    return None


def handle_tool_error(func_name: str, error: Exception, context: str = "") -> str:
    """
    统一处理工具函数错误
    
    Args:
        func_name: 工具函数名称
        error: 异常对象
        context: 额外的上下文信息
    
    Returns:
        格式化的错误输出（XML格式）
    """
    from .format import format_tool_output
    
    error_msg = str(error)
    if context:
        error_msg = f"{context}: {error_msg}"
    
    # 记录错误日志（可选，用于调试）
    if os.getenv("DEBUG_TOOLS", "false").lower() == "true":
        import traceback
        print(f"[ERROR] {func_name}: {error_msg}")
        traceback.print_exc()
    
    return format_tool_output(
        "error",
        f"Failed in {func_name}: {error_msg}"
    )


def extract_response_text(response) -> str:
    """
    安全地提取 ToolResponse 的文本内容（用于兼容 economist_core）
    
    Args:
        response: ToolResponse 对象或其他类型
    
    Returns:
        提取的文本内容，如果提取失败则返回字符串表示
    """
    if not hasattr(response, 'content'):
        return str(response)
    
    if not response.content:
        return str(response)
    
    parts = []
    for block in response.content:
        text = getattr(block, 'text', None)
        if text:
            parts.append(str(text))
    
    return "\n".join(parts).strip() if parts else str(response)
