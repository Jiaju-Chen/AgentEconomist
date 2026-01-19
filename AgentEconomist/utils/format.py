"""
格式化工具模块

提供工具输出的格式化功能。
"""

import re
from typing import Dict


def format_tool_output(
    status: str,
    message: str = "",
    **kwargs
) -> str:
    """
    格式化工具输出为 XML 标签格式
    
    Args:
        status: "success" | "error"
        message: 主要消息内容
        **kwargs: 其他键值对，会被转换为 XML 标签
    
    Returns:
        格式化的 XML 字符串
    
    Example:
        >>> format_tool_output("success", "Simulation started", manifest_path="/path/to/manifest.yaml")
        '<status>success</status><message>Simulation started</message><manifest_path>/path/to/manifest.yaml</manifest_path>'
    """
    parts = [f"<status>{status}</status>"]
    
    if message:
        parts.append(f"<message>{message}</message>")
    
    for key, value in kwargs.items():
        parts.append(f"<{key}>{value}</{key}>")
    
    return "".join(parts)


def parse_tool_output(output: str) -> Dict[str, str]:
    """
    解析工具输出的 XML 标签
    
    Args:
        output: 工具输出字符串
    
    Returns:
        解析后的字典
    """
    result = {}
    
    # 简单的正则解析
    for match in re.finditer(r"<(\w+)>(.*?)</\1>", output, re.DOTALL):
        tag_name = match.group(1)
        tag_content = match.group(2).strip()
        result[tag_name] = tag_content
    
    return result
