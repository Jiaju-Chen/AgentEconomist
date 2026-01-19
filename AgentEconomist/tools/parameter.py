"""
参数查询工具

提供仿真参数的查询功能。
"""

import sys
from typing import Optional

from ..utils.path import get_project_root
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error, extract_response_text


def get_available_parameters(category: str = "all") -> str:
    """
    获取所有可配置的经济仿真参数列表
    
    Args:
        category: 参数类别过滤 (all/tax_policy/production/labor_market/market/system_scale/redistribution/performance/monitoring/innovation)
    
    Returns:
        XML格式字符串，包含参数分类和名称
    """
    try:
        # Import from economist module
        project_root = get_project_root()
        sys.path.insert(0, str(project_root / "economist"))
        
        from design_agent import get_available_parameters as _get_params
        
        response = _get_params(category=category)
        response_text = extract_response_text(response)
        
        return response_text
        
    except Exception as e:
        return handle_tool_error("get_available_parameters", e)


def get_parameter_info(parameter_name: str) -> str:
    """
    获取特定参数的详细信息
    
    Args:
        parameter_name: 参数名称（扁平化格式，如 "system_scale.num_households"）
    
    Returns:
        XML格式字符串，包含参数的详细信息
    """
    try:
        # TODO: Implement parameter info extraction from default.yaml or docs
        return format_tool_output(
            "success",
            f"Parameter info for {parameter_name}",
            parameter=parameter_name,
            details=f"Details for {parameter_name} (to be implemented)"
        )
        
    except Exception as e:
        return handle_tool_error("get_parameter_info", e)
