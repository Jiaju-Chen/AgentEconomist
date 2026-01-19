"""
干预工具 - MCP工具定义
"""
from typing import Dict, Any
from .simulation_wrapper import get_wrapper


async def pause_simulation_tool() -> str:
    """
    暂停正在运行的仿真
    
    使用场景：
    - 观察当前状态，决定下一步干预
    - 临时停止以分析中间结果
    - 准备注入干预措施
    
    Returns:
        JSON字符串，包含暂停结果
    """
    import json
    wrapper = get_wrapper()
    result = await wrapper.pause_simulation()
    return json.dumps(result, indent=2, ensure_ascii=False)


async def resume_simulation_tool() -> str:
    """
    恢复已暂停的仿真
    
    使用场景：
    - 完成分析后继续运行
    - 注入干预后恢复仿真
    
    Returns:
        JSON字符串，包含恢复结果
    """
    import json
    wrapper = get_wrapper()
    result = await wrapper.resume_simulation()
    return json.dumps(result, indent=2, ensure_ascii=False)


async def inject_intervention_tool(
    intervention_type: str,
    target_month: int,
    parameters: dict,
    description: str = ""
) -> str:
    """
    向仿真注入干预措施
    
    Args:
        intervention_type: 干预类型
        target_month: 目标月份
        parameters: 干预参数
        description: 描述
        
    Returns:
        JSON字符串
    """
    import json
    wrapper = get_wrapper()
    result = await wrapper.inject_intervention(
        intervention_type=intervention_type,
        target_month=target_month,
        parameters=parameters,
        description=description
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


async def list_pending_interventions_tool() -> str:
    """列出所有待执行的干预"""
    import json
    wrapper = get_wrapper()
    interventions = wrapper.list_pending_interventions()
    return json.dumps({
        "success": True,
        "count": len(interventions),
        "interventions": interventions
    }, indent=2, ensure_ascii=False)


async def cancel_intervention_tool(intervention_id: str) -> str:
    """取消指定的干预"""
    import json
    wrapper = get_wrapper()
    result = wrapper.cancel_intervention(intervention_id)
    return json.dumps(result, indent=2, ensure_ascii=False)

