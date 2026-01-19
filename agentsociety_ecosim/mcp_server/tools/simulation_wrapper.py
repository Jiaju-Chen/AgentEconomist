"""
仿真包装器 - 提供仿真控制接口
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# 全局仿真包装器实例
_wrapper_instance: Optional[Any] = None


def get_wrapper():
    """
    获取仿真包装器实例
    
    Returns:
        仿真包装器实例，如果未初始化则返回占位对象
    """
    global _wrapper_instance
    
    if _wrapper_instance is None:
        # 返回一个占位对象，提供基本的接口
        _wrapper_instance = _PlaceholderWrapper()
        logger.warning("使用占位仿真包装器，仿真控制功能可能不可用")
    
    return _wrapper_instance


class _PlaceholderWrapper:
    """占位仿真包装器，提供基本接口但不执行实际操作"""
    
    async def pause_simulation(self) -> Dict[str, Any]:
        """暂停仿真（占位实现）"""
        return {
            "success": False,
            "message": "仿真包装器未初始化，暂停功能不可用"
        }
    
    async def resume_simulation(self) -> Dict[str, Any]:
        """恢复仿真（占位实现）"""
        return {
            "success": False,
            "message": "仿真包装器未初始化，恢复功能不可用"
        }
    
    async def inject_intervention(
        self,
        intervention_type: str,
        target_month: int,
        parameters: dict,
        description: str = ""
    ) -> Dict[str, Any]:
        """注入干预（占位实现）"""
        return {
            "success": False,
            "message": "仿真包装器未初始化，干预功能不可用"
        }
    
    def list_pending_interventions(self) -> list:
        """列出待执行的干预（占位实现）"""
        return []
    
    def cancel_intervention(self, intervention_id: str) -> Dict[str, Any]:
        """取消干预（占位实现）"""
        return {
            "success": False,
            "message": "仿真包装器未初始化，取消功能不可用"
        }

