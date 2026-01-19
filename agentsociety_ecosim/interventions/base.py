"""
干预系统基础模块
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


@dataclass
class Intervention:
    """干预对象基类"""
    intervention_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intervention_type: str = ""  # parameter_change, policy, shock, injection
    target_month: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: str = "pending"  # pending, executed, cancelled, failed
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type,
            "target_month": self.target_month,
            "parameters": self.parameters,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "result": self.result,
            "error_message": self.error_message
        }
    
    def mark_executed(self, result: Dict[str, Any] = None):
        """标记为已执行"""
        self.status = "executed"
        self.executed_at = datetime.now()
        self.result = result or {}
    
    def mark_failed(self, error_message: str):
        """标记为失败"""
        self.status = "failed"
        self.executed_at = datetime.now()
        self.error_message = error_message
    
    def mark_cancelled(self):
        """标记为已取消"""
        self.status = "cancelled"


class InterventionManager:
    """干预管理器"""
    
    def __init__(self):
        self._interventions: List[Intervention] = []
        self._intervention_history: List[Intervention] = []
    
    def add_intervention(self, intervention: Intervention) -> str:
        """
        添加干预
        
        Returns:
            intervention_id
        """
        self._interventions.append(intervention)
        return intervention.intervention_id
    
    def get_interventions_for_month(self, month: int) -> List[Intervention]:
        """获取指定月份待执行的干预"""
        return [
            i for i in self._interventions
            if i.target_month == month and i.status == "pending"
        ]
    
    def get_pending_interventions(self) -> List[Intervention]:
        """获取所有待执行的干预"""
        return [i for i in self._interventions if i.status == "pending"]
    
    def get_intervention(self, intervention_id: str) -> Optional[Intervention]:
        """根据ID获取干预"""
        for i in self._interventions:
            if i.intervention_id == intervention_id:
                return i
        return None
    
    def cancel_intervention(self, intervention_id: str) -> bool:
        """取消干预"""
        intervention = self.get_intervention(intervention_id)
        if intervention and intervention.status == "pending":
            intervention.mark_cancelled()
            return True
        return False
    
    def mark_executed(self, intervention_id: str, result: Dict[str, Any] = None):
        """标记干预为已执行"""
        intervention = self.get_intervention(intervention_id)
        if intervention:
            intervention.mark_executed(result)
            self._intervention_history.append(intervention)
    
    def mark_failed(self, intervention_id: str, error_message: str):
        """标记干预为失败"""
        intervention = self.get_intervention(intervention_id)
        if intervention:
            intervention.mark_failed(error_message)
            self._intervention_history.append(intervention)
    
    def get_history(self) -> List[Intervention]:
        """获取干预历史"""
        return self._intervention_history
    
    def clear_executed(self):
        """清理已执行的干预"""
        self._interventions = [
            i for i in self._interventions 
            if i.status == "pending"
        ]

