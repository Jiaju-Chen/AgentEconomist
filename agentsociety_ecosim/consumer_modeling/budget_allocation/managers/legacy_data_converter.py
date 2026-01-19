"""
历史数据转换器模块

本模块负责将旧格式（13类）预算数据转换为新格式（17类）
- 旧格式：healthcare_expenditure（混合）
- 新格式：healthcare_goods_expenditure + healthcare_services_expenditure（分离）

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import logger
from typing import Dict, Optional, Tuple
from ..config import BudgetConfig

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class LegacyDataConverter:
    """历史数据转换器"""
    
    def __init__(self):
        """初始化转换器"""
        self.split_ratios = BudgetConfig.SPLIT_RATIOS
        self.split_categories = {
            "healthcare_expenditure": (
                "healthcare_goods_expenditure", 
                "healthcare_services_expenditure"
            ),
            "transportation_expenditure": (
                "transportation_goods_expenditure", 
                "transportation_services_expenditure"
            ),
            "education_expenditure": (
                "education_goods_expenditure", 
                "education_services_expenditure"
            )
        }
    
    def split_mixed_category(
        self, 
        category_name: str, 
        expenditure_value: float, 
        annual_income: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        将混合类别支出拆分为商品和服务部分
        
        Args:
            category_name: 类别名称（healthcare_expenditure等）
            expenditure_value: 该类别的支出金额
            annual_income: 年收入（可选，用于调整拆分比例）
            
        Returns:
            tuple: (goods_expenditure, services_expenditure) 商品支出和服务支出
            
        Example:
            >>> converter = LegacyDataConverter()
            >>> goods, services = converter.split_mixed_category("healthcare_expenditure", 5000, 60000)
            >>> print(f"商品: ${goods}, 服务: ${services}")
            商品: $900.0, 服务: $4100.0
        """
        if expenditure_value is None:
            return None, None
        
        # 确定收入水平
        income_level = BudgetConfig.get_income_level(annual_income)
        
        # 获取拆分比例
        if category_name in self.split_ratios[income_level]:
            ratios = self.split_ratios[income_level][category_name]
        else:
            # 默认50-50拆分
            logger.warning(f"No split ratio for {category_name}, using 50-50 default")
            ratios = {"goods": 0.5, "services": 0.5}
        
        # 计算拆分金额
        goods = expenditure_value * ratios["goods"]
        services = expenditure_value * ratios["services"]
        
        return goods, services
    
    def convert_legacy_budget(
        self, 
        budget_dict: Dict[str, float], 
        annual_income: Optional[float] = None
    ) -> Dict[str, float]:
        """
        将旧格式预算（13类）转换为新格式（17类）
        
        Args:
            budget_dict: 旧格式预算字典，包含healthcare_expenditure等混合类别
            annual_income: 年收入（可选，用于调整拆分比例）
            
        Returns:
            dict: 新格式预算字典，混合类别已拆分为商品和服务
            
        Example:
            >>> converter = LegacyDataConverter()
            >>> old_budget = {
            ...     "food_expenditure": 8000,
            ...     "healthcare_expenditure": 5000,
            ...     "transportation_expenditure": 10000,
            ...     "education_expenditure": 3000
            ... }
            >>> new_budget = converter.convert_legacy_budget(old_budget, 60000)
        """
        new_budget = {}
        
        for category, amount in budget_dict.items():
            if category in self.split_categories:
                # 拆分混合类别
                goods, services = self.split_mixed_category(category, amount, annual_income)
                goods_cat, services_cat = self.split_categories[category]
                
                new_budget[goods_cat] = goods if goods is not None else 0.0
                new_budget[services_cat] = services if services is not None else 0.0
            else:
                # 其他类别直接复制
                new_budget[category] = amount if amount is not None else 0.0
        
        return new_budget
    
    def is_legacy_format(self, budget_dict: Dict[str, float]) -> bool:
        """
        判断预算字典是否为旧格式
        
        Args:
            budget_dict: 预算字典
            
        Returns:
            bool: True表示是旧格式（包含混合类别）
        """
        mixed_categories = set(self.split_categories.keys())
        budget_keys = set(budget_dict.keys())
        
        # 如果包含任何混合类别，就是旧格式
        return bool(mixed_categories & budget_keys)
    
    def convert_if_legacy(
        self, 
        budget_dict: Dict[str, float], 
        annual_income: Optional[float] = None
    ) -> Dict[str, float]:
        """
        智能转换：如果是旧格式则转换，否则返回原样
        
        Args:
            budget_dict: 预算字典
            annual_income: 年收入
            
        Returns:
            dict: 转换后的预算字典
        """
        if self.is_legacy_format(budget_dict):
            logger.info("Detected legacy format, converting to new format...")
            return self.convert_legacy_budget(budget_dict, annual_income)
        else:
            return budget_dict

