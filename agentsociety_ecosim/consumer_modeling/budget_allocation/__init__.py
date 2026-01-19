"""
Budget Allocation 模块化重构

本模块将原来的 BudgetAllocator 拆分为多个子模块：
- config: 配置管理
- utils: 工具函数  
- managers: 数据管理器
- calculators: 计算器（待实现）
- selectors: 选择器（待实现）

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

from .config import BudgetConfig
from .utils import BudgetUtils, BudgetOptimizer, ProfileBuilder, PromptBuilder
from .managers import LegacyDataConverter, HistoryManager
from .calculators import MonthlyBudgetCalculator, CategoryAllocator, SubcategoryAllocator
from .selectors import ProductSelector

__all__ = [
    'BudgetConfig',
    'BudgetUtils',
    'BudgetOptimizer',
    'ProfileBuilder',
    'PromptBuilder',
    'LegacyDataConverter',
    'HistoryManager',
    'MonthlyBudgetCalculator',
    'CategoryAllocator',
    'SubcategoryAllocator',
    'ProductSelector',
]

__version__ = '1.0.0'

