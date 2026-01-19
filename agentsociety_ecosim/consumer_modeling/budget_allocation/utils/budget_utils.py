"""
预算工具函数模块

本模块包含预算分配过程中使用的工具函数：
- 数值解析
- 预算归一化
- 预算重新分配
- 默认分配生成

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import logger
from typing import Dict, List, Optional

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class BudgetUtils:
    """预算工具函数类"""
    
    @staticmethod
    def parse_numeric_value(value) -> Optional[float]:
        """
        解析数字值，处理带美元符号的字符串
        
        Args:
            value: 待解析的值（可以是数字、字符串等）
            
        Returns:
            Optional[float]: 解析后的数字值，解析失败返回None
            
        Examples:
            >>> BudgetUtils.parse_numeric_value(100)
            100.0
            >>> BudgetUtils.parse_numeric_value("$123.45")
            123.45
            >>> BudgetUtils.parse_numeric_value("1,234.56")
            1234.56
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # 移除美元符号、逗号等非数字字符，只保留数字和小数点
            cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
            try:
                return float(cleaned_value) if cleaned_value else None
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def normalize_allocation_to_budget(
        allocation: Dict[str, float], 
        monthly_budget: float,
        category_keys: List[str] = None
    ) -> Dict[str, float]:
        """
        将分配结果归一化到指定预算，保留两位小数，确保总和等于预算
        
        Args:
            allocation: 原始分配字典
            monthly_budget: 目标预算总额
            category_keys: 类别键列表（用于生成默认分配）
            
        Returns:
            Dict[str, float]: 归一化后的分配字典
        """
        if not allocation:
            return allocation
        
        # 先保留两位小数，确保值是数字类型
        rounded_allocation = {}
        for k, v in allocation.items():
            if isinstance(v, (int, float)):
                rounded_allocation[k] = round(v, 2)
            else:
                # 如果值不是数字，记录警告并设为0
                logger.warning(f"Non-numeric value in allocation: {k}={v}, setting to 0")
                rounded_allocation[k] = 0.0
        
        # 计算总和
        total_allocated = sum(rounded_allocation.values())
        
        # 如果总和不等于预算，调整最大的类别
        if abs(total_allocated - monthly_budget) > 0.01:
            # 找到最大的类别
            if rounded_allocation:
                max_category = max(rounded_allocation.items(), key=lambda x: x[1])[0]
                
                # 计算需要调整的差值
                diff = monthly_budget - total_allocated
                
                # 调整最大类别
                rounded_allocation[max_category] = round(rounded_allocation[max_category] + diff, 2)
        
        return rounded_allocation
    
    @staticmethod
    def redistribute_negative_allocation(
        allocation: Dict[str, float], 
        monthly_budget: float,
        category_keys: List[str]
    ) -> Dict[str, float]:
        """
        重新分配负值分配，将负值类别的预算分配给正值类别
        
        Args:
            allocation: 原始分配字典（可能包含负值）
            monthly_budget: 目标预算总额
            category_keys: 所有类别键列表
            
        Returns:
            Dict[str, float]: 重新分配后的字典（所有值非负）
        """
        # 过滤掉负值和零值
        positive_allocations = {k: v for k, v in allocation.items() if v > 0}
        
        if not positive_allocations:
            # 如果没有正值，使用均匀分配
            equal_share = round(monthly_budget / len(category_keys), 2)
            allocation = {category: equal_share for category in category_keys}
            
            # 处理舍入误差
            total = sum(allocation.values())
            if abs(total - monthly_budget) > 0.01:
                diff = monthly_budget - total
                first_category = category_keys[0]
                allocation[first_category] = round(allocation[first_category] + diff, 2)
        else:
            # 重新分配预算到正值类别
            total_positive = sum(positive_allocations.values())
            if total_positive > 0:
                # 按比例重新分配
                allocation = {}
                for category in category_keys:
                    if category in positive_allocations:
                        proportion = positive_allocations[category] / total_positive
                        allocation[category] = round(monthly_budget * proportion, 2)
                    else:
                        allocation[category] = 0.0
                
                # 处理舍入误差
                total = sum(allocation.values())
                if abs(total - monthly_budget) > 0.01:
                    diff = monthly_budget - total
                    # 调整最大的正值类别
                    max_category = max(positive_allocations.keys(), key=lambda k: positive_allocations[k])
                    allocation[max_category] = round(allocation[max_category] + diff, 2)
            else:
                # 如果总和为0，均匀分配
                equal_share = round(monthly_budget / len(category_keys), 2)
                allocation = {category: equal_share for category in category_keys}
                
                # 处理舍入误差
                total = sum(allocation.values())
                if abs(total - monthly_budget) > 0.01:
                    diff = monthly_budget - total
                    allocation[category_keys[0]] = round(allocation[category_keys[0]] + diff, 2)
        
        return allocation
    
    @staticmethod
    def get_default_allocation(
        monthly_budget: float, 
        category_keys: List[str]
    ) -> Dict[str, float]:
        """
        获取默认的大类分配（所有方法都失败时的备选方案）
        
        使用均匀分配策略，每个类别获得相等的预算份额
        
        Args:
            monthly_budget: 月度预算总额
            category_keys: 所有类别键列表
            
        Returns:
            Dict[str, float]: 均匀分配的预算字典
        """
        # 使用均匀分配
        equal_share = round(monthly_budget / len(category_keys), 2)
        allocation = {category: equal_share for category in category_keys}
        
        # 处理舍入误差
        total = sum(allocation.values())
        if abs(total - monthly_budget) > 0.01:
            diff = monthly_budget - total
            first_category = category_keys[0]
            allocation[first_category] = round(allocation[first_category] + diff, 2)
        
        logger.info(f"使用默认分配: {allocation}")
        return allocation
    
    @staticmethod
    def get_equal_subcategory_allocation(
        subcategories: List[str], 
        budget: float
    ) -> Dict[str, float]:
        """
        均匀分配小类预算
        
        Args:
            subcategories: 小类列表
            budget: 待分配的预算
            
        Returns:
            Dict[str, float]: 小类预算分配字典
        """
        if not subcategories:
            return {}
        
        equal_share = budget / len(subcategories)
        return {subcat: round(equal_share, 2) for subcat in subcategories}
    
    @staticmethod
    def validate_allocation(
        allocation: Dict[str, float], 
        expected_total: float,
        tolerance: float = 0.01
    ) -> bool:
        """
        验证预算分配是否有效
        
        Args:
            allocation: 预算分配字典
            expected_total: 期望的总预算
            tolerance: 允许的误差范围
            
        Returns:
            bool: 分配是否有效
        """
        if not allocation:
            return False
        
        # 检查是否有负值
        if any(v < 0 for v in allocation.values()):
            logger.warning("Allocation contains negative values")
            return False
        
        # 检查总和是否接近期望值
        total = sum(allocation.values())
        if abs(total - expected_total) > tolerance:
            logger.warning(f"Allocation total {total} differs from expected {expected_total}")
            return False
        
        return True


class BudgetOptimizer:
    """
    预算优化器
    
    处理小额预算问题：
    - 设定最小有效预算
    - 过滤小额预算
    - 重新分配到有效类别
    """
    
    # 最小有效预算（低于此值无法购买商品）
    MIN_EFFECTIVE_BUDGETS = {
        # 商品类别
        'food_expenditure': 20.0,
        'clothing_expenditure': 15.0,
        'childcare_expenditure': 15.0,
        'electronics_expenditure': 25.0,
        'home_furnishing_equipment': 15.0,
        'other_recreation_expenditure': 15.0,
        'healthcare_goods_expenditure': 10.0,
        'transportation_goods_expenditure': 10.0,
        'education_goods_expenditure': 10.0,
        
        # 服务类别（可以是任意金额）
        'housing_expenditure': 0.0,
        'utilities_expenditure': 0.0,
        'healthcare_services_expenditure': 0.0,
        'transportation_services_expenditure': 0.0,
        'education_services_expenditure': 0.0,
        'travel_expenditure': 0.0,
        'phone_internet_expenditure': 0.0,
    }
    
    @classmethod
    def filter_small_budgets(
        cls, 
        allocation: Dict[str, float], 
        total_budget: float,
        category_keys: List[str]
    ) -> Dict[str, float]:
        """
        过滤小额预算并重新分配
        
        将低于最小有效预算的类别归零，并将这些预算重新分配给其他类别
        
        Args:
            allocation: 原始预算分配
            total_budget: 总预算
            category_keys: 所有类别键
            
        Returns:
            Dict[str, float]: 优化后的预算分配
        """
        filtered = {}
        small_budget_total = 0.0
        valid_categories = []
        
        # 第一遍：识别小额预算
        for category, amount in allocation.items():
            min_budget = cls.MIN_EFFECTIVE_BUDGETS.get(category, 5.0)
            
            if amount > 0 and amount < min_budget:
                # 预算太小，归零
                small_budget_total += amount
                filtered[category] = 0.0
                logger.info(f"💰 {category}: ${amount:.2f} < ${min_budget} (最小值)，归零并重新分配")
            else:
                # 预算足够或为0
                filtered[category] = amount
                if amount >= min_budget:
                    valid_categories.append(category)
        
        # 第二遍：将小额预算按比例重新分配给有效类别
        if small_budget_total > 0 and valid_categories:
            total_valid = sum(filtered[cat] for cat in valid_categories)
            
            if total_valid > 0:
                for category in valid_categories:
                    proportion = filtered[category] / total_valid
                    additional = small_budget_total * proportion
                    filtered[category] += additional
                    if additional > 0.5:
                        logger.info(f"➕ {category}: +${additional:.2f} (从小额预算重新分配)")
        
        # 归一化确保总和正确
        return BudgetUtils.normalize_allocation_to_budget(filtered, total_budget, category_keys)
    
    @classmethod
    def get_min_budget_prompt_text(cls) -> str:
        """
        生成最小预算约束的prompt文本
        
        Returns:
            str: 用于LLM的约束说明文本
        """
        prompt = "\n⚠️ CRITICAL MINIMUM BUDGET CONSTRAINTS:\n"
        prompt += "For goods categories, there are minimum effective budgets below which purchasing is impractical:\n"
        
        goods_categories = {k: v for k, v in cls.MIN_EFFECTIVE_BUDGETS.items() if v > 0}
        for category, min_amount in sorted(goods_categories.items(), key=lambda x: -x[1]):
            prompt += f"• {category}: minimum ${min_amount:.0f} (otherwise cannot buy meaningful items)\n"
        
        prompt += "\nIMPORTANT RULES:\n"
        prompt += "1. If you cannot allocate at least the minimum amount to a goods category, set it to $0.00 instead\n"
        prompt += "2. Redistribute any small amounts (<minimum) to higher-priority categories\n"
        prompt += "3. Service categories (housing, utilities, healthcare_services, transportation_services, "
        prompt += "education_services, travel, phone_internet) can be any amount including small amounts\n"
        prompt += "4. Prioritize essential categories: food > housing > utilities > healthcare > others\n\n"
        
        return prompt

