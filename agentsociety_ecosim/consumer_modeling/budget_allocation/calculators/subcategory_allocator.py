"""
小类预算分配器

负责将大类预算进一步分配到小类（商品子类别），使用：
- LLM智能分配（考虑家庭画像和商品类别特点）
- 并行处理（提高效率）
- 均匀分配（备选方案）

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import logger
import asyncio
from typing import Dict, List, Any, Union, Optional

from agentsociety_ecosim.consumer_modeling import llm_utils
from ..config import BudgetConfig
from ..utils import BudgetUtils, ProfileBuilder

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class SubcategoryAllocator:
    """
    小类预算分配器
    
    负责将大类预算分配到具体的商品小类：
    - 使用LLM进行智能分配
    - 支持并行处理多个大类
    - 提供均匀分配作为备选方案
    """
    
    def __init__(
        self,
        budget_to_walmart_main: Dict[str, List[str]] = None,
        category_keys: List[str] = None,
        llm_semaphore: Any = None
    ):
        """
        初始化小类预算分配器
        
        Args:
            budget_to_walmart_main: 大类到小类的映射
            category_keys: 预算类别键列表
            llm_semaphore: LLM并发控制信号量
        """
        self.budget_to_walmart_main = budget_to_walmart_main or BudgetConfig.get_budget_to_walmart_main()
        self.category_keys = category_keys or BudgetConfig.CATEGORY_KEYS
        self.llm_semaphore = llm_semaphore
    
    async def allocate_subcategory_budget(
        self, 
        category_budget: Dict[str, float], 
        family_id: str, 
        max_workers: int = 32, 
        ex_info: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        将大类预算分配到小类（使用多线程处理）
        
        Args:
            category_budget: 大类预算分配
            family_id: 家庭ID
            max_workers: 最大工作线程数，默认32
            ex_info: 额外信息
            
        Returns:
            Dict[str, Dict[str, float]]: 小类预算分配结果
                格式: {category: {subcategory: budget}}
        """
        subcategory_budget = {}
        
        try:
            # 构建家庭画像
            family_profile = self._get_family_profile_for_budget_calculation(family_id)
            if ex_info:
                family_profile = ex_info + "\n " + family_profile
            
            # 准备并行处理任务
            tasks = []
            for category, budget in category_budget.items():
                if budget <= 0:
                    subcategory_budget[category] = {}
                    continue
                
                # 获取该大类下的小类列表
                subcategories = self.budget_to_walmart_main.get(category, [])
                if not subcategories:
                    subcategory_budget[category] = {}
                    continue
                
                # 添加到任务列表
                tasks.append((category, subcategories, budget, family_profile))
            
            if not tasks:
                return subcategory_budget
            
            logger.info(f"开始并行处理 {len(tasks)} 个小类预算分配任务，使用 {max_workers} 个线程")
            
            # 定义处理单个任务的函数
            async def process_subcategory_task(task_data):
                category, subcategories, budget, family_profile = task_data
                try:
                    result = await self._allocate_subcategory_with_llm(
                        category=category,
                        subcategories=subcategories,
                        budget=budget,
                        family_profile=family_profile
                    )
                    return category, result, True
                except Exception as e:
                    logger.warning(f"分配{category}小类预算失败: {e}")
                    # 使用均匀分配作为备选
                    fallback_result = self._get_equal_subcategory_allocation(subcategories, budget)
                    return category, fallback_result, False
            
            # 使用asyncio并发执行异步任务（不限制并发数，由全局LLM信号量控制）
            successful_tasks = 0
            failed_tasks = 0
            
            # 创建并发任务（无限制）
            concurrent_tasks = [process_subcategory_task(task) for task in tasks]
            
            # 并发执行所有任务
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                task_data = tasks[i]
                category = task_data[0]
                
                if isinstance(result, Exception):
                    logger.error(f"处理{category}任务时发生异常: {result}")
                    # 使用均匀分配作为备选方案
                    subcategories = task_data[1]
                    budget = task_data[2]
                    subcategory_budget[category] = self._get_equal_subcategory_allocation(subcategories, budget)
                    failed_tasks += 1
                else:
                    try:
                        category, allocation_result, success = result
                        subcategory_budget[category] = allocation_result
                        if success:
                            successful_tasks += 1
                        else:
                            failed_tasks += 1
                    except Exception as e:
                        logger.error(f"解析{category}任务结果时发生异常: {e}")
                        subcategories = task_data[1]
                        budget = task_data[2]
                        subcategory_budget[category] = self._get_equal_subcategory_allocation(subcategories, budget)
                        failed_tasks += 1
            
            logger.info(f"小类预算分配完成: 成功 {successful_tasks} 个任务, 失败 {failed_tasks} 个任务")
            return subcategory_budget
            
        except Exception as e:
            logger.error(f"分配小类预算失败: {e}")
            logger.error(f"分配小类预算失败: {e}")
            return {}
    
    async def _allocate_subcategory_with_llm(
        self, 
        category: str, 
        subcategories: List[str], 
        budget: float, 
        family_profile: str
    ) -> Dict[str, float]:
        """
        使用LLM分配小类预算
        
        Args:
            category: 大类名称
            subcategories: 小类列表
            budget: 大类总预算
            family_profile: 家庭画像
            
        Returns:
            Dict[str, float]: 小类预算分配
        """
        prompt = f"""
You are a professional financial planner. Please allocate the budget for {category} to its subcategories.

Family Profile:
{family_profile}

Category: {category}
Total Budget: ${budget:.2f}

Subcategories:
"""
        
        for subcat in subcategories:
            prompt += f"- {subcat}\n"
        
        prompt += f"""
Please allocate the budget considering the family's needs and priorities.
The total must equal exactly ${budget:.2f}.

Respond with ONLY a JSON object containing the allocation.
"""
        
        # 使用全局LLM信号量控制并发
        async with self.llm_semaphore:
            content = await llm_utils.call_llm_chat_completion(
                prompt,
                system_content="You are a professional financial planner. Always respond with valid JSON."
            )
        
        allocation = llm_utils.parse_model_response(content)
        if allocation and isinstance(allocation, dict):
            # 过滤出有效的数字值，确保键在subcategories中
            cleaned_allocation = {}
            for k, v in allocation.items():
                if k in subcategories:
                    # 尝试将值转换为数字
                    numeric_value = BudgetUtils.parse_numeric_value(v)
                    if numeric_value is not None and numeric_value >= 0:
                        cleaned_allocation[k] = float(numeric_value)
                    else:
                        logger.warning(f"Invalid value for {k}: {v}, setting to 0")
                        cleaned_allocation[k] = 0.0
            
            # 如果清理后为空，使用均匀分配
            if not cleaned_allocation:
                logger.warning("No valid allocation found, using equal distribution")
                return self._get_equal_subcategory_allocation(subcategories, budget)
            
            # 归一化到预算
            allocation = BudgetUtils.normalize_allocation_to_budget(
                cleaned_allocation, budget, list(cleaned_allocation.keys())
            )
            return allocation
        else:
            logger.warning("LLM返回的分配结果无效，使用均匀分配")
            return self._get_equal_subcategory_allocation(subcategories, budget)
    
    def _get_equal_subcategory_allocation(self, subcategories: List[str], budget: float) -> Dict[str, float]:
        """
        均匀分配小类预算
        
        Args:
            subcategories: 小类列表
            budget: 总预算
            
        Returns:
            Dict[str, float]: 均匀分配的小类预算
        """
        if not subcategories:
            return {}
        
        equal_share = budget / len(subcategories)
        allocation = {subcat: equal_share for subcat in subcategories}
        
        # 处理舍入误差
        allocation = BudgetUtils.normalize_allocation_to_budget(
            allocation, budget, subcategories
        )
        return allocation
    
    def _get_family_profile_for_budget_calculation(self, family_id: str) -> str:
        """
        获取用于预算计算的家庭画像
        
        Args:
            family_id: 家庭ID
            
        Returns:
            str: 家庭画像文本
        """
        # 这个方法需要访问原BudgetAllocator的方法
        # 暂时返回简单的默认画像，实际使用时会通过参数传入
        return f"Family {family_id}"

