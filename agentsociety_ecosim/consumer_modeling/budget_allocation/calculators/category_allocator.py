"""
大类预算分配器

负责将月度预算分配到各个大类消费类别（17类），结合：
- QAIDS模型（基于历史数据）
- LLM微调（考虑家庭画像和属性需求）
- 属性引导（优先满足家庭属性缺口）

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import logger
from typing import Dict, List, Any, Optional

from agentsociety_ecosim.consumer_modeling import llm_utils, QAIDS_model
from agentsociety_ecosim.consumer_modeling.family_data import get_family_consumption_and_profile_by_id
from ..config import BudgetConfig
from ..utils import BudgetUtils, BudgetOptimizer, ProfileBuilder, PromptBuilder
from ..managers import LegacyDataConverter

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class CategoryAllocator:
    """
    大类预算分配器
    
    负责将月度预算分配到17个大类消费类别：
    - 使用QAIDS模型基于历史数据进行初始分配
    - 使用LLM进行微调，考虑家庭画像和属性需求
    - 支持属性引导，优先满足家庭属性缺口
    """
    
    def __init__(
        self,
        category_keys: List[str] = None,
        legacy_category_keys: List[str] = None,
        category_names_zh: Dict[str, str] = None,
        attribute_to_category_mapping: Dict[str, Any] = None,
        llm_semaphore: Any = None
    ):
        """
        初始化大类预算分配器
        
        Args:
            category_keys: 预算类别键列表（17类）
            legacy_category_keys: 旧版预算类别键列表（13类）
            category_names_zh: 类别中文名称映射
            attribute_to_category_mapping: 属性到类别的映射
            llm_semaphore: LLM并发控制信号量
        """
        self.category_keys = category_keys or BudgetConfig.CATEGORY_KEYS
        self.legacy_category_keys = legacy_category_keys or BudgetConfig.LEGACY_CATEGORY_KEYS
        self.category_names_zh = category_names_zh or BudgetConfig.CATEGORY_NAMES_ZH
        self.attribute_to_category_mapping = attribute_to_category_mapping or BudgetConfig.ATTRIBUTE_TO_CATEGORY_MAPPING
        self.llm_semaphore = llm_semaphore
        
        # 初始化工具类
        self.legacy_converter = LegacyDataConverter()
    
    async def allocate_monthly_budget_to_categories(
        self, 
        monthly_budget: float, 
        family_id: str, 
        ex_info: Optional[str] = None, 
        current_month: Optional[int] = None, 
        family_profile: Optional[str] = None, 
        attribute_gaps: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        将月度预算分配到大类消费类别（支持属性引导）
        
        Args:
            monthly_budget: 当前月份预算
            family_id: 家庭ID
            ex_info: 额外信息
            current_month: 当前月份
            family_profile: 家庭画像
            attribute_gaps: 家庭属性缺口（用于引导预算分配）
            
        Returns:
            Dict[str, float]: 大类消费预算分配结果
        """
        try:
            # 1. 获取家庭信息和历史消费数据
            family_info = self._get_family_info(family_id)
            if not family_profile:
                family_profile = ProfileBuilder.build_family_profile_for_allocation(family_info)
            # 拼接 ex_info（只拼接一次）
            if ex_info:
                family_profile = ex_info + "\n" + family_profile
            
            # 2. 获取过去五年的年度大类消费记录（排除2021年）
            historical_data = self._get_historical_consumption_data(family_info)
            
            # 3. 使用QAIDS方法分配预算
            qaids_allocation = self._allocate_with_qaids(monthly_budget, historical_data)
            
            if qaids_allocation:
                # 4. 使用LLM进行微调（支持属性引导）
                final_allocation = await self._adjust_allocation_with_llm(
                    qaids_allocation, monthly_budget, historical_data, family_profile, 
                    attribute_gaps=attribute_gaps
                )
                logger.info(f"QAIDS+LLM微调完成，家庭{family_id}月度预算分配: {final_allocation}")
                return final_allocation
            else:
                # QAIDS失败，直接使用LLM分配（支持属性引导）
                logger.warning(f"QAIDS分配失败，家庭{family_id}使用LLM直接分配")
                return await self._allocate_with_llm_direct(monthly_budget, family_profile, attribute_gaps=attribute_gaps)
                
        except Exception as e:
            logger.error(f"月度预算分配失败，家庭{family_id}: {e}")
            # 降级到默认分配
            return BudgetUtils.get_default_allocation(monthly_budget, self.category_keys)
    
    def _get_family_info(self, family_id: str) -> Dict:
        """获取家庭信息"""
        try:
            family_info = get_family_consumption_and_profile_by_id(family_id)
            return family_info or {}
        except Exception as e:
            logger.warning(f"获取家庭{family_id}信息失败: {e}")
            return {}
    
    def _get_historical_consumption_data(self, family_info: Dict) -> List[List[float]]:
        """
        获取过去五年的年度大类消费记录（排除2021年）
        将年度数据除以12转换为月度平均消费记录
        
        注意：自动处理老格式（13类）到新格式（17类）的转换
        """
        historical_data = []
        
        try:
            consumption_data = family_info.get("consumption", {})
            if not consumption_data:
                # 没有历史数据，创建默认数据
                default_monthly_amount = 1000.0  # 默认月度总支出
                equal_share = default_monthly_amount / len(self.category_keys)
                for _ in range(5):
                    historical_data.append([equal_share] * len(self.category_keys))
                return historical_data
            
            # 获取家庭年收入（用于商品/服务拆分）
            annual_income = family_info.get("income", {}).get("2020", None)
            if annual_income is None:
                # 尝试从其他年份获取收入
                income_data = family_info.get("income", {})
                for year in sorted(income_data.keys(), reverse=True):
                    if income_data[year] and income_data[year] > 0:
                        annual_income = income_data[year]
                        break
            
            # 获取年份列表，排除2021年，按年份降序排列
            years = [y for y in sorted(consumption_data.keys(), reverse=True) if y != "2021"]
            
            for year in years[:5]:  # 最多取5年
                year_data = consumption_data[year]
                if not year_data:
                    continue
                
                # 首先按老的13类格式读取PSID数据
                legacy_budget = {}
                for category in self.legacy_category_keys:
                    expenditure = self._get_category_expenditure_from_psid(year_data, category)
                    legacy_budget[category] = expenditure
                
                # 将老格式转换为新格式（17类）
                new_budget = self.legacy_converter.convert_legacy_budget(legacy_budget, annual_income)
                
                # 按CATEGORY_KEYS顺序提取支出金额
                category_expenditures = [new_budget.get(cat, 0.0) for cat in self.category_keys]
                
                # 计算总支出
                total_expenditure = sum(category_expenditures)
                
                if total_expenditure > 0:
                    # 将年度支出除以12转换为月度平均支出
                    monthly_expenditures = [exp / 12.0 for exp in category_expenditures]
                    historical_data.append(monthly_expenditures)
                else:
                    # 总支出为0，使用默认月度支出
                    default_monthly_amount = 1000.0
                    equal_share = default_monthly_amount / len(self.category_keys)
                    monthly_expenditures = [equal_share] * len(self.category_keys)
                    historical_data.append(monthly_expenditures)
            
            # 如果数据不足5年，用默认数据补充
            while len(historical_data) < 5:
                default_monthly_amount = 1000.0
                equal_share = default_monthly_amount / len(self.category_keys)
                historical_data.append([equal_share] * len(self.category_keys))
            
            logger.info(f"获取到{len(historical_data)}年历史消费数据（月度平均形式），已自动转换为17类格式")
            return historical_data
            
        except Exception as e:
            logger.error(f"获取历史消费数据失败: {e}")
            # 返回默认数据
            default_monthly_amount = 1000.0
            equal_share = default_monthly_amount / len(self.category_keys)
            for _ in range(5):
                historical_data.append([equal_share] * len(self.category_keys))
            return historical_data
    
    def _get_category_expenditure_from_psid(self, year_data: Dict, category: str) -> float:
        """
        从PSID数据中获取指定类别的支出金额
        PSID数据中的类别名称可能与CATEGORY_KEYS不完全匹配，需要进行映射
        """
        # PSID数据中的类别名称映射
        psid_category_mapping = {
            'food_expenditure': ['food_expenditure', 'food_at_home', 'food_away_from_home'],
            'clothing_expenditure': ['clothing_expenditure', 'clothing', 'apparel'],
            'education_expenditure': ['education_expenditure', 'education'],
            'childcare_expenditure': ['childcare_expenditure', 'childcare'],
            'electronics_expenditure': ['electronics_expenditure', 'electronics', 'appliances'],
            'home_furnishing_equipment': ['home_furnishing_equipment', 'furniture', 'home_furnishings'],
            'other_recreation_expenditure': ['other_recreation_expenditure', 'recreation', 'entertainment'],
            'housing_expenditure': ['housing_expenditure', 'housing', 'rent', 'mortgage'],
            'utilities_expenditure': ['utilities_expenditure', 'utilities', 'electricity', 'gas', 'water'],
            'transportation_expenditure': ['transportation_expenditure', 'transportation', 'vehicle'],
            'healthcare_expenditure': ['healthcare_expenditure', 'healthcare', 'medical'],
            'travel_expenditure': ['travel_expenditure', 'travel', 'vacation'],
            'phone_internet_expenditure': ['phone_internet_expenditure', 'phone', 'internet', 'communication']
        }
        
        # 获取可能的类别名称
        possible_names = psid_category_mapping.get(category, [category])
        
        # 尝试从year_data中获取支出金额
        for name in possible_names:
            if name in year_data:
                expenditure = year_data[name]
                if expenditure is not None and expenditure > 0:
                    return float(expenditure)
        
        # 如果没有找到，返回0
        return 0.0
    
    def _allocate_with_qaids(self, monthly_budget: float, historical_data: List[List[float]]) -> Dict[str, float]:
        """使用QAIDS方法分配月度预算"""
        try:
            # 直接使用月度平均消费记录作为QAIDS输入
            # historical_data现在已经是月度平均支出金额，不需要转换
            qaids_allocation = QAIDS_model.predict_q_aids(
                historical_data, 
                monthly_budget, 
                list(self.category_keys)
            )
            
            logger.info(f"QAIDS分配结果: {qaids_allocation}")
            return qaids_allocation
            
        except Exception as e:
            logger.error(f"QAIDS分配失败: {e}")
            return {}
    
    async def _adjust_allocation_with_llm(
        self, 
        qaids_allocation: Dict[str, float], 
        monthly_budget: float, 
        historical_data: List[List[float]], 
        family_profile: str, 
        attribute_gaps: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        使用LLM对QAIDS分配结果进行微调（支持属性引导）
        """
        try:
            # 构建历史数据描述
            historical_description = ProfileBuilder.build_historical_description(historical_data, self.category_keys)
            
            # 构建属性引导文本
            attribute_guidance = ""
            if attribute_gaps:
                attribute_guidance = self._build_attribute_guidance_prompt(attribute_gaps)
            
            # 使用PromptBuilder构建prompt（已集成小额预算约束）
            prompt = PromptBuilder.build_category_allocation_prompt(
                qaids_allocation=qaids_allocation,
                monthly_budget=monthly_budget,
                historical_description=historical_description,
                family_profile=family_profile,
                attribute_guidance=attribute_guidance,
                category_keys=self.category_keys,
                category_names_zh=self.category_names_zh
            )
            
            # ========================================
            # 🔧 打印：完整的大类预算分配提示词
            # # ========================================
            # logger.info(f"\n{'='*80}\n【步骤2: 大类预算分配 - LLM提示词 (QAIDS微调)】\n{'='*80}")
            # logger.info(f"{prompt}")
            # logger.info(f"{'='*80}\n")
            
            # 使用全局LLM信号量控制并发
            async with self.llm_semaphore:
                content = await llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a professional financial planner. Always respond with valid JSON."
                )
            
            # ========================================
            # 🔧 打印：完整的LLM响应
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤2: 大类预算分配 - LLM响应】\n{'='*80}")
            # logger.info(f"{content}")
            # logger.info(f"{'='*80}\n")
            
            # 解析响应
            adjusted_allocation = llm_utils.parse_model_response(content)
            
            # 验证和归一化
            if adjusted_allocation and isinstance(adjusted_allocation, dict):
                # 确保所有值都是数字
                numeric_allocation = {}
                for k, v in adjusted_allocation.items():
                    if isinstance(v, (int, float)) and v >= 0:
                        numeric_allocation[k] = float(v)
                    else:
                        logger.warning(f"Invalid allocation value for {k}: {v}, setting to 0")
                        numeric_allocation[k] = 0.0
                
                if numeric_allocation:
                    total_allocated = sum(numeric_allocation.values())
                    # logger.info(f"🔍 [调试] LLM原始返回总额: ${total_allocated:.2f}, 目标预算: ${monthly_budget:.2f}, 差异: ${total_allocated - monthly_budget:.2f}")
                    
                    if abs(total_allocated - monthly_budget) > 1e-2 and total_allocated > 0:
                        # 归一化到总预算
                        # logger.info(f"⚠️  [调试] 需要归一化: 按比例调整 {monthly_budget / total_allocated:.4f}")
                        numeric_allocation = {k: v * monthly_budget / total_allocated for k, v in numeric_allocation.items()}
                        total_after_scale = sum(numeric_allocation.values())
                        # logger.info(f"✅ [调试] 比例调整后总额: ${total_after_scale:.2f}")
                    
                    # 处理舍入误差并保留两位小数
                    adjusted_allocation = BudgetUtils.normalize_allocation_to_budget(
                        numeric_allocation, monthly_budget, self.category_keys
                    )
                    total_after_normalize = sum(adjusted_allocation.values())
                    # logger.info(f"✅ [调试] normalize后总额: ${total_after_normalize:.2f}")
                    
                    # 使用BudgetOptimizer过滤小额预算
                    adjusted_allocation = BudgetOptimizer.filter_small_budgets(
                        adjusted_allocation, 
                        monthly_budget, 
                        self.category_keys
                    )
                    total_after_filter = sum(adjusted_allocation.values())
                    # logger.info(f"✅ [调试] filter后总额: ${total_after_filter:.2f}")
                    
                    # 最终验证
                    if abs(total_after_filter - monthly_budget) > 0.01:
                        logger.error(f"❌ 严重错误：大类预算总额不符！目标=${monthly_budget:.2f}, 实际=${total_after_filter:.2f}, 差异=${total_after_filter - monthly_budget:.2f}")
                        logger.error(f"   详细分配: {adjusted_allocation}")
                        # 强制再次归一化
                        adjusted_allocation = BudgetUtils.normalize_allocation_to_budget(
                            adjusted_allocation, monthly_budget, self.category_keys
                        )
                        final_total = sum(adjusted_allocation.values())
                        # logger.info(f"🔧 [调试] 强制修正后总额: ${final_total:.2f}")
                    
                    logger.info(f"LLM微调完成: {adjusted_allocation}")
                    return adjusted_allocation
                else:
                    logger.warning("所有分配值无效，返回原始QAIDS分配")
                    return BudgetUtils.normalize_allocation_to_budget(
                        qaids_allocation, monthly_budget, self.category_keys
                    )
            else:
                logger.warning("LLM微调失败，返回原始QAIDS分配")
                return BudgetUtils.normalize_allocation_to_budget(
                    qaids_allocation, monthly_budget, self.category_keys
                )
                
        except Exception as e:
            logger.error(f"LLM微调失败: {e}")
            return BudgetUtils.normalize_allocation_to_budget(
                qaids_allocation, monthly_budget, self.category_keys
            )
    
    async def _allocate_with_llm_direct(
        self, 
        monthly_budget: float, 
        family_profile: str, 
        attribute_gaps: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        直接使用LLM进行大类分配（QAIDS失败时的备选方案，支持属性引导）
        """
        try:
            # 构建属性引导文本
            attribute_guidance = ""
            if attribute_gaps:
                attribute_guidance = self._build_attribute_guidance_prompt(attribute_gaps)
            
            prompt = f"""
You are a professional financial planner. Please allocate the monthly budget to different consumption categories for a family.

Family Profile:
{family_profile}

Monthly Budget: ${monthly_budget:.2f}

Consumption Categories:
"""
            
            for category in self.category_keys:
                category_name = self.category_names_zh.get(category, category)
                prompt += f"- {category}: {category_name}\n"
            
            prompt += f"""
{attribute_guidance}
Please allocate the budget considering:
1. The family's needs and priorities
2. **Family attribute needs (MOST IMPORTANT if attribute guidance is provided above)**

The total must equal exactly ${monthly_budget:.2f}.

Respond with ONLY a JSON object containing the allocation.
"""
            
            # ========================================
            # 🔧 打印：完整的大类预算分配提示词（直接LLM）
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤2: 大类预算分配 - LLM提示词 (直接分配)】\n{'='*80}")
            # logger.info(f"{prompt}")
            # logger.info(f"{'='*80}\n")
            
            # 使用全局LLM信号量控制并发
            async with self.llm_semaphore:
                content = await llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a professional financial planner. Always respond with valid JSON."
                )
            
            # ========================================
            # 🔧 打印：完整的LLM响应
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤2: 大类预算分配 - LLM响应】\n{'='*80}")
            # logger.info(f"{content}")
            # logger.info(f"{'='*80}\n")
            
            allocation = llm_utils.parse_model_response(content)
            
            # 验证和归一化
            if allocation and isinstance(allocation, dict):
                # 确保所有值都是数字
                numeric_allocation = {}
                for k, v in allocation.items():
                    if isinstance(v, (int, float)) and v >= 0:
                        numeric_allocation[k] = float(v)
                    else:
                        logger.warning(f"Invalid allocation value for {k}: {v}, setting to 0")
                        numeric_allocation[k] = 0.0
                
                if numeric_allocation:
                    total_allocated = sum(numeric_allocation.values())
                    if abs(total_allocated - monthly_budget) > 1e-2 and total_allocated > 0:
                        allocation = {k: v * monthly_budget / total_allocated for k, v in numeric_allocation.items()}
                    else:
                        allocation = numeric_allocation
                
                # 处理舍入误差并保留两位小数
                allocation = BudgetUtils.normalize_allocation_to_budget(
                    allocation, monthly_budget, self.category_keys
                )
                
                logger.info(f"LLM直接分配完成: {allocation}")
                return allocation
            else:
                logger.warning("LLM直接分配失败，使用默认分配")
                return BudgetUtils.get_default_allocation(monthly_budget, self.category_keys)
                
        except Exception as e:
            logger.error(f"LLM直接分配失败: {e}")
            return BudgetUtils.get_default_allocation(monthly_budget, self.category_keys)
    
    def _build_attribute_guidance_prompt(self, attribute_gaps: Dict[str, float]) -> str:
        """
        根据家庭属性缺口，生成预算分配引导文本
        
        Args:
            attribute_gaps: 家庭属性缺口 {attribute_name: gap_value}
            
        Returns:
            str: 引导文本
        """
        if not attribute_gaps:
            return ""
        
        # 按重要性和缺口大小筛选需要关注的属性
        critical_attributes = []  # 关键属性缺口 (gap > 2.0)
        high_attributes = []      # 高优先级属性缺口 (gap > 1.0)
        
        for attr, gap in attribute_gaps.items():
            if gap > 2.0:
                critical_attributes.append((attr, gap))
            elif gap > 1.0:
                high_attributes.append((attr, gap))
        
        if not critical_attributes and not high_attributes:
            return ""
        
        # 构建引导文本
        guidance = "\n=== IMPORTANT: Family Attribute Needs Guidance ===\n"
        guidance += "The family has the following attribute gaps that need to be satisfied through purchases:\n\n"
        
        # 关键属性缺口
        if critical_attributes:
            guidance += "🔴 CRITICAL Attribute Gaps (gap > 2.0, MUST address):\n"
            for attr, gap in sorted(critical_attributes, key=lambda x: x[1], reverse=True):
                mapping = self.attribute_to_category_mapping.get(attr, {})
                primary_cats = mapping.get("primary", [])
                cat_names = [self.category_names_zh.get(cat, cat) for cat in primary_cats]
                
                guidance += f"  - {attr}: gap = {gap:.1f}\n"
                guidance += f"    → Increase budget for: {', '.join(cat_names)}\n"
        
        # 高优先级属性缺口
        if high_attributes:
            guidance += "\n🟡 HIGH Priority Attribute Gaps (gap > 1.0, should address):\n"
            for attr, gap in sorted(high_attributes, key=lambda x: x[1], reverse=True):
                mapping = self.attribute_to_category_mapping.get(attr, {})
                primary_cats = mapping.get("primary", [])
                cat_names = [self.category_names_zh.get(cat, cat) for cat in primary_cats]
                
                guidance += f"  - {attr}: gap = {gap:.1f}\n"
                guidance += f"    → Consider increasing: {', '.join(cat_names)}\n"
        
        # 添加建议的最小预算分配
        guidance += "\n📊 Recommended Minimum Budget Allocation:\n"
        guidance += "Based on the attribute gaps above, please ensure the following categories receive adequate budget:\n"
        
        # 收集需要增加预算的类别
        category_priority = {}  # {category: priority_score}
        for attr, gap in critical_attributes + high_attributes:
            mapping = self.attribute_to_category_mapping.get(attr, {})
            primary_cats = mapping.get("primary", [])
            weight = 2.0 if gap > 2.0 else 1.0  # 关键属性权重更高
            
            for cat in primary_cats:
                category_priority[cat] = category_priority.get(cat, 0) + gap * weight
        
        # 按优先级排序
        sorted_categories = sorted(category_priority.items(), key=lambda x: x[1], reverse=True)
        for cat, priority in sorted_categories[:5]:  # 最多显示前5个
            cat_name = self.category_names_zh.get(cat, cat)
            guidance += f"  - {cat_name} ({cat}): priority score = {priority:.1f}\n"
        
        guidance += "\n⚠️  Please adjust the budget allocation to prioritize these categories while maintaining balance.\n"
        guidance += "=" * 60 + "\n"
        
        return guidance

