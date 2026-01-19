"""
月度预算计算器模块

负责计算家庭月度消费预算
- LLM辅助计算
- 规则约束调整
- 最低/默认预算计算

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import re
import logger
from typing import Optional, Dict, Any
from ..utils import ProfileBuilder, PromptBuilder
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class MonthlyBudgetCalculator:
    """月度预算计算器"""
    
    def __init__(self, llm_utils, llm_semaphore):
        """
        初始化计算器
        
        Args:
            llm_utils: LLM工具模块
            llm_semaphore: LLM并发控制信号量
        """
        self.llm_utils = llm_utils
        self.llm_semaphore = llm_semaphore
    
    async def calculate_monthly_budget(
        self, 
        current_income: float, 
        total_balance: float, 
        family_profile: Optional[str] = None,
        last_month_budget: Optional[float] = None,
        last_month_attributes: Optional[Dict] = None
    ) -> float:
        """
        计算月度预算
        
        Args:
            current_income: 当前月收入
            total_balance: 家庭总余额
            family_profile: 家庭画像（可选）
            last_month_budget: 上月预算（可选）
            last_month_attributes: 上月属性满足率（可选）
            
        Returns:
            float: 调整后的月度预算
        """
        stage = "start"
        family_situation = None
        prompt = None
        try:
            # 🔍 调试：打印关键参数
            # 🔧 修复：安全地格式化 last_month_budget，避免字符串类型错误
            last_month_budget_display = f"${last_month_budget:.2f}" if last_month_budget is not None else 'None'
            logger.info(f"🔍 预算计算输入参数: 收入=${current_income:.2f}, 存款=${total_balance:.2f}, 上月预算={last_month_budget_display}")
            
            # 构建家庭状况描述
            stage = "build_family_situation"
            family_situation = ProfileBuilder.build_family_situation_for_llm(
                current_income, total_balance, family_profile
            )
            
            # 🔍 调试：打印家庭状况描述
            # logger.info(f"🔍 家庭状况描述:\n{family_situation}")
            
            # ========================================
            # 🔧 修改：调用改进后的提示词构建（包含历史反馈）
            # ========================================
            stage = "build_prompt"
            prompt = PromptBuilder.build_budget_calculation_prompt(
                family_situation,
                last_month_budget,
                last_month_attributes
            )
            # 打印提示词
            # logger.info(f"提示词:\n{prompt}")
            # 调用LLM计算（使用自定义prompt）

            stage = "call_llm"
            llm_budget = await self._call_llm_for_calculation_with_prompt(prompt)
            
            # 规则调整（传入营养反馈数据和上月预算）
            stage = "adjust_with_rules"
            adjusted_budget = self._adjust_with_rules(
                llm_budget, current_income, total_balance, family_profile, last_month_attributes, last_month_budget
            )
            
            logger.info(f"LLM calculated budget: ${llm_budget:.2f}, Adjusted budget: ${adjusted_budget:.2f}")
            return adjusted_budget
            
        except Exception as e:
            logger.error(f"❌ calculate_monthly_budget 异常 (stage={stage}): {e}")
            logger.error(
                "   调试上下文: income=%.2f, balance=%.2f, last_month_budget=%s, last_month_attributes=%s",
                current_income,
                total_balance,
                last_month_budget,
                list(last_month_attributes.keys()) if isinstance(last_month_attributes, dict) else last_month_attributes,
            )
            if family_situation:
                logger.error("   family_situation: %s", family_situation)
            if prompt:
                logger.error("   prompt_preview: %s", prompt[:500])
            return self._calculate_default_budget(
                current_income, 
                total_balance, 
                family_profile, 
                last_month_attributes, 
                last_month_budget
            )
    
    async def _call_llm_for_calculation_with_prompt(self, prompt: str) -> float:
        """
        使用自定义prompt调用LLM计算预算
        
        Args:
            prompt: 完整的prompt文本
            
        Returns:
            float: LLM返回的预算值
        """
        
        # ========================================
        # 🔧 打印：完整的LLM预算决策提示词
        # ========================================
        # logger.info(f"\n{'='*80}\n【步骤1: 月度预算计算 - LLM提示词】\n{'='*80}")
        # logger.info(f"{prompt}")
        # logger.info(f"{'='*80}\n")
        
        try:
            async with self.llm_semaphore:
                content = await self.llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a professional financial planner specializing in household budget planning."
                )
            
            # ========================================
            # 🔧 打印：完整的LLM响应
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤1: 月度预算计算 - LLM响应】\n{'='*80}")
            # logger.info(f"{content}")
            # logger.info(f"{'='*80}\n")
            
            # 解析响应 - 使用多个正则表达式模式
            patterns = [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $1,000.00 或 $1000
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD)',  # 1000 dollars
                r'budget[:\s]+\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # budget: $1000
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)'  # 最后兜底：任何数字
            ]
            
            budget_value = None
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    # 移除逗号，转换为浮点数
                    budget_str = match.group(1).replace(',', '')
                    budget_value = float(budget_str)
                    
                    # 合理性检查：预算应该在 100 到 1000000 之间
                    if 100 <= budget_value <= 1000000:
                        logger.debug(f"✅ 成功解析预算: ${budget_value:.2f} (使用模式: {pattern})")
                        return budget_value
                    else:
                        logger.warning(f"⚠️ 解析到不合理的预算值: ${budget_value:.2f}，继续尝试其他模式")
                        budget_value = None
            
            # 如果所有模式都失败
            logger.error(f"❌ 无法从LLM响应中解析有效预算。响应内容: {content}")
            raise ValueError(f"No valid budget number found in LLM response: {content[:100]}")
                
        except Exception as e:
            logger.error(f"❌ 解析后的预算值无法转换为 float: {budget_value} (type={type(budget_value)})")
            logger.error(f"LLM budget calculation failed: {e}")
            raise
    
    def _adjust_with_rules(
        self, 
        llm_budget: float, 
        current_income: float, 
        total_balance: float,
        family_profile: Optional[str] = None,
        last_month_attributes: Optional[Dict] = None,
        last_month_budget: Optional[float] = None
    ) -> float:
        """根据规则调整预算"""
        logger.info(f"🔍 LLM原始预算: ${llm_budget:.2f}")
        adjusted = float(llm_budget)

        # ============================================================
        # 1️⃣ 计算最低预算（家庭规模 + 营养）
        # ============================================================
        family_size = self._extract_family_size(family_profile)
        base_min_budget = self._calculate_minimum_budget(current_income, family_size)

        min_budget = base_min_budget
        logger.info(f"🔧 基本最低预算: ${base_min_budget:.2f} (家庭规模: {family_size})")

        # 根据营养情况调整最低预算（提前执行）
        if last_month_attributes:
            over_supplied = sum(1 for rate in last_month_attributes.values() if rate > 200)
            under_supplied = sum(1 for rate in last_month_attributes.values() if rate < 50)

            if over_supplied >= 2 and under_supplied == 0:
                min_budget *= 0.8
                logger.info(f"📊 营养过剩 → 最低预算降低20%: ${base_min_budget:.2f} → ${min_budget:.2f}")
            else:
                logger.info(f"📊 营养不均衡 → 最低预算保持不变")

        # ============================================================
        # 2️⃣ 基于收入的最大预算上限
        # ============================================================
        if current_income > 0:
            income_limit = current_income * 1.2
            if adjusted > income_limit:
                logger.info(f"📊 收入上限调整: ${adjusted:.2f} → ${income_limit:.2f}")
                adjusted = income_limit

        # ============================================================
        # 3️⃣ 基于总资产（余额40%）
        # ============================================================
        balance_limit = total_balance * 0.4
        if adjusted > balance_limit:
            logger.info(f"📊 余额上限40%: ${adjusted:.2f} → ${balance_limit:.2f}")
            adjusted = balance_limit

        # ============================================================
        # 4️⃣ 确保预算不低于最低预算
        # ============================================================
        if adjusted < min_budget:
            logger.info(f"📊 不满足最低预算 → 提升: ${adjusted:.2f} → ${min_budget:.2f}")
            adjusted = min_budget

        # ============================================================
        # 5️⃣ 储蓄调整（仅对有收入家庭）
        # ============================================================
        if current_income > 0:
            savings_ratio = total_balance / current_income

            if savings_ratio < 3:
                new_val = adjusted * 0.9
                logger.info(f"📉 储蓄不足 → 降低预算10%: ${adjusted:.2f} → ${new_val:.2f}")
                adjusted = new_val
            elif savings_ratio > 12:
                new_val = adjusted * 1.1
                logger.info(f"📈 储蓄充足 → 提升预算10%: ${adjusted:.2f} → ${new_val:.2f}")
                adjusted = new_val

            # 仍然不能低于最低预算
            if adjusted < min_budget:
                adjusted = min_budget

        # ============================================================
        # 6️⃣ 平滑机制（最后执行）
        # ============================================================
        if last_month_budget:
            last_month_budget = float(last_month_budget)
            min_smooth = last_month_budget * 0.8
            max_smooth = last_month_budget * 1.2

            absolute_min = min_budget * 0.7

            logger.info(f"🔍 平滑范围: [{min_smooth:.2f}, {max_smooth:.2f}], 绝对底线: ${absolute_min:.2f}")

            if adjusted < min_smooth:
                target = max(min_smooth, absolute_min)
                logger.info(f"📊 平滑向上: ${adjusted:.2f} → ${target:.2f}")
                adjusted = target

            elif adjusted > max_smooth:
                logger.info(f"📊 平滑向下: ${adjusted:.2f} → ${max_smooth:.2f}")
                adjusted = max_smooth

            # 再次保证不低于绝对底线
            if adjusted < absolute_min:
                logger.info(f"⚠️ 最终保护 → 提升到绝对底线: ${absolute_min:.2f}")
                adjusted = absolute_min
        else:
            logger.info(f"ℹ️ 首月 → 无需平滑")

        logger.info(f"✅ 最终预算: ${adjusted:.2f}")
        return adjusted
    
    def _calculate_minimum_budget(self, current_income: float, family_size: float = 1.0) -> float:
        """
        计算最低预算（基于家庭人口）
        
        Args:
            current_income: 当前月收入
            family_size: 家庭人数
            
        Returns:
            float: 最低预算
        """
        # 🔧 优化：降低基础最低预算（原2500太高，导致过度购买）
        # 合理的食品预算应该在$800-1200/月（单人）
        base_min_budget = 1200
        
        # 根据家庭人口线性调整（营养需求按人数成比例增长）
        min_budget = base_min_budget * family_size
        
        # 不超过收入的90%
        if current_income > 0:
            max_min_budget = current_income * 0.9
        else:
            max_min_budget = 10000
        
        return min(min_budget, max_min_budget)
    
    def _calculate_default_budget(
        self,
        current_income: float,
        total_balance: float,
        family_profile: Optional[str] = None,
        last_month_attributes: Optional[Dict] = None,
        last_month_budget: Optional[float] = None
    ) -> float:
        """计算默认预算（LLM失败时）"""

        # ============================================================
        # 1️⃣ 生存需求（基本最低预算 + 营养调整）
        # ============================================================
        family_size = self._extract_family_size(family_profile)
        min_budget = self._calculate_minimum_budget(current_income, family_size)
        base_min_budget = min_budget

        if last_month_attributes:
            over_supplied = sum(1 for r in last_month_attributes.values() if r > 200)
            under_supplied = sum(1 for r in last_month_attributes.values() if r < 50)

            if over_supplied >= 2 and under_supplied == 0:
                min_budget *= 0.8
                logger.info(f"📊 营养过剩 → 最低预算降低20%: ${base_min_budget:.2f} → ${min_budget:.2f}")

        logger.info(f"🔧 默认最低预算: ${min_budget:.2f}")

        # ============================================================
        # 2️⃣ 基于收入的预算能力（软预算）
        # ============================================================
        income_budget = current_income * 0.5  # 50% 更合理
        logger.info(f"🔧 收入预算: ${income_budget:.2f}")

        # ============================================================
        # 3️⃣ 基于储蓄的预算能力（关键规则）
        # ============================================================
        savings_budget = total_balance * 0.02   # 提取2%
        logger.info(f"🔧 储蓄预算: ${savings_budget:.2f}")

        # 默认预算基础值：取三者最大
        default_raw = max(min_budget, income_budget, savings_budget)
        logger.info(f"🔍 default_raw（三者最大）= ${default_raw:.2f}")

        # ============================================================
        # 4️⃣ 能力上限：收入上限 + 储蓄上限
        # ============================================================
        max_cap = min(current_income * 1.2 + total_balance * 0.03,
                      total_balance * 0.5)
        logger.info(f"🔧 最大承受预算 max_cap = ${max_cap:.2f}")

        adjusted = min(default_raw, max_cap)

        # ============================================================
        # 5️⃣ 平滑机制：基于上个月预算
        # ============================================================
        if last_month_budget and last_month_budget > 0:
            low = last_month_budget * 0.8
            high = last_month_budget * 1.2
            logger.info(f"🔍 平滑区间: [{low:.2f}, {high:.2f}]")

            if adjusted < low:
                logger.info(f"📊 平滑上调: {adjusted:.2f} → {low:.2f}")
                adjusted = low
            elif adjusted > high:
                logger.info(f"📊 平滑下调: {adjusted:.2f} → {high:.2f}")
                adjusted = high

        logger.info(f"✅ 默认预算最终结果: ${adjusted:.2f}")
        return adjusted
    
    def _extract_family_size(self, family_profile: Optional[str]) -> float:
        """
        从家庭画像中提取家庭人数
        
        Args:
            family_profile: 家庭画像（字符串或字典）
            
        Returns:
            float: 家庭人数
        """
        if not family_profile:
            return 1.0
        
        try:
            # 如果是字典
            if isinstance(family_profile, dict):
                return float(family_profile.get('family_size', 1.0))
            
            # 如果是字符串，尝试解析
            import re
            match = re.search(r'family[_\s]size[:\s]*(\d+\.?\d*)', str(family_profile), re.IGNORECASE)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.debug(f"Failed to extract family_size: {e}")
        
        return 1.0  # 默认单人

