"""
Prompt构建器模块

统一管理所有LLM Prompt模板，便于：
- 集中管理和优化
- A/B测试
- 减少重复代码

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import json
from typing import Dict, List, Optional
from typing import Dict, List, Optional
from .budget_utils import BudgetOptimizer

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class PromptBuilder:
    """LLM Prompt构建器"""
    
    @staticmethod
    def build_budget_calculation_prompt(
        family_situation: str,
        last_month_budget: Optional[float] = None,
        last_month_attributes: Optional[Dict] = None
    ) -> str:
        """
        构建月度预算计算的prompt
        """
        try:
            # =========================
            # ① 核心提示词（基础规则）
            # =========================
            prompt = f"""
You are a financial planner for a household.
Your task is to calculate a **reasonable monthly budget**.

===============================
### GLOBAL RULES (ALWAYS APPLY)
===============================
1. Minimum survival line:
   - 1-person ≥ $1500
   - 2+ persons ≥ $2500

2. Households WITH income:
   - Budget = **70%–90% of monthly income**

3. Households WITHOUT income:
   - Savings < $20,000 → 8%–12% of savings
   - Savings < $100,000 → 4%–6% of savings
   - Savings ≥ $100,000 → 2%–4% of savings

Your output MUST be JSON only:
{{"monthly_budget": NUMBER}}

"""

            # =========================
            # ② 上月属性检查 + 数值清洗
            # =========================
            safe_attributes = {}
            if last_month_attributes is not None:
                if not isinstance(last_month_attributes, dict):
                    raise TypeError(
                        f"last_month_attributes 格式错误，应为 dict，但收到: {type(last_month_attributes)}, 值={last_month_attributes}"
                    )
                for attr, value in last_month_attributes.items():
                    try:
                        safe_attributes[attr] = float(value)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"last_month_attributes 数值无效: {attr}={value} ({type(value)}), 错误: {e}"
                        ) from e

            # =========================
            # ③ 首月（必须忽略营养 & 平滑规则）
            # =========================
            if (last_month_budget is None) or (not safe_attributes):
                prompt += """
===============================
### FIRST MONTH CONDITION
===============================
This is the **first month**.
There is **no last-month budget** and **no last-month nutrition metrics**.

➡️ IGNORE:
- all nutrition adjustment rules,
- all smoothing rules,
- all percentage increase/decrease rules.

Only use:
- household income,
- household savings,
- survival minimum,
- and recommended percent range.
"""
            else:
                # =========================
                # ④ 有上月预算 → 动态反馈（第2个月及之后）
                # =========================
                try:
                    last_month_budget_value = float(last_month_budget)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"last_month_budget 类型转换失败: {last_month_budget} ({type(last_month_budget)}), 错误: {e}"
                    ) from e

                critical = [f"{k}:{v:.0f}%" for k, v in safe_attributes.items() if v < 70]
                suboptimal = [f"{k}:{v:.0f}%" for k, v in safe_attributes.items() if 70 <= v < 90]
                excessive = [f"{k}:{v:.0f}%" for k, v in safe_attributes.items() if v > 150]

                prompt += f"""
===============================
### LAST MONTH PERFORMANCE
===============================
Last-month budget: ${last_month_budget_value:,.0f}
"""

                if critical:
                    prompt += f"- Nutrition severely insufficient ({', '.join(critical)}). Increase **5–15%**.\n"
                elif suboptimal:
                    prompt += f"- Nutrition below ideal ({', '.join(suboptimal)}). Increase **5–10%**.\n"
                elif excessive:
                    prompt += f"- Nutrition excessive ({', '.join(excessive)}). Decrease **5–15%**.\n"
                else:
                    prompt += "- Nutrition balanced. Keep budget similar.\n"

                prompt += """
- Total monthly change MUST stay within ±20%.
"""

            # =========================
            # ⑤ 解析家庭情况（收入/余额/家庭规模）
            # =========================
            import re
            balance_match = re.search(r'Total available balance: \$([0-9,]+\.?\d*)', family_situation)
            income_match = re.search(r'Current monthly income: \$([0-9,]+\.?\d*)', family_situation)
            family_size_match = re.search(r'Family Size: ([0-9\.]+)', family_situation)

            try:
                balance = float(balance_match.group(1).replace(',', '')) if balance_match else 0.0
            except Exception as e:
                raise ValueError(f"无法解析余额: {balance_match.group(1) if balance_match else 'N/A'}, 错误: {e}") from e

            try:
                income = float(income_match.group(1).replace(',', '')) if income_match else 0.0
            except Exception as e:
                raise ValueError(f"无法解析收入: {income_match.group(1) if income_match else 'N/A'}, 错误: {e}") from e

            try:
                family_size = int(float(family_size_match.group(1))) if family_size_match else 1
            except Exception as e:
                raise ValueError(f"无法解析家庭规模: {family_size_match.group(1) if family_size_match else 'N/A'}, 错误: {e}") from e

            min_survival_budget = 1500 if family_size == 1 else 2500

            # =========================
            # ⑥ 推荐预算范围
            # =========================
            prompt += f"""
===============================
### HOUSEHOLD INFORMATION
===============================
{family_situation}

===============================
### RECOMMENDED RANGE (STRICT)
===============================
Minimum survival: **${min_survival_budget:,.0f}**

Income: ${income:,.0f}
Savings: ${balance:,.0f}

If household **has income**:
- Use **70%–90% of income**.

If household **has NO income**:
Use a **small fraction of savings**:
- < $20,000  → 8% – 12%
- < $100,000 → 4% – 6%
- ≥ $100,000 → 2% – 4%

Output JSON only:
{{"monthly_budget": AMOUNT}}
"""

            return prompt

        except (ValueError, TypeError) as e:
            logger.error(f"❌ build_budget_calculation_prompt 失败: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ build_budget_calculation_prompt 发生未预期错误: {e}")
            raise ValueError(f"构建预算计算prompt时发生错误: {e}") from e
    
    @staticmethod
    def build_category_allocation_prompt(
        qaids_allocation: Dict[str, float],
        monthly_budget: float,
        historical_description: str,
        family_profile: str,
        attribute_guidance: str = "",
        category_keys: List[str] = None,
        category_names_zh: Dict[str, str] = None
    ) -> str:
        """
        构建大类预算分配的prompt（QAIDS微调）- 简化版
        
        Args:
            qaids_allocation: QAIDS初始分配结果
            monthly_budget: 月度预算总额
            historical_description: 历史消费描述（不再使用）
            family_profile: 家庭画像
            attribute_guidance: 属性引导文本（可选）
            category_keys: 类别键列表（可选）
            category_names_zh: 类别中文名（可选）
            
        Returns:
            str: 完整的prompt文本
        """
        # 简化家庭画像：只提取关键信息，去除重复的 ex_info
        simplified_profile = PromptBuilder._simplify_family_profile(family_profile)
        
        prompt = f"""Adjust the budget allocation for this family.
**Total Budget**: ${monthly_budget:.2f} (must equal exactly)

{simplified_profile}

**QAIDS Baseline**:
{json.dumps(qaids_allocation, indent=2)}
{attribute_guidance}
**Priority**: Food is highest priority if nutrition gaps exist.

{BudgetOptimizer.get_min_budget_prompt_text()}
**Output**: JSON only, no explanation."""
        
        # 添加类别说明（如果提供）
        if category_keys and category_names_zh:
            prompt += "\n\nCategories:\n"
            for key in category_keys:
                zh_name = category_names_zh.get(key, "")
                prompt += f"  • {key}: {zh_name}\n"
        
        return prompt
    
    @staticmethod
    def _simplify_family_profile(family_profile: str) -> str:
        """
        简化家庭画像：去除重复的 ex_info，只保留关键信息
        """
        import re
        
        # 去除重复的 ex_info（只保留第一次出现）
        ex_info_pattern = r'(=== Current Household Employment Status ===.*?=== Please consider.*?===)'
        ex_info_matches = list(re.finditer(ex_info_pattern, family_profile, re.DOTALL))
        
        if len(ex_info_matches) > 1:
            # 有重复，移除后面的
            for match in ex_info_matches[1:]:
                family_profile = family_profile.replace(match.group(0), '', 1)
        
        # 提取关键信息
        lines = []
        
        # 保留 ex_info（就业和税收信息）
        if ex_info_matches:
            lines.append(ex_info_matches[0].group(0))
        
        # 提取基本家庭信息
        lines.append("\n**Family Info**:")
        
        match = re.search(r'Family Size:\s*([\d.]+)', family_profile)
        if match:
            lines.append(f"- Size: {match.group(1)} people")
        
        match = re.search(r'Head Age:\s*([\d.]+)', family_profile)
        if match:
            lines.append(f"- Head age: {match.group(1)}")
        
        match = re.search(r'Number of Children:\s*([\d.]+)', family_profile)
        if match:
            lines.append(f"- Children: {match.group(1)}")
        
        match = re.search(r'Marital Status:\s*(\w+)', family_profile)
        if match:
            lines.append(f"- Marital: {match.group(1)}")
        
        return "\n".join(lines)
    
    @staticmethod
    def build_direct_allocation_prompt(
        monthly_budget: float,
        family_profile: str,
        historical_description: str,
        attribute_guidance: str = "",
        category_keys: List[str] = None,
        category_names_zh: Dict[str, str] = None
    ) -> str:
        """
        构建直接分配的prompt（QAIDS失败时使用）
        
        Args:
            monthly_budget: 月度预算总额
            family_profile: 家庭画像
            historical_description: 历史消费描述
            attribute_guidance: 属性引导文本（可选）
            category_keys: 类别键列表（可选）
            category_names_zh: 类别中文名（可选）
            
        Returns:
            str: 完整的prompt文本
        """
        prompt = f"""You are a professional financial planner. Please allocate a monthly budget for a family across different spending categories.

Family Profile:
{family_profile}

Historical Consumption Patterns:
{historical_description}

Total Monthly Budget: ${monthly_budget:.2f}
{attribute_guidance}
Please allocate this budget considering:
1. Family's specific needs and composition
2. Historical spending patterns
3. Essential vs discretionary spending
4. **Family attribute needs (HIGHEST PRIORITY if guidance provided above)**

{BudgetOptimizer.get_min_budget_prompt_text()}
The total must equal exactly ${monthly_budget:.2f}.

Respond with ONLY a JSON object with category keys and allocation amounts."""
        
        # 添加类别说明
        if category_keys and category_names_zh:
            prompt += "\n\nAvailable Categories:\n"
            for key in category_keys:
                zh_name = category_names_zh.get(key, "")
                prompt += f"  • {key}: {zh_name}\n"
        
        return prompt
    
    @staticmethod
    def build_subcategory_allocation_prompt(
        category: str,
        budget: float,
        subcategories: List[str],
        category_name_zh: str = ""
    ) -> str:
        """
        构建小类预算分配的prompt
        
        Args:
            category: 大类名称
            budget: 大类预算
            subcategories: 小类列表
            category_name_zh: 大类中文名（可选）
            
        Returns:
            str: 完整的prompt文本
        """
        cat_display = f"{category} ({category_name_zh})" if category_name_zh else category
        
        prompt = f"""You are a budget allocation expert. Please allocate the budget for category "{cat_display}" across its subcategories.

Total Budget for {cat_display}: ${budget:.2f}

Available Subcategories:
"""
        for subcat in subcategories:
            prompt += f"  • {subcat}\n"
        
        prompt += f"""
Consider:
1. Family needs and priorities
2. Typical spending patterns in each subcategory
3. Seasonal factors
4. Essential vs optional items in each subcategory

The total must equal exactly ${budget:.2f}.

Respond with ONLY a JSON object mapping subcategory names to amounts."""
        
        return prompt
    
    @staticmethod
    def build_product_selection_prompt(
        category: str,
        subcategory_budgets: Dict[str, float],
        candidates: Dict[str, List[Dict]],
        family_attributes: Dict[str, float] = None
    ) -> str:
        """
        构建商品选择的prompt
        
        Args:
            category: 大类名称
            subcategory_budgets: 小类预算字典
            candidates: 候选商品字典 {subcategory: [products]}
            family_attributes: 家庭属性缺口（可选）
            
        Returns:
            str: 完整的prompt文本
        """
        total_budget = sum(subcategory_budgets.values())
        
        prompt = f"""You are a shopping assistant. Please select products for category "{category}" within the budget.

Total Budget: ${total_budget:.2f}

Subcategory Budgets:
"""
        for subcat, budget in subcategory_budgets.items():
            prompt += f"  • {subcat}: ${budget:.2f}\n"
        
        # 添加属性引导（如果有）
        if family_attributes:
            urgent_attrs = {k: v for k, v in family_attributes.items() if v > 2.0}
            if urgent_attrs:
                prompt += "\n⚠️ URGENT FAMILY NEEDS (prioritize products that address these):\n"
                for attr, gap in sorted(urgent_attrs.items(), key=lambda x: -x[1])[:5]:
                    prompt += f"  • {attr}: gap={gap:.2f}\n"
        
        prompt += "\nAvailable Products:\n"
        
        # 列出候选商品
        for subcat, products in candidates.items():
            if products:
                prompt += f"\n{subcat} ({len(products)} items):\n"
                for i, prod in enumerate(products[:15], 1):  # 最多显示15个
                    name = prod.get('name', 'Unknown')
                    price = prod.get('price', 0)
                    prompt += f"  {i}. {name} - ${price:.2f}\n"
        
        prompt += f"""
Selection Criteria:
1. Stay within budget for each subcategory
2. Prioritize essential items
3. Select diverse products to meet family needs
4. Consider price-performance ratio
5. Address urgent family attribute needs if provided above

Respond with ONLY a JSON object in this format:
{{
  "subcategory_name": [
    {{"product_name": "exact product name", "quantity": 1}},
    ...
  ],
  ...
}}

Important: Use exact product names as shown above. Select products wisely to maximize budget utilization."""
        
        return prompt

