import json
import logger
import os
import asyncio
import hashlib
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from collections import OrderedDict

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

# 加载 .env 文件
load_dotenv()


# 全局异步客户端实例，实现真正并发 - 避免每次调用都创建新的客户端
_global_async_client = None

def get_global_async_client():
    """获取全局AsyncOpenAI客户端实例，支持连接复用和真正并发"""
    global _global_async_client
    if _global_async_client is None:
        _global_async_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
            timeout=60.0  # 设置60秒超时
        )
    return _global_async_client


# ==================== LLM响应缓存机制 ====================
class LRUCache:
    """LRU缓存实现，用于缓存LLM响应"""
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.hits += 1
            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # 删除最久未使用的项
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total_requests
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# 全局LLM缓存实例
_llm_cache = LRUCache(max_size=1000)

def _get_cache_key(prompt: str, system_content: str, temperature: float = 0.1) -> str:
    """生成缓存键（使用MD5哈希）"""
    content = f"{prompt}|{system_content}|{temperature}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_llm_cache_stats() -> Dict[str, Any]:
    """获取LLM缓存统计信息"""
    return _llm_cache.get_stats()

def clear_llm_cache():
    """清空LLM缓存"""
    _llm_cache.clear()
    logger.info("LLM缓存已清空")

# def await call_llm_chat_completion(prompt, system_content):
#     """
#     通用LLM对话接口，返回模型回复内容。
#     """
#     model_name = "USD-guiji/deepseek-v3"
#     api_key = "sk-JeCvnVJdFk1SbiUc8Klw6t0wRn4KjT4G9DD7V1zjT9n26NIw"
#     base_url = "http://35.220.164.252:3888/v1/"
#     temperature = 0.1
#     # logger.debug(f"Prompt: {prompt}")
#     client = OpenAI(api_key=api_key, base_url=base_url)
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=temperature,
#         stream=False
#     )
#     content = response.choices[0].message.content.strip()
#     return content

async def call_llm_chat_completion(prompt, system_content, max_retries=3, use_cache=True, call_name="LLM"):
    """
    通用异步LLM对话接口，返回模型回复内容。
    使用全局异步客户端实例，实现真正并发，大幅提升性能。
    带重试机制处理超时和网络错误。
    支持响应缓存以减少重复调用。
    
    Args:
        prompt: 用户提示词
        system_content: 系统提示词
        max_retries: 最大重试次数
        use_cache: 是否使用缓存（默认True）
        call_name: 调用名称（用于性能监控）
    
    Returns:
        str: LLM响应内容
    """
    import time
    overall_start = time.perf_counter()
    
    model_name = os.getenv("MODEL", "")
    temperature = 0.1
    
    # 检查缓存
    cache_check_start = time.perf_counter()
    if use_cache:
        cache_key = _get_cache_key(prompt, system_content, temperature)
        cached_response = _llm_cache.get(cache_key)
        if cached_response is not None:
            # cache_time = time.perf_counter() - overall_start
            # print(f"[{call_name}] 💾 缓存命中 | 耗时:{cache_time:.3f}s")
            return cached_response
    cache_check_time = time.perf_counter() - cache_check_start
    
    # 使用全局异步客户端实例，支持连接复用和真正并发
    client = get_global_async_client()
    
    for attempt in range(max_retries):
        try:
            api_call_start = time.perf_counter()
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=False
            )
            api_call_time = time.perf_counter() - api_call_start
            
            content = response.choices[0].message.content.strip()
            
            # 存入缓存
            cache_save_start = time.perf_counter()
            if use_cache:
                _llm_cache.put(cache_key, content)
            cache_save_time = time.perf_counter() - cache_save_start
            
            # total_time = time.perf_counter() - overall_start
            
            # 📊 详细性能日志（已关闭）
            # print(f"[{call_name}] API调用:{api_call_time:.3f}s | 缓存检查:{cache_check_time:.3f}s | 缓存保存:{cache_save_time:.3f}s | 总计:{total_time:.3f}s | 尝试:{attempt+1}/{max_retries}")
            
            return content
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                print(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待 {wait_time}s 后重试...")
                await asyncio.sleep(wait_time)
            else:
                print(f"LLM调用最终失败: {e}")
                raise e

def parse_model_response(response_text: str) -> Dict[str, float]:
    """
    仅提取第一个 JSON 对象并解析为 dict，不做其他修改。
    """
    try:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model response")
        json_str = response_text[start:end+1]
        allocation = json.loads(json_str)
        return allocation
    except Exception as e:
        # logger.error(f"Failed to parse model response: {e}")
        return {}

async def allocate_with_llm(budget: float, items: List[str], item_names: Dict[str, str] = None, family_profile: str = None) -> Dict[str, float]:
    """
    Use LLM to allocate the given total budget to the specified items. The sum must be exactly the total budget. Output strictly in JSON.
    """
    if not family_profile:
        raise ValueError("family_profile must be provided and not None!")
    # logger.info(f"Allocating budget {budget} to items: {items}")
    prompt = [
        # 1. 目的
        f"Your task is to allocate the total budget of {budget} CNY to the following items. The sum of all allocations MUST be exactly equal to the total budget. Output strictly in JSON format.",
        # 2. 约束
        "Do not assign zero to all items. Do not allocate equally unless you have a strong reason. Consider the typical importance and necessity of each item for a real family.",
        "Respond strictly in JSON format, with keys matching the item identifiers and values as the allocation amounts (rounded to two decimal places). Do not output any other value. ONLY output the JSON format data.",
        # 3. 家庭画像
        f"Family profile: {family_profile}",
        # 4. Items
        "Items (identifier: description):"
    ]
    for item in items:
        name = item_names.get(item, item) if item_names else item
        prompt.append(f"- {item}: {name}")
    prompt.append("")
    prompt.append(f"CURRENT TOTAL BUDGET (MUST MATCH SUM): {budget}")
    prompt.append(f"(If you do not strictly follow the budget, your answer will be considered invalid.)")
    prompt = "\n".join(prompt)
    # logger.debug(f"Prompt: {prompt}")
    try:
        content = await call_llm_chat_completion(
            prompt,
            system_content="You are a professional financial planner."
        )
        allocation = parse_model_response(content)
        # logger.info(f"Raw LLM allocation: {allocation}")
        # 归一化处理，确保分配总和等于预算
        if allocation and abs(sum(allocation.values()) - budget) > 1e-2:
            total = sum(allocation.values())
            if total > 0:
                allocation = {k: round(v * budget / total, 2) for k, v in allocation.items()}
                diff = round(budget - sum(allocation.values()), 2)
                if abs(diff) > 0 and allocation:
                    first_key = next(iter(allocation))
                    allocation[first_key] = round(allocation[first_key] + diff, 2)
        # 兜底：如果全为0，则均匀分配
        if allocation and all(v == 0 for v in allocation.values()):
            equal_share = round(budget / len(items), 2)
            allocation = {item: equal_share for item in items}
            # logger.warning(f"LLM allocated all zeros, fallback to equal allocation: {allocation}")
        return allocation
    except Exception as e:
        # logger.error(f"LLM allocation failed: {e}")
        equal_share = round(budget / len(items), 2)
        allocation = {item: equal_share for item in items}
        # logger.info(f"Equal allocation: {allocation}")
        return allocation

async def adjust_allocation_with_llm(allocation: Dict[str, float], total_budget: float, past: list = None, family_profile: str = None, category_keys: List[str] = None, category_names_zh: Dict[str, str] = None) -> Dict[str, float]:
    """
    Use LLM to review and adjust the initial annual budget allocation for a family, considering the importance of each category and the family's past spending. The sum MUST be exactly the total budget provided. Output strictly in JSON.
    """
    if not family_profile:
        raise ValueError("family_profile must be provided and not None!")
    if category_keys is None:
        category_keys = list(allocation.keys())
    if category_names_zh is None:
        category_names_zh = {k: k for k in category_keys}
    prompt = [
        # 1. 目的
        f"Your task is to review and adjust the following initial annual budget allocation for a family. The sum MUST be exactly the total budget provided. Output strictly in JSON format.",
        # 2. 约束
        "Do not assign zero to all categories. Do not allocate equally unless it is truly reasonable. Consider the importance of each category and the family's past spending.",
        "Respond strictly in JSON, with keys matching the category identifiers and values as the allocation amounts (rounded to two decimal places). Do not output any other text.",
        # 3. 家庭画像
        f"Family profile: {family_profile}",
        # 4. 历史消费
        "Here is the family's past N years of category spending (each row is a year, columns are categories in order):"
    ]
    if past is not None:
        header = ', '.join(category_keys)
        prompt.append(header)
        for row in past:
            prompt.append(', '.join(str(x) for x in row))
    prompt += [
        "",
        f"Total budget: {total_budget}",
        "Initial allocation:",
        json.dumps(allocation, ensure_ascii=False, indent=2),
        "",
        "Categories:",
    ]
    for key in category_keys:
        zh = category_names_zh.get(key, "")
        prompt.append(f"- {key}: {zh}")
    prompt = "\n".join(prompt)
    try:
        content = await call_llm_chat_completion(
            prompt,
            system_content="You are a professional financial planner."
        )
        new_allocation = parse_model_response(content)
        if new_allocation and abs(sum(new_allocation.values()) - total_budget) > 1e-2:
            total = sum(new_allocation.values())
            if total > 0:
                new_allocation = {k: round(v * total_budget / total, 2) for k, v in new_allocation.items()}
                diff = round(total_budget - sum(new_allocation.values()), 2)
                if abs(diff) > 0 and new_allocation:
                    first_key = next(iter(new_allocation))
                    new_allocation[first_key] = round(new_allocation[first_key] + diff, 2)
        if new_allocation and all(v == 0 for v in new_allocation.values()):
            equal_share = round(total_budget / len(category_keys), 2)
            new_allocation = {item: equal_share for item in category_keys}
        return new_allocation
    except Exception as e:
        # logger.error(f"LLM adjustment failed: {e}")
        return allocation

async def allocate_monthly_subcat_budget_with_llm(monthly_allocation: dict, budget_to_walmart_main: dict, family_profile: str = None) -> dict:
    """
    For each month's category budget, use LLM to allocate to subcategories, prioritizing basic needs. The sum must be exactly the total budget. Output strictly in JSON.
    """
    if not family_profile:
        raise ValueError("family_profile must be provided and not None!")
    import logger
    us_holidays = (
        "Consider the following major US holidays and events: New Year's Day (Jan), Easter (Mar/Apr), Memorial Day (May), Independence Day (July), Labor Day (Sep), Halloween (Oct), Thanksgiving (Nov), Christmas (Dec), and school terms (school starts in Aug/Sep, summer break in Jun-Aug). Also consider seasonal changes (e.g., higher utility costs in winter/summer, back-to-school shopping, holiday gifts, summer vacations, etc.)."
    )
    monthly_subcat_budget = {month+1: {} for month in range(12)}
    # 构造所有任务
    tasks = []
    for month in range(12):
        for category, month_budgets in monthly_allocation.items():
            budget = month_budgets[month]
            subcats = budget_to_walmart_main.get(category, [])
            if not subcats or budget <= 0:
                continue
            tasks.append((month, category, budget, subcats))
    async def process_one(month, category, budget, subcats):
        prompt = [
            # 1. 目的
            f"Your task is to allocate the total budget of {budget} CNY for month {month+1} in category '{category}' to the following subcategories. The sum must be exactly the total budget. Output strictly in JSON format.",
            # 2. 约束
            "You MUST ONLY allocate to the subcategories listed below. You MUST NOT invent, use, or output any subcategory not in the list. If you output any subcategory not in the list, your answer will be considered invalid.",
            "You must prioritize basic living needs (such as food, household essentials, health, personal care, etc.) if present. Do not assign zero to all subcategories. Do not allocate equally unless it is truly reasonable for this month.",
            "Respond strictly in JSON format. Do not output any other text.",
            # 3. 家庭画像
            f"Family profile: {family_profile}",
            # 4. US holidays/seasonality
            us_holidays,
            "Subcategories (choose ONLY from the following, do NOT use any other subcategory):"
        ]
        for subcat in subcats:
            prompt.append(f"- {subcat}")
        prompt.append("")
        prompt.append(f"Total budget: {budget}")
        prompt = "\n".join(prompt)
        try:
            content = await call_llm_chat_completion(
                prompt,
                system_content="You are a professional US-based financial planner."
            )
            # logger.info(f"budget_sum={budget},[LLM raw output][Month {month+1}][{category}]: {content}")
            subcat_allocation = parse_model_response(content)
            # --- 严格过滤，只保留 allowed_subcats ---
            filtered = {k: v for k, v in (subcat_allocation or {}).items() if k in subcats}
            missing = [s for s in subcats if s not in filtered]
            filtered_sum = sum(filtered.values())
            # 若有遗漏，均分剩余预算
            if missing:
                remain = round(budget - filtered_sum, 2)
                if remain > 0 and len(missing) > 0:
                    avg = round(remain / len(missing), 2)
                    for s in missing:
                        filtered[s] = avg
                # 再次归一化
                total = sum(filtered.values())
                if abs(total - budget) > 1e-2 and total > 0:
                    filtered = {k: round(v * budget / total, 2) for k, v in filtered.items()}
                    diff = round(budget - sum(filtered.values()), 2)
                    if abs(diff) > 0 and filtered:
                        first_key = next(iter(filtered))
                        filtered[first_key] = round(filtered[first_key] + diff, 2)
                # logger.warning(f"[LLM filtered][Month {month+1}][{category}] missing subcats: {missing}, fallback filled: {filtered}")
            # 若 LLM 返回结构完全不符或全为0，直接 fallback
            if not filtered or all(v == 0 for v in filtered.values()):
                equal_share = round(budget / len(subcats), 2) if subcats else 0
                filtered = {item: equal_share for item in subcats}
                # logger.error(f"[LLM fallback][Month {month+1}][{category}] fallback to equal allocation: {filtered}")
            return (month+1, category, filtered)
        except Exception as e:
            # logger.error(f"LLM monthly subcat allocation failed for month {month+1}, category {category}: {e}")
            # 兜底均分
            equal_share = round(budget / len(subcats), 2) if subcats else 0
            subcat_allocation = {item: equal_share for item in subcats}
            # print(f"Month {month+1} Category: {category} Subcategory budget allocation (fallback): {subcat_allocation}")
            return (month+1, category, subcat_allocation)
    # 串行处理，保证顺序和调试
    for (month, category, budget, subcats) in tasks:
        month_idx, category, subcat_allocation = await process_one(month, category, budget, subcats)
        monthly_subcat_budget[month_idx][category] = subcat_allocation
    return monthly_subcat_budget

async def llm_split_annual_budget_to_months(category: str, annual_budget: float, family_profile: str = None, year: int = None) -> list:
    """
    Use LLM to split a category's annual budget into 12 months, considering seasonality, US holidays, school terms, weather, and family background. The sum must be exactly the annual budget. Output strictly in JSON.
    """
    if not family_profile:
        raise ValueError("family_profile must be provided and not None!")
    us_holidays = (
        "Consider the following major US holidays and events: New Year's Day (Jan), Easter (Mar/Apr), Memorial Day (May), Independence Day (July), Labor Day (Sep), Halloween (Oct), Thanksgiving (Nov), Christmas (Dec), and school terms (school starts in Aug/Sep, summer break in Jun-Aug). Also consider seasonal changes (e.g., higher utility costs in winter/summer, back-to-school shopping, holiday gifts, summer vacations, etc.)."
    )
    prompt = [
        # 1. 目的
        f"Your task is to split the annual budget of {annual_budget} CNY for category '{category}' into 12 months. The sum must be exactly the annual budget. Output strictly as a JSON array of 12 numbers.",
        # 2. 约束
        "Do not allocate the entire budget to only one or two months. Do not allocate exactly the same amount to every month unless you provide a strong, realistic explanation. Extreme or unrealistic allocations will be considered invalid.",
        "Respond strictly as a JSON array of 12 numbers, each representing the budget for month 1 to 12. Do not output any other text.",
        # 3. 家庭画像
        f"Family profile: {family_profile}",
        # 4. US holidays/seasonality
        us_holidays
    ]
    if year:
        prompt.append(f"The year is {year}.")
    prompt = "\n".join(prompt)
    try:
        content = await call_llm_chat_completion(
            prompt,
            system_content="You are a professional US-based financial planner."
        )
        # Extract the first JSON array only
        import re
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            arr = json.loads(match.group(0))
            # Normalize to ensure sum equals annual_budget
            total = sum(arr)
            if abs(total - annual_budget) > 1e-2 and total > 0:
                arr = [round(v * annual_budget / total, 2) for v in arr]
                diff = round(annual_budget - sum(arr), 2)
                if abs(diff) > 0 and arr:
                    arr[0] = round(arr[0] + diff, 2)
            # --- 极端分配检测与修正 ---
            # 1. 全部均分
            if all(abs(x - arr[0]) < 1e-2 for x in arr):
                # fallback: 加入微小扰动
                import random
                arr = [round(arr[0] + random.uniform(-0.02, 0.02) * arr[0], 2) for _ in arr]
                total = sum(arr)
                arr = [round(v * annual_budget / total, 2) for v in arr]
                diff = round(annual_budget - sum(arr), 2)
                if abs(diff) > 0:
                    arr[0] += diff
            # 2. 只有一个月非零
            nonzero_months = [i for i, v in enumerate(arr) if abs(v) > 1e-2]
            if len(nonzero_months) <= 2:
                # fallback: 均匀分配
                avg = round(annual_budget / 12, 2)
                arr = [avg] * 12
                diff = round(annual_budget - sum(arr), 2)
                if abs(diff) > 0:
                    arr[0] += diff
            return arr
        else:
            raise ValueError("No JSON array found in model response")
    except Exception as e:
        # logger.error(f"LLM month split failed for category {category}: {e}")
        # Fallback: equal split
        avg = round(annual_budget / 12, 2)
        arr = [avg] * 12
        diff = round(annual_budget - sum(arr), 2)
        if abs(diff) > 0:
            arr[0] += diff
        return arr

async def llm_score_products(candidates, budget, subcat, family_profile=None, nutrition_needs=None):
    """
    Use LLM to select a combination of products and quantities from the candidates so that the total spending reaches 85-100% of the budget. Output strictly as a JSON array. All instructions must be in English.
    
    Args:
        candidates: 候选商品列表
        budget: 预算
        subcat: 小类名称
        family_profile: 家庭画像
        nutrition_needs: 营养需求 {'carbohydrate': 79.3, 'protein': 49.2, 'fat': 30.5, 'water': 16.4}
    """
    # import json
    # from llm_utils import call_llm_chat_completion  # 绝对导入，兼容脚本直接运行
    
    if not candidates:
        return []
    
    # 限制候选商品数量，避免prompt过长
    if len(candidates) > 12:
        # 按价格排序，选择价格合理的商品
        candidates = sorted(candidates, key=lambda x: abs(x['price'] - budget/5))[:12]
    
    # 根据小类类型给出具体指导
    category_guidance = {
        "food": "FOOD is consumed daily by families. For monthly shopping, consider bulk purchases, family packs, multiple varieties of staples (rice, pasta, snacks, beverages). Quantities should reflect monthly consumption for a family.",
        "household essentials": "HOUSEHOLD ESSENTIALS like cleaning supplies, paper products, toiletries are used regularly. Families buy these in bulk monthly. Consider larger quantities and multiple types.",
        "personal care": "PERSONAL CARE items are used daily by all family members. Consider multiple products for different needs and family members, with reasonable monthly quantities.",
        "health": "HEALTH products may include vitamins, supplements, first aid supplies. Families often stock up on these items for monthly/seasonal use.",
        "clothing": "CLOTHING purchases can include multiple items for different family members, seasons, or occasions. Consider sets or multiple pieces.",
        "home": "HOME items include furniture, appliances, décor, storage solutions. These can be higher-value items or multiple smaller home goods.",
        "toys": "TOYS can include multiple items for different ages, educational materials, games. Consider variety and quantities for family entertainment.",
        "electronics": "ELECTRONICS may include accessories, gadgets, or entertainment devices. Consider multiple items or higher-value single purchases."
    }
    
    guidance = category_guidance.get(subcat.lower(), "Consider typical family monthly consumption patterns for this category. Families often buy multiple items or larger quantities for monthly needs.")
    
    # 🔧 优化：根据营养状况动态调整预算要求
    if nutrition_needs:
        # 检查是否有严重过剩的营养素
        over_supplied = sum(1 for rate in nutrition_needs.values() if rate > 200)
        critical_deficiency = sum(1 for rate in nutrition_needs.values() if rate < 50)
        
        if over_supplied >= 2:
            # 营养严重过剩，降低预算要求
            min_spend = budget * 0.60
            max_spend = budget * 0.90
            budget_priority = "LOW"
        elif critical_deficiency >= 1:
            # 有严重缺失，正常预算
            min_spend = budget * 0.75
            max_spend = budget * 1.05
            budget_priority = "MEDIUM"
        else:
            # 营养基本均衡
            min_spend = budget * 0.70
            max_spend = budget * 1.00
            budget_priority = "MEDIUM"
    else:
        # 没有营养数据，使用默认值
        min_spend = budget * 0.70
        max_spend = budget * 1.00
        budget_priority = "MEDIUM"
    
    prompt = f"""
🎯 PRIMARY GOAL: ENSURE FAMILY'S BASIC LIVING NEEDS AND NUTRITIONAL BALANCE

Your task: Select products that prioritize nutritional balance for monthly family shopping.

Family profile: {family_profile or "General family with regular consumption needs"}
Subcategory: {subcat}
Budget: {budget} CNY
Suggested spending range: {min_spend:.2f} - {max_spend:.2f} CNY

⚠️ IMPORTANT PRIORITIES (in order):
1. 🥗 NUTRITIONAL BALANCE - Most important!
2. 🏠 BASIC LIVING NEEDS - Essential items
3. 💰 BUDGET EFFICIENCY - Reasonable spending (NOT mandatory to spend all)

📋 BUDGET GUIDELINES:
- It's ACCEPTABLE to spend 60-90% of budget if nutrition is balanced
- Better to underspend than create nutritional imbalance
- A second补充 phase will address any remaining nutritional gaps
- Focus on QUALITY and BALANCE, not quantity

{guidance}
"""
    
    # ========================================
    # 🔧 新增：营养引导（增强版：添加避免过剩逻辑）
    # ========================================
    if nutrition_needs and subcat.lower() in ['food', 'beverages', 'snacks', 'drinks']:
        prompt += "\n" + "="*60 + "\n"
        prompt += "🥗 NUTRITIONAL GUIDANCE (Last Month's Status):\n"
        prompt += "="*60 + "\n"
        
        # 分类营养素：不足 vs 充足 vs 过剩
        critical = []           # < 50%
        needs_improvement = []  # 50-90%
        sufficient = []         # 90-150%
        over_supplied = []      # 150-300%
        severely_over = []      # > 300%
        
        for nutrient, rate in nutrition_needs.items():
            if rate < 50:
                critical.append((nutrient, rate))
            elif rate < 90:
                needs_improvement.append((nutrient, rate))
            elif rate <= 150:
                sufficient.append((nutrient, rate))
            elif rate <= 300:
                over_supplied.append((nutrient, rate))
            else:
                severely_over.append((nutrient, rate))
        
        # 显示紧急缺失
        if critical:
            prompt += "\n🔴 CRITICAL DEFICIENCIES (< 50% - URGENT):\n"
            for nutrient, rate in sorted(critical, key=lambda x: x[1]):
                prompt += f"  • {nutrient.capitalize()}: {rate:.1f}% of monthly needs\n"
                
                # 给出具体建议
                if nutrient == 'water':
                    prompt += "    → PRIORITIZE: Fresh fruits (oranges, watermelon, grapes), vegetables (lettuce, cucumber), juices, soups, milk\n"
                elif nutrient == 'protein':
                    prompt += "    → PRIORITIZE: Meat, fish, eggs, beans, tofu, nuts, dairy products\n"
                elif nutrient == 'carbohydrate':
                    prompt += "    → PRIORITIZE: Rice, bread, pasta, cereals, potatoes, oats\n"
                elif nutrient == 'fat':
                    prompt += "    → PRIORITIZE: Cooking oil, nuts, avocado, fatty fish, seeds\n"
        
        # 显示需要改善
        if needs_improvement:
            prompt += "\n🟡 NEEDS IMPROVEMENT (50-90%):\n"
            for nutrient, rate in sorted(needs_improvement, key=lambda x: x[1]):
                prompt += f"  • {nutrient.capitalize()}: {rate:.1f}%\n"
        
        # 显示充足
        if sufficient:
            prompt += "\n✅ SUFFICIENT (90-150% - Good balance):\n"
            for nutrient, rate in sufficient:
                prompt += f"  • {nutrient.capitalize()}: {rate:.1f}% - Maintain current level\n"
        
        # 🔧 新增：显示过剩
        if over_supplied:
            prompt += "\n⚠️ OVER-SUPPLIED (150-300% - Already Excessive):\n"
            for nutrient, rate in over_supplied:
                prompt += f"  • {nutrient.capitalize()}: {rate:.1f}% - AVOID products high in this nutrient\n"
        
        if severely_over:
            prompt += "\n🔴 SEVERELY OVER-SUPPLIED (>300% - CRITICAL EXCESS):\n"
            for nutrient, rate in severely_over:
                prompt += f"  • {nutrient.capitalize()}: {rate:.1f}% - MUST AVOID products with this nutrient\n"
        
        # 🔧 新增：具体避免建议
        if severely_over or over_supplied:
            prompt += "\n❌ FOODS TO AVOID (Already have too much):\n"
            all_over = dict(over_supplied + severely_over)
            for nutrient in all_over.keys():
                if nutrient == 'carbohydrate':
                    prompt += "  • AVOID: Rice, bread, pasta, cereals, grains, potatoes, crackers\n"
                elif nutrient == 'protein':
                    prompt += "  • AVOID: Meat, fish, eggs, protein-rich products, protein bars\n"
                elif nutrient == 'fat':
                    prompt += "  • AVOID: Oils, butter, fatty meats, fried foods, high-fat dairy\n"
        
        # 总体策略（优化版：强调营养优先）
        prompt += "\n" + "="*60 + "\n"
        prompt += "📋 SELECTION STRATEGY (MANDATORY PRIORITY ORDER):\n"
        prompt += "="*60 + "\n"
        
        if critical:
            critical_names = ', '.join([n for n, r in critical])
            prompt += f"\n🔴 PRIORITY 1 (CRITICAL - MUST DO):\n"
            prompt += f"   Address deficiencies in: {critical_names}\n"
            prompt += f"   → You MUST select products rich in these nutrients\n"
            prompt += f"   → Allocate sufficient budget to meet at least 80% of these needs\n"
        
        if needs_improvement:
            improve_names = ', '.join([n for n, r in needs_improvement])
            prompt += f"\n🟡 PRIORITY 2 (IMPORTANT):\n"
            prompt += f"   Improve: {improve_names}\n"
            prompt += f"   → Select products that help reach 90%+ satisfaction\n"
        
        if severely_over or over_supplied:
            all_over_names = [n for n, r in (severely_over + over_supplied)]
            over_str = ', '.join(all_over_names)
            max_rate = max([r for n, r in (severely_over + over_supplied)])
            prompt += f"\n❌ PRIORITY 3 (MANDATORY CONSTRAINT):\n"
            prompt += f"   AVOID products high in: {over_str}\n"
            prompt += f"   → These nutrients are already at {max_rate:.0f}% (target: 100%)\n"
            prompt += f"   → DO NOT select products primarily providing these nutrients\n"
            prompt += f"   → If a product is high in over-supplied nutrients, SKIP IT\n"
        
        if sufficient:
            sufficient_names = ', '.join([n for n, r in sufficient])
            prompt += f"\n✅ PRIORITY 4 (MAINTAIN):\n"
            prompt += f"   Keep balanced: {sufficient_names}\n"
            prompt += f"   → These are well-balanced (90-150%), maintain current level\n"
        
        prompt += "\n" + "="*60 + "\n"
        prompt += "💡 CORE PRINCIPLES (READ CAREFULLY):\n"
        prompt += "="*60 + "\n"
        prompt += "1. ⭐ NUTRITIONAL BALANCE is THE TOP PRIORITY\n"
        prompt += "2. 💰 Budget is FLEXIBLE - OK to spend 60-90% if nutrition is balanced\n"
        prompt += "3. 🎯 Better to UNDERSPEND than create nutritional imbalance\n"
        prompt += "4. 🔄 A补充 phase will fill remaining gaps - don't over-buy now\n"
        prompt += "5. ✨ QUALITY and VARIETY over quantity\n"
        prompt += "="*60 + "\n\n"
    
    prompt += "Product candidates (MUST choose from these only, exact names, prices, and company IDs):\n"
    prompt += "⚠️ IMPORTANT: The same product name may be produced by different companies with different prices, quality, and attributes. You need to carefully compare and choose the best option.\n\n"
    for idx, c in enumerate(candidates, 1):
        owner_id = c.get('owner_id', 'N/A')
        prompt += f"{idx}. {c['name']} - {c['price']} CNY (Company: {owner_id})\n"
    
    prompt += f"""
MANDATORY REQUIREMENTS:
1. 🥗 PRIORITIZE NUTRITIONAL BALANCE over budget spending
2. Use exact product names and prices from the list above
3. Select quantities based on FAMILY NEEDS, not budget targets
4. This is MONTHLY family shopping - quantities should be realistic
5. It's OK to spend {min_spend:.2f}-{max_spend:.2f} CNY (flexible range)
6. Better to UNDERSPEND with good nutrition than OVERSPEND with imbalance

QUANTITY GUIDELINES (Based on Family Needs):
- Food items: 3-12 units (based on nutritional gaps, not price)
- Household essentials: 2-8 units (monthly household needs)  
- Personal care: 2-6 units (for family members)
- Other categories: 1-6 units (reasonable monthly purchases)

⚠️ IMPORTANT: If over-supplied nutrients exist, reduce quantities or skip products!

OUTPUT FORMAT (JSON array only, no explanations):
⚠️ CRITICAL: You MUST include "owner_id" (company ID) for each selected product!
[
  {{"name": "Product A", "price": 10.5, "quantity": 8, "owner_id": "company_123"}},
  {{"name": "Product B", "price": 25.0, "quantity": 3, "owner_id": "company_456"}}
]

⚠️ REMINDER: If multiple companies produce the same product name, compare their prices, quality, and attributes, then select the best option. Always include the owner_id in your response!

CHECK: Total = (10.5×8) + (25.0×3) = 84 + 75 = 159 CNY
TARGET: At least {min_spend:.2f} CNY

IMPORTANT: Count your total before responding. If total < {min_spend:.2f}, ADD MORE or INCREASE quantities!
"""
    
    # ========================================
    # 🔧 新增：打印提示词（用于调试）
    # ========================================
    logger.info(f"📝 商品选择提示词 (小类: {subcat}, 预算: {budget:.2f}):")
    # logger.info(f"{prompt[:1500]}...")  # 打印前1500字符
    if nutrition_needs:
        logger.info(f"🥗 营养需求数据: {nutrition_needs}")
    
    try:
        content = await call_llm_chat_completion(
            prompt,
            system_content=f"You are a family budget optimization assistant. Your PRIMARY GOAL is to reach at least 85% budget utilization ({min_spend:.2f} CNY minimum) while selecting realistic monthly quantities. Always verify your total spending reaches the minimum target."
        )
        
        # 尝试解析JSON
        result = json.loads(content)
        
        # 验证结果格式和计算总花费
        if isinstance(result, list):
            validated_result = []
            total_spent = 0
            
            for item in result:
                if isinstance(item, dict) and 'name' in item and 'price' in item and 'quantity' in item:
                    # 🆕 优先通过 (name, owner_id) 匹配，如果没有owner_id则通过name匹配
                    owner_id = item.get('owner_id', '')
                    if owner_id:
                        matching_candidate = next(
                            (c for c in candidates 
                             if c['name'] == item['name'] and c.get('owner_id', '') == owner_id), 
                            None
                        )
                    else:
                        # 如果没有owner_id，回退到只通过name匹配（兼容旧格式）
                        matching_candidate = next((c for c in candidates if c['name'] == item['name']), None)
                    
                    if matching_candidate:
                        quantity = max(1, min(20, int(item['quantity'])))  # 允许更大数量
                        price = float(item['price'])
                        # 🆕 从LLM返回中获取owner_id，如果没有则从候选商品中获取
                        result_owner_id = item.get('owner_id') or matching_candidate.get('owner_id', '')
                        validated_result.append({
                            'name': item['name'],
                            'price': price,
                            'quantity': quantity,
                            'owner_id': result_owner_id  # 🆕 添加owner_id
                        })
                        total_spent += price * quantity
            
            # 检查预算利用率
            utilization_rate = total_spent / budget if budget > 0 else 0
            # print(f"[LLM结果] 小类{subcat}: 预算{budget}, 花费{total_spent:.2f}, 利用率{utilization_rate:.1%}")
            
            # 如果预算利用率过低，启用增强策略
            if utilization_rate < 0.7 and validated_result:
                # print(f"[预算增强] 小类{subcat}利用率过低({utilization_rate:.1%})，增加数量...")
                # 选择最便宜的商品增加数量
                cheapest_item = min(validated_result, key=lambda x: x['price'])
                additional_quantity = int((min_spend - total_spent) / cheapest_item['price'])
                if additional_quantity > 0:
                    cheapest_item['quantity'] += additional_quantity
                    # print(f"[预算增强] 为{cheapest_item['name']}增加{additional_quantity}个数量")
            
            if validated_result:
                return validated_result
    
    except Exception as e:
        # print(f"[LLM商品评分异常] {e}, 使用增强备用方案")
        pass
    
    # 增强的备用方案：确保达到85%预算利用率
    result = []
    remaining_budget = budget
    target_budget = min_spend  # 目标至少85%
    candidates_sorted = sorted(candidates, key=lambda x: x['price'])
    
    # 第一轮：选择基础商品
    for candidate in candidates_sorted[:5]:  # 选择前5个最便宜的
        price = candidate['price']
        if price <= remaining_budget:
            quantity = max(1, min(8, int(target_budget / (price * len(candidates_sorted)))))
            if quantity > 0:
                result.append({
                    'name': candidate['name'],
                    'price': price,
                    'quantity': quantity
                })
                remaining_budget -= price * quantity
    
    # 第二轮：如果还没达到85%，增加数量
    current_total = sum(item['price'] * item['quantity'] for item in result)
    if current_total < min_spend and result:
        # 从最便宜的商品开始增加数量
        for item in sorted(result, key=lambda x: x['price']):
            needed = min_spend - current_total
            additional_qty = int(needed / item['price'])
            if additional_qty > 0:
                item['quantity'] += min(additional_qty, 10)  # 最多再增加10个
                current_total = sum(r['price'] * r['quantity'] for r in result)
                if current_total >= min_spend:
                    break
    
    return result
