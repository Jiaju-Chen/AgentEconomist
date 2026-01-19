"""
商品选择模块

本模块负责根据预算和家庭需求选择合适的商品：
- 商品检索与候选收集
- 批量LLM商品选择
- 回退处理与默认选择
- 响应解析与结果处理

作者：Agent Society Ecosim Team  
日期：2025-10-22
"""

import asyncio
import json
import logger
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import ray

from agentsociety_ecosim.consumer_modeling import llm_utils

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class ProductSelector:
    """商品选择器 - 负责根据预算选择合适的商品"""
    
    def __init__(
        self,
        product_dataframe: pd.DataFrame,
        product_market: Any = None,
        economic_center: Any = None,
        llm_semaphore: asyncio.Semaphore = None
    ):
        """
        初始化商品选择器
        
        Args:
            product_dataframe: 商品数据DataFrame
            product_market: 商品市场实例（用于向量搜索）
            economic_center: 经济中心实例
            llm_semaphore: LLM并发控制信号量
        """
        self.df = product_dataframe
        self.product_market = product_market
        self.economic_center = economic_center
        self.llm_semaphore = llm_semaphore or asyncio.Semaphore(50)
    
    def _get_real_time_price(self, product_id: str, product_name: str, owner_id: str = None) -> Optional[float]:
        """
        获取商品的实时价格（同步版本）
        
        Args:
            product_id: 商品ID
            product_name: 商品名称
            owner_id: 公司ID（可选）
        
        Returns:
            实时价格，如果查询失败则返回None
        """
        # 策略1: 如果有product_id和owner_id，直接从economic_center查询
        if product_id and owner_id and self.economic_center:
            try:
                price = ray.get(self.economic_center.query_price.remote(owner_id, product_id))
                if price and price > 0:
                    return price
            except Exception as e:
                logger.debug(f"通过economic_center查询价格失败 (product_id={product_id}, owner_id={owner_id}): {e}")
        
        # 策略2: 如果有product_id但没有owner_id，需要从CSV查找owner_id（这里暂时跳过，因为ProductSelector没有pro_firm_df）
        # 策略3: 如果只有product_name，尝试从ProductMarket查询
        if product_name and self.product_market:
            try:
                prices = ray.get(self.product_market.get_current_prices.remote(product_name))
                if prices and len(prices) > 0:
                    # 返回最低价格（如果有多个公司生产）
                    return min(prices)
            except Exception as e:
                logger.debug(f"通过ProductMarket查询价格失败 (product_name={product_name}): {e}")
        
        # 策略4: 如果都失败，返回None，调用者可以使用CSV中的价格作为fallback
        return None
    
    # ============================================================================
    # 商品检索与候选收集
    # ============================================================================
    
    def _search_products_sync(self, query: str, top_k: int, must_contain: str = None):
        """
        调用本地的 ProductMarket 实例进行商品检索（是Ray调用）
        如果未提供或不可用，则返回空列表
        """
        try:
            if self.product_market is None:
                logger.warning("🔍 [向量检索] product_market is None, 返回空列表")
                return []
            
            logger.debug(f"🔍 [向量检索] 开始检索: query='{query}', top_k={top_k}, must_contain='{must_contain}'")
            
            search_method = getattr(self.product_market, "search_products", None)
            if not callable(search_method):
                logger.warning("🔍 [向量检索] product_market 没有 search_products 方法")
                return []
            
            logger.debug(f"🔍 [向量检索] 调用 search_products.remote()...")
            result = ray.get(search_method.remote(
                query=query,
                top_k=top_k,
                must_contain=must_contain,
                economic_center=None
            ))
            logger.debug(f"🔍 [向量检索] 成功返回 {len(result) if result else 0} 个商品")
            return result
            
        except AttributeError as e:
            logger.error(f"🔍 [向量检索] AttributeError (可能缺少 .remote 方法): {e}")
            logger.error(f"🔍 [向量检索] product_market 类型: {type(self.product_market)}")
            return []
        except Exception as e:
            logger.error(f"🔍 [向量检索] 失败: {type(e).__name__}: {e}", exc_info=True)
            return []
    
    def retrieve_candidates(self, query_text, tokenizer, model, subcat, topn=50):
        """
        语义检索商品，并过滤出属于当前小类的商品
        使用ProductMarket的search_products方法进行检索
        """
        try:
            logger.debug(f"📋 [候选收集] retrieve_candidates: query='{query_text}', subcat='{subcat}', topn={topn}")
            
            # 使用本地 ProductMarket 的同步方法
            products = self._search_products_sync(
                query=query_text,
                top_k=topn,
                must_contain=subcat,
            )
            
            if not products:
                logger.warning(f"📋 [候选收集] _search_products_sync 返回空列表")
                return []
            
            logger.debug(f"📋 [候选收集] 原始检索结果: {len(products)} 个商品")
            
            # 转换为原有格式，并过滤无效价格
            candidates = []
            filtered_count = 0
            
            for i, product in enumerate(products):
                # 检查价格是否有效
                price = getattr(product, 'price', None)
                if price is None or pd.isna(price) or price <= 0:
                    filtered_count += 1
                    logger.debug(f"📋 [候选收集] 过滤商品 {i}: 价格无效 ({price})")
                    continue
                    
                candidates.append({
                    'name': getattr(product, 'name', f'Unknown_{i}'),
                    'price': float(price),  # 确保是float类型
                    'classification': getattr(product, 'classification', ''),
                    'brand': getattr(product, 'brand', ''),
                    'description': getattr(product, 'description', ''),
                    'product_id': getattr(product, 'product_id', ''),
                    'owner_id': getattr(product, 'owner_id', '')  # 🆕 添加公司ID
                })
            
            # logger.info(f"📋 [候选收集] 小类 '{subcat}' 检索完成: 原始{len(products)}个, 过滤{filtered_count}个, 最终{len(candidates)}个候选商品")
            return candidates
            
        except Exception as e:
            logger.error(f"📋 [候选收集] 异常: {type(e).__name__}: {e}", exc_info=True)
            return []
    
    def _collect_candidates_for_subcategory(
        self,
        category: str,
        subcategory: str,
        budget: float,
        tokenizer=None,
        model=None
    ) -> List[Dict]:
        """
        为单个小类收集候选商品（包含3层回退机制）
        
        Args:
            category: 大类名称
            subcategory: 小类名称
            budget: 预算
            tokenizer: 分词器（可选）
            model: 模型（可选）
            
        Returns:
            List[Dict]: 候选商品列表
        """
        # logger.info(f"🎯 [候选收集] 开始为小类 '{subcategory}' (大类: '{category}') 收集候选商品, 预算: ${budget:.2f}")
        candidates = []
        
        # 方案1: 语义检索（如果有ProductMarket）
        if self.product_market is not None:
            # logger.info(f"🎯 [候选收集] 方案1: 尝试语义检索 (product_market 可用)")
            try:
                query_text = f"{category} {subcategory}"
                logger.debug(f"🎯 [候选收集] 查询文本: '{query_text}'")
                
                candidates = self.retrieve_candidates(
                    query_text,
                    tokenizer,
                    model,
                    subcategory,
                    topn=50
                )
                
                if len(candidates) >= 5:
                    # logger.info(f"✅ [语义检索成功] 小类 '{subcategory}' 找到 {len(candidates)} 个候选商品")
                    return candidates
                else:
                    logger.warning(f"⚠️ [语义检索不足] 小类 '{subcategory}' 只找到 {len(candidates)} 个候选商品 (< 5)，将使用方案2")
            except Exception as e:
                logger.error(f"❌ [语义检索失败] 小类 '{subcategory}': {type(e).__name__}: {e}", exc_info=True)
        else:
            logger.warning(f"⚠️ [候选收集] product_market is None，跳过语义检索")
        
        # 方案2: 直接从商品库筛选
        if len(candidates) < 5:
            # logger.info(f"🎯 [候选收集] 方案2: 从商品库 (CSV) 筛选 (当前候选数: {len(candidates)})")
            try:
                # 精确匹配level1
                logger.debug(f"🎯 [商品库筛选] 在 CSV 中查找 level1 == '{subcategory}'")
                
                if 'level1' not in self.df.columns:
                    logger.error(f"❌ [商品库筛选] CSV 中没有 'level1' 列，可用列: {list(self.df.columns)}")
                    subcat_products = pd.DataFrame()
                else:
                    subcat_products = self.df[
                        self.df['level1'].str.lower() == subcategory.strip().lower()
                    ]
                    logger.debug(f"🎯 [商品库筛选] level1 匹配结果: {len(subcat_products)} 个商品")
                
                # 价格过滤：不超过预算的120%，不低于预算的1%
                price_min = budget * 0.01
                price_max = budget * 1.2
                before_filter = len(subcat_products)
                
                if 'List Price' in subcat_products.columns:
                    subcat_products = subcat_products[subcat_products['List Price'] <= price_max]
                    subcat_products = subcat_products[subcat_products['List Price'] >= price_min]
                    logger.debug(f"🎯 [商品库筛选] 价格过滤 (${price_min:.2f} ~ ${price_max:.2f}): {before_filter} -> {len(subcat_products)} 个商品")
                
                # 转换为候选格式
                added_count = 0
                existing_names = {c['name'] for c in candidates}
                
                for _, item in subcat_products.head(30).iterrows():
                    product_name = item["Product Name"]
                    if product_name not in existing_names:
                        product_id = item.get('Uniq Id', '')
                        owner_id = item.get('owner_id', '') or item.get('company_id', '')
                        # 🆕 查询实时价格
                        real_time_price = self._get_real_time_price(
                            product_id=product_id,
                            product_name=product_name,
                            owner_id=owner_id
                        )
                        # 如果查询失败，使用CSV价格作为fallback
                        price = real_time_price if real_time_price is not None else float(item["List Price"])
                        
                        candidates.append({
                            'name': product_name,
                            'price': price,  # ✅ 使用实时价格
                            'classification': f"{item.get('level1', '')}/{item.get('level2', '')}",
                            'brand': item.get('Brand', ''),
                            'description': item.get('description', ''),
                            'product_id': product_id,
                            'owner_id': owner_id  # 🆕 添加公司ID
                        })
                        added_count += 1
                
                logger.info(f"✅ [商品库筛选成功] 小类 '{subcategory}' 从 CSV 添加 {added_count} 个商品，总计 {len(candidates)} 个候选商品")
            except Exception as e:
                logger.error(f"❌ [商品库筛选异常] 小类 '{subcategory}': {type(e).__name__}: {e}", exc_info=True)
        
        # 方案3: 最后的备用方案 - 从同一大类下的其他小类借用商品
        if len(candidates) < 5:
            logger.info(f"🎯 [候选收集] 方案3: 从同大类其他小类借用 (当前候选数: {len(candidates)})")
            try:
                from agentsociety_ecosim.consumer_modeling.budget_allocation.config import BudgetConfig
                same_category_subcats = BudgetConfig.BUDGET_TO_WALMART_MAIN.get(category, [])
                
                logger.debug(f"🎯 [同类借用] 大类 '{category}' 包含小类: {same_category_subcats}")
                
                borrowed_count = 0
                for other_subcat in same_category_subcats:
                    if other_subcat != subcategory and len(candidates) < 15:
                        logger.debug(f"🎯 [同类借用] 尝试从小类 '{other_subcat}' 借用商品")
                        
                        other_products = self.df[
                            self.df['level1'].str.lower() == other_subcat.strip().lower()
                        ]
                        
                        before_price_filter = len(other_products)
                        other_products = other_products[other_products['List Price'] <= budget * 1.2]
                        other_products = other_products[other_products['List Price'] >= budget * 0.01]
                        
                        logger.debug(f"🎯 [同类借用] 小类 '{other_subcat}': 原始{before_price_filter}个, 价格过滤后{len(other_products)}个")
                        
                        existing_names = {c['name'] for c in candidates}
                        added_from_this = 0
                        
                        for _, item in other_products.head(8).iterrows():
                            if item["Product Name"] not in existing_names:
                                product_id = item.get('Uniq Id', '')
                                owner_id = item.get('owner_id', '') or item.get('company_id', '')
                                # 🆕 查询实时价格
                                real_time_price = self._get_real_time_price(
                                    product_id=product_id,
                                    product_name=item["Product Name"],
                                    owner_id=owner_id
                                )
                                # 如果查询失败，使用CSV价格作为fallback
                                price = real_time_price if real_time_price is not None else float(item["List Price"])
                                
                                candidates.append({
                                    'name': item["Product Name"],
                                    'price': price,  # ✅ 使用实时价格
                                    'classification': f"{item.get('level1', '')}/{item.get('level2', '')}",
                                    'brand': item.get('Brand', ''),
                                    'description': item.get('description', ''),
                                    'product_id': product_id,
                                    'owner_id': owner_id  # 🆕 添加公司ID
                                })
                                borrowed_count += 1
                                added_from_this += 1
                                if len(candidates) >= 15:
                                    break
                        
                        if added_from_this > 0:
                            logger.debug(f"🎯 [同类借用] 从 '{other_subcat}' 借用了 {added_from_this} 个商品")
                
                logger.info(f"✅ [同类借用完成] 小类 '{subcategory}' 借用了 {borrowed_count} 个商品，最终 {len(candidates)} 个候选商品")
            except Exception as e:
                logger.error(f"❌ [同类借用失败] 小类 '{subcategory}': {type(e).__name__}: {e}", exc_info=True)
        
        # 最终总结
        if len(candidates) == 0:
            logger.error(f"❌ [候选收集失败] 小类 '{subcategory}' 所有方案都未找到候选商品！")
        else:
            logger.info(f"🎉 [候选收集完成] 小类 '{subcategory}' 最终收集到 {len(candidates)} 个候选商品")
        
        return candidates
    
    # ============================================================================
    # 批量LLM商品选择
    # ============================================================================
    
    async def _batch_select_products_for_category(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        family_profile: str,
        current_month: int,
        topn: int = 20,
        family_id: str = None
    ) -> Dict[str, List[Dict]]:
        """
        【方案A：分层批量】为单个大类的所有小类批量选择商品
        
        1. 一次性收集该大类下所有小类的候选商品
        2. 构建包含所有小类的批量prompt
        3. 一次LLM调用为所有小类选择商品
        4. 解析响应并分配到各小类
        
        优点：大大减少LLM调用次数（每个大类只调用1次）
        缺点：单次prompt较长，可能存在token限制
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[批量选择] 开始为大类 {category} 批量选择商品")
        logger.info(f"  - 小类数量: {len(subcategory_budgets)}")
        logger.info(f"  - 总预算: ${sum(subcategory_budgets.values()):.2f}")
        
        # Step 1: 收集所有小类的候选商品
        all_candidates = {}
        for subcategory, budget in subcategory_budgets.items():
            if budget <= 0:
                all_candidates[subcategory] = []
                continue
            
            candidates = self._collect_candidates_for_subcategory(
                category,
                subcategory,
                budget,
                tokenizer=None,
                model=None
            )
            all_candidates[subcategory] = candidates[:topn]
        
        # Step 2: 如果所有小类都没有候选商品，直接返回空结果
        total_candidates = sum(len(c) for c in all_candidates.values())
        if total_candidates == 0:
            logger.warning(f"[批量选择] 大类 {category} 所有小类都没有候选商品，跳过LLM选择")
            return {subcat: [] for subcat in subcategory_budgets.keys()}
        
        # Step 3: 构建批量prompt并调用LLM
        try:
            prompt = self._build_batch_product_selection_prompt(
                category,
                subcategory_budgets,
                all_candidates,
                family_profile,
                current_month
            )
            
            # ========================================
            # 🔧 打印：完整的商品选择提示词
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤3: 商品选择 - LLM提示词】大类: {category}\n{'='*80}")
            # logger.info(f"{prompt}")
            # logger.info(f"{'='*80}\n")
            
            async with self.llm_semaphore:
                content = await llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a smart shopping assistant. Always respond with valid JSON."
                )
            
            # ========================================
            # 🔧 打印：完整的LLM响应
            # ========================================
            # logger.info(f"\n{'='*80}\n【步骤3: 商品选择 - LLM响应】大类: {category}\n{'='*80}")
            # logger.info(f"{content}")
            # logger.info(f"{'='*80}\n")
            
            # Step 4: 解析响应
            batch_results = self._parse_batch_response_flexible(content)
            
            # Step 5: 处理结果
            selected_products = self._process_batch_product_results(
                category,
                subcategory_budgets,
                all_candidates,
                batch_results
            )
            
            # 统计
            total_selected = sum(len(products) for products in selected_products.values())
            logger.info(f"[批量选择完成] 大类 {category} 共选择 {total_selected} 个商品")
            logger.info(f"{'='*80}\n")
            
            return selected_products
            
        except Exception as e:
            logger.error(f"[批量选择失败] 大类 {category}: {e}")
            # 回退到生成默认选择
            return {
                subcat: self._generate_fallback_selection(
                    subcat,
                    budget,
                    all_candidates.get(subcat, [])
                )
                for subcat, budget in subcategory_budgets.items()
            }
    
    # ============================================================================
    # 提示构建与响应解析
    # ============================================================================
    
    def _build_batch_product_selection_prompt(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        all_candidates: Dict[str, List[Dict]],
        family_profile: str,
        current_month: int
    ) -> str:
        """
        构建批量商品选择的LLM提示词
        
        包含所有小类的预算、候选商品和选择要求
        """
        # 季节提示
        season = "Winter" if current_month in [12, 1, 2] else \
                "Spring" if current_month in [3, 4, 5] else \
                "Summer" if current_month in [6, 7, 8] else "Fall"
        
        prompt = f"""
You are a smart shopping assistant helping a family select products within their budget.

**Family Profile:**
{family_profile}

**Current Season:** {season} (Month {current_month})

**Category:** {category}
**Total Budget:** ${sum(subcategory_budgets.values()):.2f}

**Task:** For each subcategory below, select products that:
1. **Stay within the subcategory budget** (required)
2. Match the family's needs and season
3. Provide good value for money
4. Use the EXACT product names from the candidate list

---

"""
        
        # 为每个小类添加详细信息
        for subcategory, budget in subcategory_budgets.items():
            candidates = all_candidates.get(subcategory, [])
            
            prompt += f"\n### Subcategory: {subcategory}\n"
            prompt += f"**Budget:** ${budget:.2f}\n"
            
            if not candidates:
                prompt += "**Note:** No candidates available for this subcategory\n"
                continue
            
            prompt += f"**Candidates ({len(candidates)} products):**\n"
            for i, product in enumerate(candidates[:15], 1):  # 限制每个小类最多15个候选
                prompt += f"  {i}. \"{product['name']}\" - ${product['price']:.2f}\n"
        
        # 输出格式说明
        prompt += """

---

**Output Format (JSON only, no explanations):**

```json
{
  "subcategory_name_1": [
    {"name": "exact_product_name", "price": 12.99, "quantity": 2, "total_spent": 25.98},
    ...
  ],
  "subcategory_name_2": [
    ...
  ]
}
```

**Important Rules:**
- Use EXACT product names from the candidate list
- Ensure total_spent ≤ budget for each subcategory
- If no good options, return empty array []
- Respond with ONLY the JSON, no additional text
"""
        
        return prompt
    
    def _parse_batch_response_flexible(self, content: str) -> Dict:
        """
        灵活解析批量响应（支持多种JSON格式）
        
        尝试多种解析策略：
        1. 直接JSON解析
        2. 提取code block中的JSON
        3. 正则提取JSON对象
        4. 文本解析提取商品信息
        """
        if not content:
            return {}
        
        # 策略1: 直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 策略2: 提取code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 策略3: 查找最大的JSON对象
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
        matches = json_pattern.findall(content)
        if matches:
            # 尝试最长的匹配
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # 策略4: 文本提取
        logger.warning("[响应解析] 无法解析为JSON，尝试文本提取")
        return self._extract_from_text(content)
    
    def _extract_from_text(self, content: str) -> Dict:
        """
        从纯文本中提取商品信息（最后的回退方案）
        
        查找类似 "product_name - $price" 的模式
        """
        results = {}
        
        # 查找小类名称和商品
        subcategory_pattern = re.compile(
            r'(?:Subcategory|Category):\s*([^\n:]+)',
            re.IGNORECASE
        )
        product_pattern = re.compile(
            r'["\']([^"\']+)["\']\s*-?\s*\$?(\d+\.?\d*)',
            re.IGNORECASE
        )
        
        subcategories = subcategory_pattern.findall(content)
        for subcat in subcategories:
            subcat = subcat.strip()
            results[subcat] = []
        
        # 提取所有商品
        products = product_pattern.findall(content)
        if products and not subcategories:
            # 如果没有找到小类，创建一个默认小类
            results["unspecified"] = [
                {"name": name, "price": float(price), "quantity": 1, "total_spent": float(price)}
                for name, price in products
            ]
        
        return results
    
    def _process_batch_product_results(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        all_candidates: Dict[str, List[Dict]],
        batch_results: Dict
    ) -> Dict[str, List[Dict]]:
        """
        处理批量LLM返回的结果，进行验证和修正
        
        包括：
        - 商品名称匹配
        - 预算验证
        - 缺失字段补全
        - 回退处理
        """
        processed_results = {}
        
        for subcategory, budget in subcategory_budgets.items():
            candidates = all_candidates.get(subcategory, [])
            llm_selected = batch_results.get(subcategory, [])
            
            if not llm_selected or not isinstance(llm_selected, list):
                # LLM没有返回结果，使用回退
                processed_results[subcategory] = self._generate_fallback_selection(
                    subcategory,
                    budget,
                    candidates,
                    None
                )
                continue
            
            # 验证和修正LLM选择的商品
            validated_products = []
            total_spent = 0
            
            for item in llm_selected:
                if not isinstance(item, dict):
                    continue
                
                # 提取商品名称
                product_name = item.get('name', '').strip()
                if not product_name:
                    continue
                
                # 在候选中查找匹配的商品（模糊匹配）
                matched_product = None
                for candidate in candidates:
                    if candidate['name'].lower() == product_name.lower():
                        matched_product = candidate
                        break
                
                if not matched_product:
                    # 尝试部分匹配
                    for candidate in candidates:
                        if product_name.lower() in candidate['name'].lower() or \
                           candidate['name'].lower() in product_name.lower():
                            matched_product = candidate
                            break
                
                if matched_product:
                    # 使用候选商品的价格（更准确）
                    quantity = int(item.get('quantity', 1))
                    if quantity < 1:
                        quantity = 1
                    
                    item_cost = matched_product['price'] * quantity
                    
                    # 检查预算
                    if total_spent + item_cost <= budget * 1.05:  # 允许5%溢出
                        validated_products.append({
                            'name': matched_product['name'],
                            'price': matched_product['price'],
                            'quantity': quantity,
                            'total_spent': round(item_cost, 2),
                            'classification': matched_product.get('classification', ''),
                            'brand': matched_product.get('brand', ''),
                            'product_id': matched_product.get('product_id', '')
                        })
                        total_spent += item_cost
            
            # 如果验证后没有商品，使用回退
            if not validated_products:
                processed_results[subcategory] = self._generate_fallback_selection(
                    subcategory,
                    budget,
                    candidates,
                    None
                )
            else:
                processed_results[subcategory] = validated_products
        
        return processed_results
    
    # ============================================================================
    # 回退处理与默认选择
    # ============================================================================
    
    def _generate_fallback_selection(
        self,
        subcategory: str,
        budget: float,
        candidates: List[Dict],
        llm_selected: List[Dict] = None
    ) -> List[Dict]:
        """
        生成回退商品选择（基于规则的默认算法）
        
        策略：
        1. 优先选择价格接近预算60%-80%的商品
        2. 如果预算很小，选择1-2件便宜商品
        3. 如果预算大，选择多件不同价位的商品
        """
        if not candidates or budget <= 0:
            return []
        
        selected_products = []
        total_spent = 0
        
        # 按价格排序
        sorted_candidates = sorted(candidates, key=lambda x: x['price'])
        
        # 策略1: 小预算（< $20）- 选1-2件便宜商品
        if budget < 20:
            for product in sorted_candidates:
                if product['price'] <= budget and product['price'] >= budget * 0.3:
                    selected_products.append({
                        'name': product['name'],
                        'price': product['price'],
                        'quantity': 1,
                        'total_spent': product['price'],
                        'classification': product.get('classification', ''),
                        'brand': product.get('brand', ''),
                        'product_id': product.get('product_id', '')
                    })
                    break
        
        # 策略2: 中等预算（$20-$100）- 选2-3件商品
        elif budget < 100:
            target_price = budget * 0.6
            for product in sorted_candidates:
                if total_spent + product['price'] <= budget:
                    if abs(product['price'] - target_price) / budget < 0.3:
                        selected_products.append({
                            'name': product['name'],
                            'price': product['price'],
                            'quantity': 1,
                            'total_spent': product['price'],
                            'classification': product.get('classification', ''),
                            'brand': product.get('brand', ''),
                            'product_id': product.get('product_id', '')
                        })
                        total_spent += product['price']
                        
                        if len(selected_products) >= 2:
                            break
        
        # 策略3: 大预算（>= $100）- 选3-5件商品
        else:
            # 分成不同价格段
            low_price = budget * 0.1
            mid_price = budget * 0.3
            high_price = budget * 0.5
            
            for product in sorted_candidates:
                if total_spent + product['price'] <= budget:
                    price = product['price']
                    # 尝试均衡选择不同价位
                    if (price <= low_price or 
                        (low_price < price <= mid_price and len([p for p in selected_products if p['price'] <= mid_price]) < 2) or
                        (price > mid_price and len([p for p in selected_products if p['price'] > mid_price]) < 2)):
                        
                        selected_products.append({
                            'name': product['name'],
                            'price': product['price'],
                            'quantity': 1,
                            'total_spent': product['price'],
                            'classification': product.get('classification', ''),
                            'brand': product.get('brand', ''),
                            'product_id': product.get('product_id', '')
                        })
                        total_spent += product['price']
                        
                        if len(selected_products) >= 4:
                            break
        
        # 如果还是没有选中任何商品，至少选一个最便宜的
        if not selected_products and sorted_candidates:
            cheapest = sorted_candidates[0]
            if cheapest['price'] <= budget:
                selected_products.append({
                    'name': cheapest['name'],
                    'price': cheapest['price'],
                    'quantity': 1,
                    'total_spent': cheapest['price'],
                    'classification': cheapest.get('classification', ''),
                    'brand': cheapest.get('brand', ''),
                    'product_id': cheapest.get('product_id', '')
                })
        
        return selected_products
    
    # ============================================================================
    # 小批量处理（针对大类商品数量过多的情况）
    # ============================================================================
    
    async def _mini_batch_processing(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        family_profile: str,
        current_month: int,
        batch_size: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        【方案B：小批量处理】将小类分成更小的批次处理
        
        用于大类商品过多、单次LLM prompt过长的情况
        """
        logger.info(f"[小批量处理] 大类 {category} 分批处理，每批 {batch_size} 个小类")
        
        # 将小类分组
        subcategory_items = list(subcategory_budgets.items())
        batches = [
            dict(subcategory_items[i:i+batch_size])
            for i in range(0, len(subcategory_items), batch_size)
        ]
        
        # 并发处理每个批次
        tasks = [
            self._batch_select_products_for_category(
                category,
                batch,
                family_profile,
                current_month
            )
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        final_results = {}
        for result in batch_results:
            if isinstance(result, dict):
                final_results.update(result)
            elif isinstance(result, Exception):
                logger.error(f"[小批量处理] 批次处理失败: {result}")
        
        # 补充缺失的小类
        for subcategory, budget in subcategory_budgets.items():
            if subcategory not in final_results:
                final_results[subcategory] = []
        
        return final_results
    
    def _build_mini_batch_prompt(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        all_candidates: Dict[str, List[Dict]],
        family_profile: str,
        current_month: int
    ) -> str:
        """
        构建小批量处理的提示词（与批量提示类似，但更简洁）
        """
        return self._build_batch_product_selection_prompt(
            category,
            subcategory_budgets,
            all_candidates,
            family_profile,
            current_month
        )
    
    # ============================================================================
    # 回退到单个商品选择（最终回退方案）
    # ============================================================================
    
    async def _fallback_individual_product_selection(
        self,
        category: str,
        subcategory_budgets: Dict[str, float],
        family_profile: str,
        current_month: int
    ) -> Dict[str, List[Dict]]:
        """
        【方案C：单个小类处理】完全回退到为每个小类单独选择商品
        
        最保守的方案，但LLM调用次数最多
        """
        logger.warning(f"[单个处理回退] 大类 {category} 使用单个小类处理模式")
        
        results = {}
        
        for subcategory, budget in subcategory_budgets.items():
            if budget <= 0:
                results[subcategory] = []
                continue
            
            # 收集候选商品
            candidates = self._collect_candidates_for_subcategory(
                category,
                subcategory,
                budget
            )
            
            # 直接使用回退选择（不调用LLM）
            results[subcategory] = self._generate_fallback_selection(
                subcategory,
                budget,
                candidates
            )
        
        return results

