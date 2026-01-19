import os
import json
import logger
import datetime
import time
import random
import sys
import re
from typing import Dict, Any, List, Union, Optional
from collections import OrderedDict
import concurrent.futures
import pandas as pd
import asyncio
import ray
# 尝试导入科学计算库，如果失败则使用备用方法
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 项目相关导入
from agentsociety_ecosim.consumer_modeling import llm_utils
from agentsociety_ecosim.consumer_modeling import QAIDS_model  # 导入QAIDS模型
from agentsociety_ecosim.consumer_modeling.family_data import get_family_consumption_and_profile_by_id, get_latest_expenditures_by_family_id
from agentsociety_ecosim.center.assetmarket import ProductMarket  # 导入ProductMarket类
from agentsociety_ecosim.center.ecocenter import EconomicCenter  # 导入EconomicCenter类
from agentsociety_ecosim.utils.data_loader import load_processed_products
from agentsociety_ecosim.consumer_modeling.family_attribute_manager import FamilyAttributeManager
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)
# 🔧 导入重构后的模块化组件
from agentsociety_ecosim.consumer_modeling.budget_allocation import (
    BudgetConfig,
    LegacyDataConverter,
    HistoryManager,
    MonthlyBudgetCalculator,
    CategoryAllocator,
    SubcategoryAllocator,
    ProductSelector
)


# logger = setup_global_logger(name=__name__, level=logger.INFO)
# logger = setup_global_logger(name=__name__, level=logging.INFO)



# logger = setup_global_logger(name=__name__, level=logger.INFO)
# logger = setup_global_logger(name=__name__, level=logging.INFO)


class BudgetAllocator:
    """
    A class to allocate family budget based on past expenditures using a large language model.
    
    配置常量已迁移到 BudgetConfig 模块，通过 BudgetConfig.xxx 访问
    """
    
    # 类级别的全局LLM并发控制信号量（所有BudgetAllocator实例共享）
    _global_llm_semaphore = None
    _semaphore_limit = 50  # 默认值，可通过 set_global_llm_limit 修改
    
    # 🔧 初始化模块化组件实例（类级别）
    _legacy_converter = LegacyDataConverter()
    _history_manager = HistoryManager()
    _budget_calculator = None  # 延迟初始化，需要llm_utils和llm_semaphore
    _category_allocator = None  # 延迟初始化，需要llm_semaphore
    _subcategory_allocator = None  # 延迟初始化，需要llm_semaphore
    _product_selector = None  # 延迟初始化，需要product_dataframe等
    
    @classmethod
    def set_global_llm_limit(cls, limit: int):
        """设置全局LLM并发限制"""
        cls._semaphore_limit = limit
        cls._global_llm_semaphore = asyncio.Semaphore(limit)
        logger.info(f"全局LLM并发限制已设置为: {limit}")
    
    @classmethod
    def get_global_llm_semaphore(cls):
        """获取全局LLM信号量，如果未初始化则创建"""
        if cls._global_llm_semaphore is None:
            cls._global_llm_semaphore = asyncio.Semaphore(cls._semaphore_limit)
        return cls._global_llm_semaphore

    def __init__(self,
                 model_name: str = "USD-guiji/deepseek-v3",
                 temperature: float = 0.1,
                 api_key: str = "sk-JeCvnVJdFk1SbiUc8Klw6t0wRn4KjT4G9DD7V1zjT9n26NIw",
                 llm_option: str = "custom",
                 product_market: ProductMarket = None,
                 economic_center: EconomicCenter = None,
                 attribute_manager = None,
                 product_df = None):
        """
        初始化 BudgetAllocator。
        :param model_name: LLM 模型名称，如 "gpt-4" 或 "gpt-3.5-turbo"
        :param temperature: LLM 调用时的温度参数
        :param api_key: OpenAI API Key，如果为 None，则从环境变量读取
        :param llm_option: LLM调用选项，可选 "deepseek" 或 "custom"
        :param attribute_manager: FamilyAttributeSystem 实例，用于获取营养需求数据
        :param product_df: 商品DataFrame，如果为None则从文件读取
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.llm_option = llm_option
        
        # 🔧 优化：优先使用传入的商品DataFrame，确保与测试配置一致
        if product_df is not None:
            self.df = product_df
            logger.info(f"✅ BudgetAllocator 使用外部传入的商品DataFrame ({len(self.df)} 个商品)")
        else:
            self.df = pd.read_csv('data/products.csv')
            # 过滤掉价格为0或负数的商品
            if 'List Price' in self.df.columns:
                self.df = self.df[self.df['List Price'] > 0].copy()
            elif 'price' in self.df.columns:
                self.df = self.df[self.df['price'] > 0].copy()
            logger.info(f"✅ BudgetAllocator 从文件读取商品DataFrame ({len(self.df)} 个商品)")
        
        self.pro_firm_df = pd.read_csv('data/company_product_map_rescaled.csv')
        # if not ray.is_initialized():
        #     ray.init(ignore_reinit_error=True)
        
        # 初始化ProductMarket来处理向量搜索

        self.product_market = product_market
        self.economic_center = economic_center
        
        # 初始化家庭属性管理器（从外部传入，通常是 Household 的 attribute_system）
        self.attribute_manager = attribute_manager
        if self.attribute_manager:
            logger.info(f"✅ BudgetAllocator 已接收 attribute_manager: {type(self.attribute_manager).__name__}")
        else:
            logger.warning(f"⚠️ BudgetAllocator 初始化时 attribute_manager 为 None")
        
        # 从BudgetConfig加载无二级子类的大类配置
        self.no_subcat_categories = BudgetConfig.NO_SUBCAT_CATEGORIES
        
        # 🔧 初始化分配器实例（需要llm_semaphore）
        if BudgetAllocator._category_allocator is None:
            BudgetAllocator._category_allocator = CategoryAllocator(
                category_keys=BudgetConfig.CATEGORY_KEYS,
                legacy_category_keys=BudgetConfig.LEGACY_CATEGORY_KEYS,
                category_names_zh=BudgetConfig.CATEGORY_NAMES_ZH,
                attribute_to_category_mapping=BudgetConfig.ATTRIBUTE_TO_CATEGORY_MAPPING,
                llm_semaphore=BudgetAllocator.get_global_llm_semaphore()
            )
        
        if BudgetAllocator._subcategory_allocator is None:
            BudgetAllocator._subcategory_allocator = SubcategoryAllocator(
                budget_to_walmart_main=BudgetConfig.BUDGET_TO_WALMART_MAIN,
                category_keys=BudgetConfig.CATEGORY_KEYS,
                llm_semaphore=BudgetAllocator.get_global_llm_semaphore()
            )
        
        # 🔧 初始化商品选择器实例（需要dataframe等）
        if BudgetAllocator._product_selector is None:
            BudgetAllocator._product_selector = ProductSelector(
                product_dataframe=self.df,
                product_market=self.product_market,
                economic_center=self.economic_center,
                llm_semaphore=BudgetAllocator.get_global_llm_semaphore()
            )
        
        # 🔧 初始化月度预算计算器实例（需要llm_utils和llm_semaphore）
        if BudgetAllocator._budget_calculator is None:
            BudgetAllocator._budget_calculator = MonthlyBudgetCalculator(
                llm_utils=llm_utils,
                llm_semaphore=BudgetAllocator.get_global_llm_semaphore()
            )

    def _search_products_sync(self, query: str, top_k: int, must_contain: str = None):
        """
        调用本地的 ProductMarket 实例进行商品检索（是Ray调用）
        
        🔧 委托给 ProductSelector 处理
        """
        return BudgetAllocator._product_selector._search_products_sync(query, top_k, must_contain)
    
    def retrieve_candidates(self, query_text, tokenizer, model, subcat, topn=50):
        """
        语义检索商品，并过滤出属于当前小类的商品
        
        🔧 委托给 ProductSelector 处理
        """
        return BudgetAllocator._product_selector.retrieve_candidates(query_text, tokenizer, model, subcat, topn)
    
    def find_product_id_by_name(self, product_name: str, product_data) -> str:
        """
        通过商品名称在商品库中精确匹配对应的product_id，匹配失败返回null
        """
        try:
            if hasattr(product_data, 'columns') and 'Product Name' in product_data.columns:
                # 只进行精确匹配
                exact_match = product_data[product_data['Product Name'] == product_name]
                if not exact_match.empty:
                    # 优先返回有Uniq Id列的记录
                    if 'Uniq Id' in exact_match.columns:
                        product_id = exact_match.iloc[0]['Uniq Id']
                        if pd.notna(product_id) and str(product_id).strip():
                            return str(product_id)
            
            # 如果找不到，返回null
            return None
        
        except Exception as e:
            # print(f"[Product ID匹配异常] {e}")
            return None
    
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
        
        # 策略2: 如果有product_id但没有owner_id，先查找owner_id
        if product_id and not owner_id and self.economic_center:
            try:
                # 从pro_firm_df查找所有可能的公司
                matched_companies = self.pro_firm_df[self.pro_firm_df['product_id'] == product_id]['company_id'].values
                if len(matched_companies) > 0:
                    # 尝试查询第一个公司的价格
                    price = ray.get(self.economic_center.query_price.remote(matched_companies[0], product_id))
                    if price and price > 0:
                        return price
            except Exception as e:
                logger.debug(f"通过product_id查找owner_id后查询价格失败 (product_id={product_id}): {e}")
        
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

    
    def save_allocation_results_with_history(self, family_id: str, current_month: int, 
                                           monthly_budget: float, category_budget: Dict[str, float],
                                           subcategory_budget: Dict, shopping_plan: Dict) -> None:
        """
        保存预算分配结果到4个独立的JSON文件，支持历史数据管理
        
        🔧 委托给 HistoryManager 处理
        """
        BudgetAllocator._history_manager.save_allocation_results_with_history(
            family_id, current_month, monthly_budget, category_budget, 
            subcategory_budget, shopping_plan
        )
    
    def _get_nutrition_needs(self, family_id: str) -> Optional[Dict[str, float]]:
        """
        获取家庭的营养需求（上月满足率）
        
        Returns:
            {'carbohydrate': 79.3, 'protein': 49.2, 'fat': 30.5, 'water': 16.4}
            或 None（如果无数据）
        """
        try:
            import os
            import json
            
            logger.info(f"🔍 [_get_nutrition_needs] 开始获取家庭 {family_id} 的营养需求...")
            
            # 从attribute_manager读取上月数据
            if not self.attribute_manager:
                logger.warning(f"⚠️ [_get_nutrition_needs] attribute_manager 为 None，无法获取营养数据")
                return None
            
            logger.info(f"✅ [_get_nutrition_needs] attribute_manager 存在")
            
            output_dir = self.attribute_manager.config.get('output_dir', 'output')
            state_file = os.path.join(output_dir, f"family_{family_id}", "family_state.json")
            
            logger.info(f"📁 [_get_nutrition_needs] 状态文件路径: {state_file}")
            
            if not os.path.exists(state_file):
                logger.warning(f"⚠️ [_get_nutrition_needs] 状态文件不存在: {state_file}")
                return None
            
            logger.info(f"✅ [_get_nutrition_needs] 状态文件存在，开始读取...")
            
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ [_get_nutrition_needs] 文件读取成功")
            
            # 读取上月满足率
            nutrition_ref = data.get('current_state', {}).get('nutrition_reference', {})
            last_supply = nutrition_ref.get('last_month_supply', {})
            last_consumption = nutrition_ref.get('last_month_consumption', {})
            
            logger.info(f"📊 [_get_nutrition_needs] last_supply keys: {list(last_supply.keys()) if last_supply else 'None'}")
            logger.info(f"📊 [_get_nutrition_needs] last_consumption keys: {list(last_consumption.keys()) if last_consumption else 'None'}")
            
            if not last_supply or not last_consumption:
                logger.warning(f"⚠️ [_get_nutrition_needs] 家庭 {family_id} 没有上月营养数据 (supply={bool(last_supply)}, consumption={bool(last_consumption)})")
                return None
            
            # 计算满足率
            result = {}
            for attr in ['carbohydrate_g', 'protein_g', 'fat_g', 'water_g']:
                supply = last_supply.get(attr, 0)
                consumption = last_consumption.get(attr, 1)
                
                if consumption > 0:
                    rate = (supply / consumption * 100)
                    rate = max(0, min(rate, 200))  # 限制在0-200%
                else:
                    rate = 0
                
                # 简化属性名
                attr_name = attr.replace('_g', '')
                result[attr_name] = rate
                
                logger.info(f"  • {attr_name}: supply={supply}, consumption={consumption}, rate={rate:.1f}%")
            
            logger.info(f"✅ [_get_nutrition_needs] 成功计算营养需求: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ [_get_nutrition_needs] 获取家庭 {family_id} 营养需求失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    async def allocate(self, family_id: str = None, current_month: int = None, current_income: float = None, total_balance: float = None, family_profile: str = None, max_workers: int = 32, ex_info=None,
                      nutrition_stock: Dict[str, float] = None, life_quality: Dict[str, float] = None, needs: Dict[str, Any] = None, benchmark_data: Dict[str, Any] = None,
                      last_month_budget: Optional[float] = None, last_month_attributes: Optional[Dict] = None) -> Dict[str, Any]:
        """
        主入口：分层分配家庭年度预算，输出每月每小类商品清单。
        输入：家庭id，当前月份，当前月份工资，家庭余额，家庭画像
        输出：当前月消费预算，当前月份大类预算，当前月份小类预算，当前月份待购买商品清单
        
        Args:
            family_id: 家庭ID
            current_month: 当前月份 (1-12)
            current_income: 当前月收入
            total_balance: 家庭总余额
            family_profile: 家庭画像信息
            max_workers: 最大工作线程数，默认32
        """
        total_start = time.perf_counter()
        timings: Dict[str, float] = {}
        # 如果family_id没有传入，从1-8000中随机选择
        if not family_id:
            family_id = str(random.randint(1, 8000))
            logger.info(f"随机选择家庭ID: {family_id}")
        
        # 如果家庭画像没有传入，根据家庭id获取对应的家庭画像
        if not family_profile:
            t0 = time.perf_counter()
            family_profile = self._get_family_profile_for_budget_calculation(family_id)
            timings["get_family_profile"] = time.perf_counter() - t0
            logger.info(f"已获取家庭{family_id}的画像信息")
        
        if ex_info:
            family_profile = ex_info + "\n " + family_profile
        
        # 设置默认月份为当前月份（如果未传入）
        if current_month is None:
            current_month = datetime.datetime.now().month
        
        # 计算当前月份的总消费预算
        t0 = time.perf_counter()
        monthly_budget = await self.calculate_monthly_budget(
            current_income=current_income,
            total_balance=total_balance,
            family_profile=family_profile,
            last_month_budget=last_month_budget,  # 🔧 新增：传递上月预算
            last_month_attributes=last_month_attributes  # 🔧 新增：传递上月属性
        )
        timings["calculate_monthly_budget"] = time.perf_counter() - t0
        
        # 🔧 修复：确保 monthly_budget 是数字类型（防御性编程）
        try:
            monthly_budget = float(monthly_budget)
        except (TypeError, ValueError) as e:
            logger.error(f"❌ monthly_budget 类型转换失败: {monthly_budget} ({type(monthly_budget)}), 错误: {e}")
            monthly_budget = 0.0
        
        logger.info(f"计算得出月度预算: {monthly_budget:.2f}")
        
        # 【新增】计算家庭属性缺口，用于引导预算分配（新版：基于营养和生活品质）
        # 优先使用 Household 传入的新版属性值
        attribute_gaps = {}
        if nutrition_stock is not None and life_quality is not None and needs is not None:
            # 使用新版属性系统计算缺口
            t0 = time.perf_counter()
            
            # 营养缺口
            nutrition_needs = needs.get('nutrition_needs', {})
            for attr, need in nutrition_needs.items():
                current = nutrition_stock.get(attr, 0.0)
                gap = max(0.0, need - current)
                attribute_gaps[f"nutrition_{attr}"] = gap
            
            # 生活品质缺口
            quality_needs = needs.get('quality_needs', {})
            for attr, need in quality_needs.items():
                current = life_quality.get(attr, 0.0)
                gap = max(0.0, need - current)
                attribute_gaps[f"quality_{attr}"] = gap
            
            timings["calculate_attribute_gaps"] = time.perf_counter() - t0
            
            # 记录详细的缺口信息
            urgent_attrs = {attr: gap for attr, gap in attribute_gaps.items() if gap > 100.0}  # 营养缺口阈值更高
            high_attrs = {attr: gap for attr, gap in attribute_gaps.items() if 50.0 < gap <= 100.0}
            # logger.info(
            #     f"✅ 计算属性缺口完成（新版属性系统），共{len(attribute_gaps)}个属性 | "
            #     f"急需(>100): {len(urgent_attrs)}个 {list(urgent_attrs.keys())} | "
            #     f"高优先级(50-100): {len(high_attrs)}个 {list(high_attrs.keys())}"
            # )
        elif self.attribute_manager and family_id and current_month:
            # 向后兼容：如果没有传入属性值，内部计算
            try:
                t0 = time.perf_counter()
                profile_dict = self._extract_family_profile_dict(family_profile)
                previous_month = str(max(0, current_month - 1))
                attribute_gaps = self.attribute_manager.calculate_family_attribute_gaps(
                    family_id, previous_month, family_profile=profile_dict
                )
                timings["calculate_attribute_gaps"] = time.perf_counter() - t0
                
                urgent_attrs = {attr: gap for attr, gap in attribute_gaps.items() if gap > 2.0}
                high_attrs = {attr: gap for attr, gap in attribute_gaps.items() if 1.0 < gap <= 2.0}
                # logger.info(
                #     f"计算属性缺口完成（内部计算，向后兼容），共{len(attribute_gaps)}个属性 | "
                #     f"急需(>2.0): {len(urgent_attrs)}个 {list(urgent_attrs.keys())} | "
                #     f"高优先级(1.0-2.0): {len(high_attrs)}个 {list(high_attrs.keys())}"
                # )
            except Exception as e:
                logger.warning(f"计算属性缺口失败: {e}，预算分配将不考虑属性需求")
                attribute_gaps = {}
        else:
            # 如果既没有传入属性值，也没有attribute_manager，则不使用属性缺口
            attribute_gaps = {}
            logger.info("属性系统未初始化，预算分配将不考虑属性需求")
        
        # 格式化社会基准数据，添加到家庭画像中（用于LLM参考）
        if benchmark_data and (nutrition_stock is not None or life_quality is not None):
            try:
                from agentsociety_ecosim.consumer_modeling.attribute_benchmark import AttributeBenchmarkManager
                
                # ========================================
                # 🔧 修复：传入正确的输出目录路径
                # 问题：AttributeBenchmarkManager 默认使用相对路径 "output"，导致找不到文件
                # 解决：传入绝对路径 /root/.../consumer_modeling/output
                # ========================================
                import os
                output_dir = os.path.join(
                    os.path.dirname(__file__),  # consumer_modeling 目录
                    "output"
                )
                benchmark_manager = AttributeBenchmarkManager(output_dir=output_dir)
                logger.debug(f"🔍 AttributeBenchmarkManager 使用输出目录: {output_dir}")
                
                # 准备当前家庭的属性信息
                current_family_attrs = {
                    'family_size': self._extract_family_profile_dict(family_profile).get('family_size', 1),
                    'nutrition_stock': nutrition_stock if nutrition_stock else {},
                    'life_quality': life_quality if life_quality else {},
                    'non_food_inventory': []  # 这里简化处理，实际库存信息在 attribute_system 中
                }
                
                # 格式化为 prompt 文本
                benchmark_text = benchmark_manager.format_benchmark_for_prompt(benchmark_data, current_family_attrs)
                
                # 将基准信息添加到 ex_info（在家庭画像之前提供参考）
                if ex_info:
                    ex_info = benchmark_text + "\n\n" + ex_info
                else:
                    ex_info = benchmark_text
                
                logger.info(f"✅ 社会基准信息已添加到决策上下文")
            except Exception as e:
                logger.debug(f"格式化基准数据失败（非致命错误）: {e}")
        
        # 调用allocate_monthly_budget_to_categories将月度预算分配到大类
        t0 = time.perf_counter()
        category_budget = await self.allocate_monthly_budget_to_categories(
            monthly_budget=monthly_budget,
            family_id=family_id,
            ex_info=ex_info,
            current_month=current_month,
            family_profile=family_profile,
            attribute_gaps=attribute_gaps
        )
        timings["allocate_monthly_budget_to_categories"] = time.perf_counter() - t0
        # logger.info(f"大类预算分配完成: {category_budget}")
        
        # 调用_allocate_subcategory_budget进行小类预算分配
        t0 = time.perf_counter()
        subcategory_budget = await self._allocate_subcategory_budget(
            category_budget=category_budget,
            family_id=family_id,
            max_workers=max_workers,
            ex_info=ex_info
        )
        timings["_allocate_subcategory_budget"] = time.perf_counter() - t0
        # logger.info(f"小类预算分配完成")
        
        # 【方案A：分层批量】生成商品清单
        t0 = time.perf_counter()
        shopping_plan = await self.allocate_subcategory_budget_to_products_hierarchical_batch(
            subcategory_budget=subcategory_budget,
            family_profile=family_profile,
            current_month=current_month,
            topn=20,
            max_workers=max_workers,
            ex_info=ex_info,
            family_id=family_id
        )
        timings["allocate_subcategory_budget_to_products_hierarchical_batch"] = time.perf_counter() - t0
        # logger.info(f"商品清单生成完成（分层批量优化）")
        # logger.info(f"商品清单生成完成（分层批量优化）")
        
        # 【旧版本】原始的商品分配方式（已注释，保留用于回退）
        # t0 = time.perf_counter()
        # shopping_plan = self.allocate_subcategory_budget_to_products(
        #     subcategory_budget=subcategory_budget,
        #     family_profile=family_profile,
        #     current_month=current_month,
        #     topn=20
        # )
        # timings["allocate_subcategory_budget_to_products"] = time.perf_counter() - t0
        # logger.info(f"商品清单生成完成")
        
        # 构建返回结果
        result = {
            "family_id": family_id,
            "current_month": current_month,
            "monthly_budget": monthly_budget,
            "category_budget": category_budget,
            "subcategory_budget": subcategory_budget,
            "shopping_plan": shopping_plan
        }
        
        # 保存预算分配结果到文件（带历史数据管理）
        try:
            t0 = time.perf_counter()
            self.save_allocation_results_with_history(
                family_id=family_id,
                current_month=current_month,
                monthly_budget=monthly_budget,
                category_budget=category_budget,
                subcategory_budget=subcategory_budget,
                shopping_plan=shopping_plan
            )
            timings["save_allocation_results_with_history"] = time.perf_counter() - t0
        except Exception as e:
            # 保存失败不影响主流程，只记录警告
            logger.warning(f"保存预算分配结果失败: {e}")
        
        # 【已废弃】计算并保存属性值 - 现在由 household.py 的 update_attributes_after_purchase() 完成
        # 属性更新逻辑已迁移到 Household 类中，由购买完成后触发
        # try:
        #     t0 = time.perf_counter()
        #     self._calculate_and_save_attributes(
        #         family_id=family_id,
        #         current_month=current_month,
        #         shopping_plan=shopping_plan,
        #         family_profile=family_profile
        #     )
        #     timings["calculate_and_save_attributes"] = time.perf_counter() - t0
        # except Exception as e:
        #     # 属性计算失败不影响主流程，只记录警告
        #     logger.warning(f"计算和保存属性值失败: {e}")
        
        # 在函数结尾统一打印各阶段耗时
        total_elapsed = time.perf_counter() - total_start
        try:
            ordered_keys = [
                "get_family_profile",
                "calculate_monthly_budget",
                "allocate_monthly_budget_to_categories",
                "_allocate_subcategory_budget",
                "allocate_subcategory_budget_to_products_hierarchical_batch",  # 新的分层批量方法
                "save_allocation_results_with_history",
                "calculate_and_save_attributes",  # 新增的属性计算
            ]
            summary_parts = [f"{k}={timings[k]:.3f}s" for k in ordered_keys if k in timings]
            summary = " | ".join(summary_parts + [f"total={total_elapsed:.3f}s"])
            logger.info(f"[allocate] timing: {summary}")
        except Exception:
            # 打印失败不影响返回
            logger.error(f"打印各阶段耗时失败: {e}")
            pass
        
        return result


    async def allocate_with_metrics(self, family_id: str = None, current_month: int = None, current_income: float = None,
                              total_balance: float = None, family_profile: str = None, max_workers: int = 32, ex_info=None,
                              nutrition_stock: Dict[str, float] = None, life_quality: Dict[str, float] = None, needs: Dict[str, Any] = None,
                              benchmark_data: Dict[str, Any] = None, last_month_budget: Optional[float] = None, last_month_attributes: Optional[Dict] = None) -> Dict[str, Any]:
        """
        使用本方法调用 allocate，并精确统计 LLM Token 用量与函数运行时间。

        说明：
        - 通过在本进程内“猴子补丁”方式拦截 OpenAI SDK 的 chat.completions.create 方法，
          从返回对象中读取 usage.prompt_tokens 与 usage.completion_tokens。
        - 若 OpenAI 兼容服务未返回 usage 字段，则 Token 数可能为 0，但 llm_calls 仍会统计调用次数。
        - 不修改现有 allocate 逻辑，统计完成后会恢复被替换的方法，避免影响其他代码路径。

        返回：包含原 allocate 返回值与 metrics 指标的字典：
        {
            "result": <allocate 的返回>,
            "metrics": {
                "allocate_elapsed_seconds": float,
                "llm_calls": int,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            }
        }
        """
        import time
        import threading

        # 线程安全的统计容器
        lock = threading.Lock()
        stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        # 从返回对象中尽力提取 usage（兼容不同 SDK 版本的对象结构）
        def _try_get_usage_from_response(resp):
            try:
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    prompt = getattr(usage, "prompt_tokens", None)
                    completion = getattr(usage, "completion_tokens", None)
                    if prompt is None and hasattr(usage, "get"):
                        prompt = usage.get("prompt_tokens", 0)
                        completion = usage.get("completion_tokens", 0)
                    return int(prompt or 0), int(completion or 0)

                if hasattr(resp, "model_dump"):
                    data = resp.model_dump()
                elif hasattr(resp, "dict"):
                    data = resp.dict()
                else:
                    data = resp

                if isinstance(data, dict) and "usage" in data:
                    u = data["usage"] or {}
                    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
            except Exception:
                pass
            return 0, 0

        # 猴子补丁：优先使用新版路径 openai.resources.chat.completions.Completions
        original_create = None
        patched_class = None
        try:
            try:
                from openai.resources.chat.completions import Completions as _Completions  # type: ignore
                patched_class = _Completions
                original_create = _Completions.create

                def wrapped_create(self, *args, **kwargs):  # type: ignore
                    resp = original_create(self, *args, **kwargs)
                    p, c = _try_get_usage_from_response(resp)
                    with lock:
                        stats["llm_calls"] += 1
                        stats["prompt_tokens"] += p
                        stats["completion_tokens"] += c
                    return resp

                _Completions.create = wrapped_create  # type: ignore
            except Exception:
                # 兼容路径：通过 OpenAI 实例拿到底层类
                from openai import OpenAI  # type: ignore
                tmp_client = OpenAI()
                comps = tmp_client.chat.completions
                patched_class = comps.__class__
                original_create = patched_class.create

                def wrapped_create(self, *args, **kwargs):  # type: ignore
                    resp = original_create(self, *args, **kwargs)
                    p, c = _try_get_usage_from_response(resp)
                    with lock:
                        stats["llm_calls"] += 1
                        stats["prompt_tokens"] += p
                        stats["completion_tokens"] += c
                    return resp

                patched_class.create = wrapped_create  # type: ignore
        except Exception:
            # 如果补丁失败，不影响主流程，只是无法统计精确 Token
            original_create = None
            patched_class = None

        # 计时开始
        t0 = time.perf_counter()
        result = await self.allocate(
            family_id=family_id,
            current_month=current_month,
            current_income=current_income,
            total_balance=total_balance,
            family_profile=family_profile,
            max_workers=max_workers,
            ex_info=ex_info,
            nutrition_stock=nutrition_stock,
            life_quality=life_quality,
            needs=needs,
            benchmark_data=benchmark_data,
            last_month_budget=last_month_budget,  # 🔧 新增：传递上月预算
            last_month_attributes=last_month_attributes  # 🔧 新增：传递上月属性
        )
        elapsed = time.perf_counter() - t0

        # 恢复被替换的方法
        if patched_class is not None and original_create is not None:
            try:
                patched_class.create = original_create  # type: ignore
            except Exception:
                pass

        # 组织指标
        prompt_tokens = int(stats["prompt_tokens"]) if isinstance(stats.get("prompt_tokens"), int) else 0
        completion_tokens = int(stats["completion_tokens"]) if isinstance(stats.get("completion_tokens"), int) else 0
        total_tokens = prompt_tokens + completion_tokens
        llm_calls = int(stats["llm_calls"]) if isinstance(stats.get("llm_calls"), int) else 0

        metrics = {
            "allocate_elapsed_seconds": round(elapsed, 3),
            "llm_calls": llm_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # 将统计结果写入到 monthly_consumption_history.json 对应月份的记录中
        try:
            # 与现有历史文件保持一致的目录结构
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            family_dir = os.path.join(output_dir, f"family_{family_id}")
            os.makedirs(family_dir, exist_ok=True)
            file_path = os.path.join(family_dir, "monthly_consumption_history.json")

            data = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            if "monthly_records" not in data:
                data = {
                    "family_id": family_id,
                    "file_type": "monthly_consumption",
                    "created_at": datetime.datetime.now().isoformat(),
                    "last_updated": datetime.datetime.now().isoformat(),
                    "monthly_records": []
                }

            # 合并到当月记录
            found = False
            for rec in data["monthly_records"]:
                if rec.get("month") == current_month:
                    rec.update(metrics)
                    found = True
                    break
            if not found:
                data["monthly_records"].append({"month": current_month, **metrics})

            data["last_updated"] = datetime.datetime.now().isoformat()
            data["monthly_records"].sort(key=lambda x: x.get("month", 0))

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            # 写入失败不影响主流程
            pass

        return result


    async def calculate_monthly_budget(self, current_income: float, total_balance: float, family_profile: str = None,
                                       last_month_budget: Optional[float] = None, last_month_attributes: Optional[Dict] = None) -> float:
        """
        计算月度预算，根据本月收入和总资产，调用LLM确定当前月的消费金额
        
        🔧 委托给 MonthlyBudgetCalculator 处理
        """
        return await BudgetAllocator._budget_calculator.calculate_monthly_budget(
            current_income, total_balance, family_profile,
            last_month_budget, last_month_attributes  # 🔧 新增：传递历史数据
        )
    
    async def allocate_monthly_budget_to_categories(self, monthly_budget: float, family_id: str, ex_info=None, 
                                                    current_month: int = None, family_profile: str = None, 
                                                    attribute_gaps: Dict[str, float] = None) -> Dict[str, float]:
        """
        将月度预算分配到大类消费类别（支持属性引导）
        
        🔧 委托给 CategoryAllocator 处理
        """
        return await BudgetAllocator._category_allocator.allocate_monthly_budget_to_categories(
            monthly_budget, family_id, ex_info, current_month, family_profile, attribute_gaps
        )
    
    def _get_family_info(self, family_id: str) -> Dict:
        """
        获取家庭信息
        """
        try:
            family_info = get_family_consumption_and_profile_by_id(family_id)
            return family_info or {}
        except Exception as e:
            logger.warning(f"获取家庭{family_id}信息失败: {e}")
            return {}
    
    def _build_family_profile_for_allocation(self, family_info: Dict) -> str:
        """
        构建用于预算分配的家庭画像
        """
        try:
            basic_info = family_info.get("basic_family_info", {})
            wealth_info = family_info.get("family_wealth_situation", {})
            family_profile_text = family_info.get("family_profile", "")
            
            profile_text = f"""
Family Profile for Budget Allocation:
{family_profile_text}

Basic Family Information:
- Family Size: {basic_info.get('family_size', 'N/A')} people
- Head Age: {basic_info.get('head_age', 'N/A')}
- Head Gender: {basic_info.get('head_gender', 'N/A')}
- Marital Status: {basic_info.get('head_marital_status', 'N/A')}
- Number of Children: {basic_info.get('num_children', 0)}
- Number of Vehicles: {basic_info.get('num_vehicles', 0)}

Wealth Analysis:
{wealth_info.get('wealth_analysis', 'No wealth analysis available')}
"""
            return profile_text.strip()
        except Exception as e:
            logger.warning(f"构建家庭画像失败: {e}")
            return "Standard family of 3 people, middle income"
    
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
                mapping = self.ATTRIBUTE_TO_CATEGORY_MAPPING.get(attr, {})
                primary_cats = mapping.get("primary", [])
                cat_names = [self.CATEGORY_NAMES_ZH.get(cat, cat) for cat in primary_cats]
                
                guidance += f"  - {attr}: gap = {gap:.1f}\n"
                guidance += f"    → Increase budget for: {', '.join(cat_names)}\n"
        
        # 高优先级属性缺口
        if high_attributes:
            guidance += "\n🟡 HIGH Priority Attribute Gaps (gap > 1.0, should address):\n"
            for attr, gap in sorted(high_attributes, key=lambda x: x[1], reverse=True):
                mapping = self.ATTRIBUTE_TO_CATEGORY_MAPPING.get(attr, {})
                primary_cats = mapping.get("primary", [])
                cat_names = [self.CATEGORY_NAMES_ZH.get(cat, cat) for cat in primary_cats]
                
                guidance += f"  - {attr}: gap = {gap:.1f}\n"
                guidance += f"    → Consider increasing: {', '.join(cat_names)}\n"
        
        # 添加建议的最小预算分配
        guidance += "\n📊 Recommended Minimum Budget Allocation:\n"
        guidance += "Based on the attribute gaps above, please ensure the following categories receive adequate budget:\n"
        
        # 收集需要增加预算的类别
        category_priority = {}  # {category: priority_score}
        for attr, gap in critical_attributes + high_attributes:
            mapping = self.ATTRIBUTE_TO_CATEGORY_MAPPING.get(attr, {})
            primary_cats = mapping.get("primary", [])
            weight = 2.0 if gap > 2.0 else 1.0  # 关键属性权重更高
            
            for cat in primary_cats:
                category_priority[cat] = category_priority.get(cat, 0) + gap * weight
        
        # 按优先级排序
        sorted_categories = sorted(category_priority.items(), key=lambda x: x[1], reverse=True)
        for cat, priority in sorted_categories[:5]:  # 最多显示前5个
            cat_name = self.CATEGORY_NAMES_ZH.get(cat, cat)
            guidance += f"  - {cat_name} ({cat}): priority score = {priority:.1f}\n"
        
        guidance += "\n⚠️  Please adjust the budget allocation to prioritize these categories while maintaining balance.\n"
        guidance += "=" * 60 + "\n"
        
        return guidance
    
    def _get_historical_consumption_data(self, family_info: Dict) -> List[List[float]]:
        """
        获取过去五年的年度大类消费记录（排除2021年）
        将年度数据除以12转换为月度平均消费记录
        """
        historical_data = []
        
        try:
            consumption_data = family_info.get("consumption", {})
            if not consumption_data:
                # 没有历史数据，创建默认数据
                default_monthly_amount = 1000.0  # 默认月度总支出
                equal_share = default_monthly_amount / len(self.CATEGORY_KEYS)
                for _ in range(5):
                    historical_data.append([equal_share] * len(self.CATEGORY_KEYS))
                return historical_data
            
            # 获取年份列表，排除2021年，按年份降序排列
            years = [y for y in sorted(consumption_data.keys(), reverse=True) if y != "2021"]
            
            for year in years[:5]:  # 最多取5年
                year_data = consumption_data[year]
                if not year_data:
                    continue
                
                # 获取该年各类别支出，确保所有类别都有值
                category_expenditures = []
                for category in self.CATEGORY_KEYS:
                    # 从PSID数据中获取对应类别的支出
                    # PSID数据中的类别名称可能与CATEGORY_KEYS不完全匹配，需要映射
                    expenditure = self._get_category_expenditure_from_psid(year_data, category)
                    category_expenditures.append(expenditure)
                
                # 计算总支出
                total_expenditure = sum(category_expenditures)
                
                if total_expenditure > 0:
                    # 将年度支出除以12转换为月度平均支出
                    monthly_expenditures = [exp / 12.0 for exp in category_expenditures]
                    historical_data.append(monthly_expenditures)
                else:
                    # 总支出为0，使用默认月度支出
                    default_monthly_amount = 1000.0
                    equal_share = default_monthly_amount / len(self.CATEGORY_KEYS)
                    monthly_expenditures = [equal_share] * len(self.CATEGORY_KEYS)
                    historical_data.append(monthly_expenditures)
            
            # 如果数据不足5年，用默认数据补充
            while len(historical_data) < 5:
                default_monthly_amount = 1000.0
                equal_share = default_monthly_amount / len(self.CATEGORY_KEYS)
                historical_data.append([equal_share] * len(self.CATEGORY_KEYS))
            
            logger.info(f"获取到{len(historical_data)}年历史消费数据（月度平均形式）")
            return historical_data
            
        except Exception as e:
            logger.error(f"获取历史消费数据失败: {e}")
            # 返回默认数据
            default_monthly_amount = 1000.0
            equal_share = default_monthly_amount / len(self.CATEGORY_KEYS)
            for _ in range(5):
                historical_data.append([equal_share] * len(self.CATEGORY_KEYS))
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
        """
        使用QAIDS方法分配月度预算
        """
        try:
            # from agentsociety_ecosim.consumer_modeling import QAIDS_model
            
            # 直接使用月度平均消费记录作为QAIDS输入
            # historical_data现在已经是月度平均支出金额，不需要转换
            qaids_allocation = QAIDS_model.predict_q_aids(
                historical_data, 
                monthly_budget, 
                list(self.CATEGORY_KEYS)
            )
            
            logger.info(f"QAIDS分配结果: {qaids_allocation}")
            return qaids_allocation
            
        except Exception as e:
            logger.error(f"QAIDS分配失败: {e}")
            return {}
    
    async def _adjust_allocation_with_llm(self, qaids_allocation: Dict[str, float], monthly_budget: float, 
                                   historical_data: List[List[float]], family_profile: str, 
                                   attribute_gaps: Dict[str, float] = None) -> Dict[str, float]:
        """
        使用LLM对QAIDS分配结果进行微调（支持属性引导）
        """
        try:
            
            # 构建历史数据描述
            historical_description = self._build_historical_description(historical_data)
            
            # 构建属性引导文本
            attribute_guidance = ""
            if attribute_gaps:
                attribute_guidance = self._build_attribute_guidance_prompt(attribute_gaps)
            
            # 构建微调提示
            prompt = f"""
You are a professional financial planner. Please review and adjust the following monthly budget allocation for a family.

Family Profile:
{family_profile}

Historical Consumption Patterns (proportions for the last 5 years):
{historical_description}

Current Monthly Budget: ${monthly_budget:.2f}

Initial QAIDS Allocation:
{json.dumps(qaids_allocation, indent=2)}
{attribute_guidance}
Please adjust this allocation considering:
1. The family's profile and needs
2. Historical consumption patterns
3. Seasonal factors
4. Basic living requirements
5. **Family attribute needs (MOST IMPORTANT if attribute guidance is provided above)**

The total must equal exactly ${monthly_budget:.2f}. Respond with ONLY a JSON object containing the adjusted allocation.
"""
            
            # 🔧 使用全局LLM信号量控制并发
            llm_semaphore = self.get_global_llm_semaphore()
            semaphore_wait_start = time.perf_counter()
            async with llm_semaphore:
                semaphore_wait_time = time.perf_counter() - semaphore_wait_start
                
                content = await llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a professional financial planner. Always respond with valid JSON.",
                    use_cache=True,  # 启用缓存
                    call_name="LLM-2-大类分配"
                )
                
                # if semaphore_wait_time > 0.1:
                #     print(f"⏳ [LLM-2] 信号量等待: {semaphore_wait_time:.2f}s")
            
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
                    if abs(total_allocated - monthly_budget) > 1e-2 and total_allocated > 0:
                        # 归一化到总预算
                        numeric_allocation = {k: v * monthly_budget / total_allocated for k, v in numeric_allocation.items()}
                    
                    # 处理舍入误差并保留两位小数
                    adjusted_allocation = self._normalize_allocation_to_budget(numeric_allocation, monthly_budget)
                    
                    logger.info(f"LLM微调完成: {adjusted_allocation}")
                    return adjusted_allocation
                else:
                    logger.warning("所有分配值无效，返回原始QAIDS分配")
                    return self._normalize_allocation_to_budget(qaids_allocation, monthly_budget)
            else:
                logger.warning("LLM微调失败，返回原始QAIDS分配")
                return self._normalize_allocation_to_budget(qaids_allocation, monthly_budget)
                
        except Exception as e:
            logger.error(f"LLM微调失败: {e}")
            return self._normalize_allocation_to_budget(qaids_allocation, monthly_budget)
    
    async def _allocate_with_llm_direct(self, monthly_budget: float, family_profile: str, 
                                       attribute_gaps: Dict[str, float] = None) -> Dict[str, float]:
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
            
            for category in self.CATEGORY_KEYS:
                category_name = self.CATEGORY_NAMES_ZH.get(category, category)
                prompt += f"- {category}: {category_name}\n"
            
            prompt += f"""
{attribute_guidance}
Please allocate the budget considering:
1. The family's needs and priorities
2. **Family attribute needs (MOST IMPORTANT if attribute guidance is provided above)**

The total must equal exactly ${monthly_budget:.2f}.

Respond with ONLY a JSON object containing the allocation.
"""
            
            # 🔧 使用全局LLM信号量控制并发
            llm_semaphore = self.get_global_llm_semaphore()
            semaphore_wait_start = time.perf_counter()
            async with llm_semaphore:
                semaphore_wait_time = time.perf_counter() - semaphore_wait_start
                
                content = await llm_utils.call_llm_chat_completion(
                    prompt,
                    system_content="You are a professional financial planner. Always respond with valid JSON.",
                    use_cache=True,  # 启用缓存
                    call_name="LLM-3-子类别分配"
                )
                
                # if semaphore_wait_time > 0.1:
                #     print(f"⏳ [LLM-3] 信号量等待: {semaphore_wait_time:.2f}s")
            
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
                allocation = self._normalize_allocation_to_budget(allocation, monthly_budget)
                
                logger.info(f"LLM直接分配完成: {allocation}")
                return allocation
            else:
                logger.warning("LLM直接分配失败，使用默认分配")
                return self._get_default_allocation(monthly_budget)
                
        except Exception as e:
            logger.error(f"LLM直接分配失败: {e}")
            return self._get_default_allocation(monthly_budget)
    
    def _build_historical_description(self, historical_data: List[List[float]]) -> str:
        """
        构建历史数据的描述文本
        """
        description = "Monthly average expenditures for each year:\n"
        
        for i, year_data in enumerate(historical_data):
            year_num = 2020 - i  # 假设从2020年开始
            description += f"Year {year_num}: "
            
            expenditures = []
            for j, amount in enumerate(year_data):
                category = self.CATEGORY_KEYS[j]
                category_name = self.CATEGORY_NAMES_ZH.get(category, category)
                expenditures.append(f"{category_name}: ${amount:.2f}")
            
            description += ", ".join(expenditures) + "\n"
        
        return description
    
    def _normalize_allocation_to_budget(self, allocation: Dict[str, float], monthly_budget: float) -> Dict[str, float]:
        """
        将分配结果归一化到指定预算，保留两位小数，确保总和等于预算
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
            max_category = max(rounded_allocation.items(), key=lambda x: x[1])[0]
            
            # 计算需要调整的差值
            diff = monthly_budget - total_allocated
            
            # 调整最大类别
            rounded_allocation[max_category] = round(rounded_allocation[max_category] + diff, 2)
            
            # 确保调整后的值不为负数
            if rounded_allocation[max_category] < 0:
                # 如果调整后为负数，重新分配
                rounded_allocation = self._redistribute_negative_allocation(rounded_allocation, monthly_budget)
        
        return rounded_allocation
    
    def _redistribute_negative_allocation(self, allocation: Dict[str, float], monthly_budget: float) -> Dict[str, float]:
        """
        重新分配负值分配
        """
        # 过滤掉负值和零值
        positive_allocations = {k: v for k, v in allocation.items() if v > 0}
        
        if not positive_allocations:
            # 如果没有正值，使用均匀分配
            equal_share = round(monthly_budget / len(self.CATEGORY_KEYS), 2)
            allocation = {category: equal_share for category in self.CATEGORY_KEYS}
            
            # 处理舍入误差
            total = sum(allocation.values())
            if abs(total - monthly_budget) > 0.01:
                diff = monthly_budget - total
                first_category = self.CATEGORY_KEYS[0]
                allocation[first_category] = round(allocation[first_category] + diff, 2)
        else:
            # 重新分配预算到正值类别
            total_positive = sum(positive_allocations.values())
            if total_positive > 0:
                # 按比例重新分配
                allocation = {}
                for category in self.CATEGORY_KEYS:
                    if category in positive_allocations:
                        proportion = positive_allocations[category] / total_positive
                        allocation[category] = round(monthly_budget * proportion, 2)
                    else:
                        allocation[category] = 0.0
                
                # 处理舍入误差
                total = sum(allocation.values())
                if abs(total - monthly_budget) > 0.01:
                    diff = monthly_budget - total
                    # 找到最大的正值类别进行调整
                    max_category = max(positive_allocations.items(), key=lambda x: x[1])[0]
                    allocation[max_category] = round(allocation[max_category] + diff, 2)
            else:
                # 如果所有值都为0，使用均匀分配
                equal_share = round(monthly_budget / len(self.CATEGORY_KEYS), 2)
                allocation = {category: equal_share for category in self.CATEGORY_KEYS}
                
                # 处理舍入误差
                total = sum(allocation.values())
                if abs(total - monthly_budget) > 0.01:
                    diff = monthly_budget - total
                    first_category = self.CATEGORY_KEYS[0]
                    allocation[first_category] = round(allocation[first_category] + diff, 2)
        
        return allocation
    
    def _get_default_allocation(self, monthly_budget: float) -> Dict[str, float]:
        """
        获取默认的大类分配（所有方法都失败时的备选方案）
        """
        # 使用均匀分配
        equal_share = round(monthly_budget / len(self.CATEGORY_KEYS), 2)
        allocation = {category: equal_share for category in self.CATEGORY_KEYS}
        
        # 处理舍入误差
        total = sum(allocation.values())
        if abs(total - monthly_budget) > 0.01:
            diff = monthly_budget - total
            first_category = self.CATEGORY_KEYS[0]
            allocation[first_category] = round(allocation[first_category] + diff, 2)
        
        logger.info(f"使用默认分配: {allocation}")
        return allocation

    
    def _get_family_profile_for_budget_calculation(self, family_id: str) -> str:
        """
        获取用于预算计算的家庭画像
        """
        try:
            family_info = self._get_family_info(family_id)
            return self._build_family_profile_for_allocation(family_info)
        except Exception as e:
            logger.warning(f"获取家庭{family_id}画像失败: {e}")
            return f"Family ID: {family_id}, Standard family of 3 people, middle income"
    
    async def _allocate_subcategory_budget(self, category_budget: Dict[str, float], family_id: str, max_workers: int = 32, ex_info=None) -> Dict[str, Dict[str, float]]:
        """
        将大类预算分配到小类（使用多线程处理）
        
        🔧 委托给 SubcategoryAllocator 处理
        """
        return await BudgetAllocator._subcategory_allocator.allocate_subcategory_budget(
            category_budget, family_id, max_workers, ex_info
        )
    
    async def _batch_select_products_for_category(self, category: str, subcategory_budgets: Dict[str, float], 
                                          family_profile: str, current_month: int, topn: int = 20, 
                                          family_id: str = None) -> tuple:
        """
        【方案A：分层批量】为单个大类的所有小类批量选择商品
        
        注：部分逻辑委托给 ProductSelector，但保留属性管理相关逻辑
        
        Returns:
            (选择结果, 候选商品池): 返回选择结果和用于选择的候选商品
        """
        if not subcategory_budgets:
            return {}
        
        # ========================================
        # 🔧 新增：获取营养需求并添加到 family_profile（仅对食品类）
        # ========================================
        logger.info(f"🔍 [营养引导检查] category={category}, category.lower()={category.lower()}, family_id={family_id}")
        
        if category.lower() in ['food_expenditure', 'food'] and family_id:
            logger.info(f"✅ [营养引导] 条件满足，开始获取家庭 {family_id} 的营养需求...")
            nutrition_needs = self._get_nutrition_needs(family_id)
            
            if nutrition_needs:
                logger.info(f"🥗 [营养引导] 成功获取营养需求: {nutrition_needs}")
                
                # 统计营养状况
                critical_count = sum(1 for rate in nutrition_needs.values() if rate < 50)
                improvement_count = sum(1 for rate in nutrition_needs.values() if 50 <= rate < 90)
                sufficient_count = sum(1 for rate in nutrition_needs.values() if rate >= 90)
                
                logger.info(f"📊 [营养引导] 营养状况统计: 严重不足={critical_count}, 需改善={improvement_count}, 充足={sufficient_count}")
                
                family_profile += "\n\n🥗 LAST MONTH'S NUTRITIONAL STATUS:\n"
                for nutrient, rate in nutrition_needs.items():
                    if rate < 50:
                        status = "🔴 CRITICAL SHORTAGE"
                    elif rate < 90:
                        status = "🟡 NEEDS IMPROVEMENT"
                    else:
                        status = "✅ SUFFICIENT"
                    family_profile += f"  • {nutrient.capitalize()}: {rate:.1f}% {status}\n"
                family_profile += "\n💡 Please prioritize products that address nutritional deficiencies.\n"
                
                logger.info(f"✅ [营养引导] 营养信息已添加到 family_profile")
            else:
                logger.warning(f"⚠️ [营养引导] 家庭 {family_id} 未获取到营养需求数据（可能是第1个月或数据缺失）")
        else:
            if category.lower() not in ['food_expenditure', 'food']:
                logger.info(f"⏭️ [营养引导] 跳过非食品类: {category}")
            elif not family_id:
                logger.warning(f"⚠️ [营养引导] family_id 为空，无法获取营养需求")
        
        logger.info(f"开始为大类 {category} 批量选择商品，包含 {len(subcategory_budgets)} 个小类")
        
        # 1. 为每个小类收集候选商品
        all_candidates = {}
        total_candidates = 0
        
        for subcategory, budget in subcategory_budgets.items():
            if budget < 3:  # 预算太小，跳过
                all_candidates[subcategory] = []
                continue
                
            # 不限制topn，让预算和level2逻辑自然控制候选数量
            # level2改进方案已经智能控制候选数，max_total_candidates = min(80, max(topn, budget/5))
            candidates = self._collect_candidates_for_subcategory(
                category, subcategory, budget, topn, family_id=family_id
            )
            all_candidates[subcategory] = candidates
            total_candidates += len(candidates)
        
        # 为所有候选商品添加属性值
        all_candidates = self._enrich_candidates_with_attributes(all_candidates)
            
        if total_candidates == 0:
            logger.warning(f"大类 {category} 没有找到任何候选商品")
            return {subcat: [] for subcat in subcategory_budgets.keys()}
        
        # 检查是否需要分批处理（如果小类太多）
        if len(subcategory_budgets) > 4:
            logger.info(f"大类 {category} 有 {len(subcategory_budgets)} 个小类，自动使用小批量处理")
            return await self._mini_batch_processing(
                category, subcategory_budgets, family_profile, current_month, topn, 
                family_id=family_id, all_candidates=all_candidates
            )
        
        # 2. 构建批量LLM prompt（新版不使用attribute_gaps）
        prompt = self._build_batch_product_selection_prompt(
            category, subcategory_budgets, all_candidates, family_profile, current_month, attribute_gaps=None
        )
        # logger.info(f"\n{'='*80}\n【步骤3: 商品选择 - LLM提示词】大类: {category}\n{'='*80}")
        # logger.info(prompt)
        # logger.info(f"{'='*80}\n")
        
        # 3. 调用LLM进行批量商品选择（带重试机制，使用全局信号量控制并发）
        max_retries = 3
        last_error = None
        
        # 获取全局LLM信号量
        llm_semaphore = self.get_global_llm_semaphore()
        
        for retry_count in range(max_retries):
            try:
                # 使用全局信号量控制LLM调用并发
                semaphore_wait_start = time.perf_counter()
                async with llm_semaphore:
                    semaphore_wait_time = time.perf_counter() - semaphore_wait_start
                    
                    content = await llm_utils.call_llm_chat_completion(
                        prompt,
                        system_content="You are a professional shopping assistant. Select appropriate products and quantities for each subcategory within the given budgets.",
                        call_name=f"LLM-5-商品选择-{category}"
                    )
                    # logger.info(f"\n{'='*80}\n【步骤3: 商品选择 - LLM响应】大类: {category}\n{'='*80}")
                    # logger.info(content)
                    # logger.info(f"{'='*80}\n")
                    
                    # if semaphore_wait_time > 0.1:
                    #     print(f"⏳ [LLM-5-{category}] 信号量等待: {semaphore_wait_time:.2f}s")
                
                llm_time = time.perf_counter() - semaphore_wait_start
                
                # 4. 解析批量响应（更宽松的解析）
                batch_selections = self._parse_batch_response_flexible(content)
                
                if batch_selections and len(batch_selections) > 0:
                    logger.info(f"大类 {category} 批量LLM调用成功，耗时: {llm_time:.3f}s，重试次数: {retry_count}")
                    
                    # 5. 验证和处理结果
                    final_results = self._process_batch_product_results(
                        category, subcategory_budgets, all_candidates, batch_selections
                    )
                    
                    return final_results
                else:
                    # 记录LLM响应内容用于调试
                    logger.error(f"🔍 大类 {category} LLM响应解析为空:")
                    logger.error(f"🔍   LLM响应长度: {len(content) if content else 0} 字符")
                    logger.error(f"🔍   LLM响应内容（前500字符）: {content[:500] if content else 'None'}")
                    logger.error(f"🔍   LLM响应内容（后500字符）: {content[-500:] if content and len(content) > 500 else content if content else 'None'}")
                    raise ValueError("批量响应解析结果为空")
                    
            except Exception as e:
                last_error = e
                # 🔍 调试信息：打印错误详情
                logger.error(f"🔍 大类 {category} 批量处理异常详情:")
                logger.error(f"🔍   异常类型: {type(e).__name__}")
                logger.error(f"🔍   异常信息: {str(e)}")
                logger.error(f"🔍   batch_selections类型: {type(batch_selections) if 'batch_selections' in locals() else 'undefined'}")
                if 'batch_selections' in locals() and batch_selections:
                    logger.error(f"🔍   batch_selections内容: {batch_selections}")
                logger.error(f"🔍   all_candidates keys: {list(all_candidates.keys()) if 'all_candidates' in locals() else 'undefined'}")
                logger.error(f"🔍   subcategory_budgets: {subcategory_budgets}")
                # 记录LLM响应内容
                if 'content' in locals() and content:
                    logger.error(f"🔍   LLM响应长度: {len(content)} 字符")
                    logger.error(f"🔍   LLM响应内容（前1000字符）: {content[:1000]}")
                    logger.error(f"🔍   LLM响应内容（后1000字符）: {content[-1000:] if len(content) > 1000 else ''}")
                
                if retry_count < max_retries - 1:
                    logger.warning(f"大类 {category} 批量处理第{retry_count + 1}次失败: {e}，正在重试...")
                    continue
                else:
                    logger.error(f"大类 {category} 批量处理重试{max_retries}次后仍失败: {e}")
        
        # 所有重试都失败，尝试小批量处理
        if len(subcategory_budgets) > 2:
            logger.info(f"大类 {category} 尝试小批量处理")
            try:
                return await self._mini_batch_processing(
                    category, subcategory_budgets, family_profile, current_month, topn, 
                    family_id=family_id, all_candidates=all_candidates
                )
            except Exception as e:
                logger.warning(f"大类 {category} 小批量处理也失败: {e}")
        
        # 最终回退到单独处理
        logger.warning(f"大类 {category} 回退到单独处理")
        return await self._fallback_individual_product_selection(
            category, subcategory_budgets, family_profile, current_month, topn, family_id=family_id
        )
    
    def _parse_batch_response_flexible(self, content: str) -> Dict:
        """更宽松的批量响应解析，允许部分成功"""
        try:
            # 首先尝试标准解析
            return llm_utils.parse_model_response(content)
        except Exception as e:
            logger.warning(f"标准解析失败: {e}，尝试宽松解析")
            
            # 尝试更宽松的JSON提取
            import json
            import re
            
            # 查找JSON块
            json_pattern = r'\{[\s\S]*?\}'
            json_matches = re.findall(json_pattern, content)
            
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict) and len(result) > 0:
                        logger.info("宽松解析成功")
                        return result
                except:
                    continue
            
            # 如果还是失败，尝试提取关键信息
            logger.warning("JSON解析完全失败，尝试文本提取")
            return self._extract_from_text(content)
    
    def _extract_from_text(self, content: str) -> Dict:
        """从文本中提取商品选择信息"""
        # 简单的文本解析逻辑
        result = {}
        lines = content.split('\n')
        
        current_category = None
        for line in lines:
            line = line.strip()
            if ':' in line and any(word in line.lower() for word in ['category', 'subcategory', '类别']):
                current_category = line.split(':')[1].strip()
                result[current_category] = []
            elif current_category and any(word in line.lower() for word in ['name', 'product', '商品', '产品']):
                # 提取商品信息的简单逻辑
                if current_category not in result:
                    result[current_category] = []
                # 这里可以添加更复杂的文本解析逻辑
        
        return result if result else {}
    
    async def _mini_batch_processing(self, category: str, subcategory_budgets: Dict[str, float],
                                   family_profile: str, current_month: int, topn: int, 
                                   family_id: str = None, all_candidates: Dict[str, List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        小批量处理：将大批量分解为更小的批次
        
        功能：
            - 将大类下的所有小类分成多个批次，每批最多2个小类
            - 每个批次单独调用LLM选择商品
            - 如果传入了 all_candidates，直接使用（已包含属性值），避免重复收集
        
        输入：
            category: 大类名称
            subcategory_budgets: 小类预算字典
            family_profile: 家庭画像
            current_month: 当前月份
            topn: 候选商品数量限制
            family_id: 家庭ID（可选）
            all_candidates: 已收集的候选商品字典（可选，如果传入则直接使用，避免重复收集）
        
        输出：
            Dict[str, List[Dict]] - 所有小类的商品选择结果
        """
        logger.info(f"开始小批量处理大类 {category}，共 {len(subcategory_budgets)} 个小类")
        
        # 按预算大小排序，优先处理预算大的
        sorted_subcats = sorted(subcategory_budgets.items(), key=lambda x: x[1], reverse=True)
        
        # 每次处理最多2个小类
        batch_size = 2
        all_results = {}
        
        for i in range(0, len(sorted_subcats), batch_size):
            batch = sorted_subcats[i:i + batch_size]
            batch_dict = dict(batch)
            
            try:
                # 🔧 优化：如果传入了已收集的候选商品（已包含属性值），直接使用；否则重新收集
                if all_candidates:
                    batch_candidates = {subcat: all_candidates.get(subcat, []) 
                                      for subcat in batch_dict.keys()}
                    logger.debug(f"使用已收集的候选商品（包含属性值）: {list(batch_candidates.keys())}")
                else:
                    # 回退到重新收集（向后兼容）
                    batch_candidates = {}
                    for subcat, budget in batch_dict.items():
                        candidates = self._collect_candidates_for_subcategory(category, subcat, budget, min(topn, 8), family_id=family_id)
                        # 为新收集的候选商品添加属性值
                        enriched_candidates = self._enrich_candidates_with_attributes({subcat: candidates})
                        batch_candidates[subcat] = enriched_candidates.get(subcat, candidates)
                    logger.debug(f"重新收集候选商品: {list(batch_candidates.keys())}")
                
                # 构建小批量prompt（更简洁）
                prompt = self._build_mini_batch_prompt(category, batch_dict, batch_candidates, family_profile)
                
                # 🔧 使用全局LLM信号量控制并发
                llm_semaphore = self.get_global_llm_semaphore()
                async with llm_semaphore:
                    content = await llm_utils.call_llm_chat_completion(
                        prompt,
                        system_content="You are a shopping assistant. Select products within budget."
                    )
                
                # 解析结果
                batch_result = self._parse_batch_response_flexible(content)
                all_results.update(batch_result)
                
                logger.info(f"小批次处理成功: {list(batch_dict.keys())}")
                
            except Exception as e:
                logger.warning(f"小批次处理失败: {e}，回退到单独处理")
                # 对这个小批次中的每个小类单独处理
                for subcat, budget in batch_dict.items():
                    try:
                        candidates = self._collect_candidates_for_subcategory(category, subcat, budget, topn, family_id=family_id)
                        selected = await llm_utils.llm_score_products(
                            candidates, budget, subcat, family_profile=family_profile
                        )
                        all_results[subcat] = selected
                    except Exception as e2:
                        logger.error(f"单独处理 {subcat} 也失败: {e2}")
                        all_results[subcat] = []
        
        return all_results
    
    def _build_mini_batch_prompt(self, category: str, subcategory_budgets: Dict[str, float],
                               all_candidates: Dict[str, List[Dict]], family_profile: str) -> str:
        """构建小批量处理的简化prompt"""
        # 压缩家庭画像
        profile_lines = family_profile.split('\n')[:3]  # 只取前3行关键信息
        compressed_profile = '\n'.join(profile_lines)
        
        prompt = f"""=== TASK ===
Select products for {category} within budget.

=== FAMILY ===
{compressed_profile}

=== BUDGETS ===
"""
        
        for subcat, budget in subcategory_budgets.items():
            prompt += f"{subcat}: ${budget:.2f}\n"
        
        prompt += "\n=== PRODUCTS ===\n"
        
        for subcat, candidates in all_candidates.items():
            prompt += f"\n{subcat}:\n"
            for i, product in enumerate(candidates[:6]):  # 只显示前6个商品
                # 🔧 新增：显示商品属性值（如果存在）
                attrs_str = ""
                attrs = product.get('attributes', {})
                if attrs:
                    if attrs.get('is_food'):
                        # 显示营养值
                        nutr = attrs.get('nutrition', {})
                        parts = [f"{k.replace('_g', '')}:{v:.1f}g" for k, v in nutr.items() 
                                if k.endswith('_g') and v > 0][:4]
                        attrs_str = f" [Nutr: {', '.join(parts)}]" if parts else ""
                    else:
                        # 显示满意度属性和持续时间
                        satis = attrs.get('satisfaction', {})
                        duration = attrs.get('duration_months')
                        parts = []
                        for attr_key in ['functional_utility', 'aesthetic_utility', 'symbolic_utility', 'social_utility', 'growth_utility']:
                            attr_data = satis.get(attr_key, {})
                            if isinstance(attr_data, dict):
                                monthly_supply = attr_data.get('monthly_supply', 0)
                                if monthly_supply > 0:
                                    display_name = attr_key.replace('_utility', '')
                                    parts.append(f"{display_name}:{monthly_supply:.2f}")
                        if duration:
                            parts.append(f"Duration:{duration}mo")
                        attrs_str = f" [Attrs: {', '.join(parts)}]" if parts else ""
                
                prompt += f"  {i+1}. {product.get('name', 'Unknown')} - ${product.get('price', 0):.2f}{attrs_str}\n"
        
        prompt += """
=== RESPONSE FORMAT ===
Return JSON only:
{
  "subcategory1": [{"name": "product_name", "quantity": 1, "price": 10.99}],
  "subcategory2": [{"name": "product_name", "quantity": 2, "price": 5.99}]
}

=== RULES ===
- Stay within budget for each subcategory
- Select 1-2 products per subcategory
- Use exact product names from the list above
"""
        
        return prompt

    def _get_level2_categories_for_level1(self, level1_name: str) -> List[Dict]:
        """
        获取指定level1分类下的所有level2子分类及其商品统计
        
        Args:
            level1_name: level1分类名称（如 "food"）
            
        Returns:
            List[Dict]: level2分类列表，每个包含 {name, product_count, count_ratio}
        """
        try:
            if not hasattr(self.df, 'columns') or 'level1' not in self.df.columns or 'level2' not in self.df.columns:
                return []
            
            # 获取该level1下的所有商品
            level1_df = self.df[self.df['level1'].str.lower() == level1_name.strip().lower()]
            total_products = len(level1_df)
            
            if total_products == 0:
                return []
            
            # 统计每个level2的商品数量
            level2_stats = []
            for level2_name in level1_df['level2'].dropna().unique():
                level2_df = level1_df[level1_df['level2'] == level2_name]
                product_count = len(level2_df)
                
                level2_stats.append({
                    'name': level2_name,
                    'product_count': product_count,
                    'count_ratio': product_count / total_products
                })
            
            # 按商品数量排序
            level2_stats.sort(key=lambda x: x['product_count'], reverse=True)
            
            return level2_stats
            
        except Exception as e:
            logger.warning(f"获取level1 {level1_name} 的level2分类失败: {e}")
            return []
    
    def _select_important_level2_categories(self, level1_name: str, budget: float, 
                                          max_level2_count: int = 15,
                                          min_coverage: float = 0.7) -> List[Dict]:
        """
        为level1下的所有level2分配候选数量（改进方案：全覆盖 + 按比例分配）
        
        Args:
            level1_name: level1分类名称
            budget: 该level1的预算
            max_level2_count: 不再使用（保持兼容）
            min_coverage: 不再使用（保持兼容）
            
        Returns:
            List[Dict]: 所有level2分类列表，每个包含 {name, product_count, weight, candidate_count}
        """
        # 获取所有level2
        level2_stats = self._get_level2_categories_for_level1(level1_name)
        
        if not level2_stats:
            return []
        
        total_level2 = len(level2_stats)
        
        # 🆕 改进：根据level2数量动态确定总候选池大小
        if total_level2 <= 10:
            total_candidate_pool = 60
        elif total_level2 <= 20:
            total_candidate_pool = 80
        else:
            # level2很多时，控制总数
            total_candidate_pool = min(100, total_level2 * 3)
        
        # 🆕 改进：动态确定每个level2的最小候选数
        if total_level2 <= 10:
            min_candidates_per_level2 = 2
        elif total_level2 <= 30:
            min_candidates_per_level2 = 1
        else:
            # 非常多的level2，允许某些只有1个候选
            min_candidates_per_level2 = 1
        
        # 🆕 改进：为每个level2按比例分配候选数量
        for stat in level2_stats:
            # 候选数 = max(最小值, min(最大值, 总池大小 × 该level2占比))
            proportional_count = int(total_candidate_pool * stat['count_ratio'])
            stat['candidate_count'] = max(min_candidates_per_level2, min(15, proportional_count))
            stat['weight'] = stat['count_ratio']
        
        # logger.info(
        #     f"Level1 {level1_name}: 全部 {total_level2} 个level2都会检索 "
        #     f"(总候选池目标: {total_candidate_pool}, 每个level2: {min_candidates_per_level2}-15个)"
        # )
        
        return level2_stats  # 🆕 返回所有level2，不再筛选
    
    def _generate_personalized_query(self, level2_name: str, level1_name: str = None, family_id: str = None) -> str:
        """
        根据家庭特征生成个性化检索关键字
        
        Args:
            level2_name: level2分类名称（基础关键字）
            level1_name: level1分类名称（用于类别约束，减少跨类别误检）
            family_id: 家庭ID
            
        Returns:
            个性化的检索关键字
        """
        # 🆕 优化1: 基础关键字优先包含level1作为硬性类别约束
        if level1_name:
            query = f"{level1_name} {level2_name}"
        else:
            query = level2_name
        
        if not family_id:
            return query
        
        try:
            # 获取家庭数据
            family_info = get_family_consumption_and_profile_by_id(int(family_id))
            if not family_info:
                return query
            
            modifiers = []  # 修饰词列表
            
            basic_info = family_info.get('basic_family_info', {})
            profile_text = family_info.get('family_profile', '').lower()
            exp_categories = family_info.get('expenditure_categories', {})
            
            # 1. 年龄特征（老年人）
            head_age = basic_info.get('head_age', 0)
            if head_age and head_age >= 65:
                modifiers.append('senior-friendly')
            
            # 2. 家庭规模
            family_size = basic_info.get('family_size', 0)
            if family_size and family_size >= 5:
                modifiers.append('family-size')
            elif family_size and family_size <= 2:
                modifiers.append('individual')
            
            # 3. 子女信息（婴幼儿特别重要）
            num_children = basic_info.get('num_children', 0)
            youngest_age = basic_info.get('youngest_child_age', 0)
            if num_children and num_children > 0:
                if youngest_age and youngest_age < 3:
                    modifiers.append('baby-safe')
                    modifiers.append('infant')
                elif youngest_age and youngest_age < 12:
                    modifiers.append('kids')
            
            # 4. 经济水平（通过食物支出估算）
            food_exp = exp_categories.get('food_expenditure', [])
            if food_exp:
                valid_exp = [x for x in food_exp if x is not None and x > 0]
                if valid_exp:
                    avg_food_exp = sum(valid_exp) / len(valid_exp)
                    if avg_food_exp > 10000:
                        modifiers.append('premium')
                    elif avg_food_exp < 3000:
                        modifiers.append('affordable')
            
            # 5. 从profile中提取关键词
            if 'health' in profile_text or 'organic' in profile_text:
                modifiers.append('healthy')
            
            # 组合：修饰词 + 基础关键字
            if modifiers:
                # 最多使用2个修饰词，避免查询太长（因为已经包含了level1）
                query = ' '.join(modifiers[:2]) + ' ' + query
                
        except Exception as e:
            logger.warning(f"生成个性化查询失败: {e}")
        
        return query
    
    def _collect_candidates_for_subcategory(self, category: str, subcategory: str, 
                                          budget: float, topn: int, family_id: str = None) -> List[Dict]:
        """
        为单个小类收集候选商品（使用level2改进方案）
        
        改进点：
        1. 不再直接用level1名称检索（如"food"）
        2. 而是先获取该level1下的重要level2子类（如"fresh vegetables", "meat & seafood"等）
        3. 为每个level2分别检索少量商品，然后合并
        4. 控制总候选数量，避免候选池爆炸
        """
        candidates = []
        all_candidates_with_level2 = []  # 记录每个候选商品来自哪个level2
        
        # 第1步：智能筛选重要的level2子类
        selected_level2 = self._select_important_level2_categories(
            level1_name=subcategory,
            budget=budget,
            max_level2_count=15,
            min_coverage=0.7
        )
        
        if not selected_level2:
            # 如果没有level2信息，回退到原始方案
            logger.warning(f"小类 {subcategory} 没有level2信息，使用原始检索方案")
            return self._collect_candidates_fallback(category, subcategory, budget, topn)
        
        # 第2步：为每个选中的level2检索候选商品
        price_range_min = budget * 0.005  # 0.5% (更宽松的下限)
        
        # 对小预算使用更宽松的上限，避免过度过滤
        if budget < 10:
            # 小预算：至少允许到$15，或预算的2倍
            price_range_max = max(budget * 2.0, 15.0)
        else:
            # 正常预算：使用0.8倍
            price_range_max = budget * 0.8
        
        logger.debug(f"价格过滤范围: ${price_range_min:.2f} - ${price_range_max:.2f} (预算: ${budget:.2f})")
        
        for level2_info in selected_level2:
            level2_name = level2_info['name']
            candidate_count = level2_info['candidate_count']
            
            try:
                # 🆕 生成个性化检索关键字（优化1: 传入level1_name以约束类别）
                personalized_query = self._generate_personalized_query(
                    level2_name=level2_name,
                    level1_name=subcategory,  # 传入level1作为类别约束
                    family_id=family_id
                )
                
                # 使用个性化关键字进行向量检索
                # 注意：暂时不使用 must_contain，因为向量库中的 classification 字段
                # 存储的是更细的分类（如 'Sugars, Oils, and Seasonings'），而不是 level1 值
                products = self._search_products_sync(
                    query=personalized_query,  # 🆕 关键改进：使用个性化关键字
                    top_k=candidate_count * 5,  # 🆕 优化2: 增加到5倍以应对高过滤率（80%+）
                    must_contain=None  # 暂时移除，避免过度过滤
                )
                
                # logger.debug(f"  └─ level2 '{level2_name}': 查询='{personalized_query}'")
                # logger.debug(f"      向量检索返回 {len(products)} 个原始商品")
                logger.debug(f"  └─ level2 '{level2_name}': 查询='{personalized_query}'")
                logger.debug(f"      向量检索返回 {len(products)} 个原始商品")
                
                level2_candidates = []
                price_filtered = 0
                category_filtered = 0  # 跨类别过滤计数
                
                for product in products:
                    if pd.isna(product.price) or product.price <= 0:
                        continue
                    
                    # 🆕 跨类别过滤：验证商品是否属于正确的 level1 大类
                    # 通过 product_id 在 CSV 中查询实际的 level1
                    product_id = getattr(product, 'product_id', '')
                    if product_id and hasattr(self, 'df') and 'Uniq Id' in self.df.columns:
                        product_row = self.df[self.df['Uniq Id'] == product_id]
                        if not product_row.empty:
                            actual_level1 = product_row.iloc[0].get('level1', '')
                            # 检查是否匹配预期的 level1（subcategory 就是 level1 名称）
                            if actual_level1 and actual_level1.lower() != subcategory.lower():
                                category_filtered += 1
                                continue  # 跳过不匹配的商品
                    
                    candidate = {
                        "name": product.name,
                        "price": float(product.price),
                        "product_id": product_id,
                        "owner_id": getattr(product, 'owner_id', ''),  # 🆕 添加公司ID
                        "source_level2": level2_name  # 记录来源
                    }
                    
                    # 价格过滤
                    if price_range_min <= product.price <= price_range_max:
                        level2_candidates.append(candidate)
                    else:
                        price_filtered += 1
                    
                    if len(level2_candidates) >= candidate_count:
                        break
                
                all_candidates_with_level2.extend(level2_candidates)
                # logger.info(
                #     f"     检索到 {len(level2_candidates)} 个候选 (原始:{len(products)}, 类别过滤:{category_filtered}, 价格过滤:{price_filtered}, 权重:{level2_info['weight']:.2f})"
                # )
                
            except Exception as e:
                logger.warning(f"level2 '{level2_name}' 向量检索失败: {e}")
                continue
        
        # 第3步：去重并控制总候选数量（基于商品名+公司ID）
        seen_products = set()  # 使用 (name, owner_id) 作为唯一标识
        for candidate in all_candidates_with_level2:
            # 🆕 使用 (name, owner_id) 作为唯一标识，允许同一商品的不同公司版本
            product_key = (candidate['name'], candidate.get('owner_id', ''))
            if product_key not in seen_products:
                seen_products.add(product_key)
                candidates.append(candidate)
        
        # 第4步：如果候选商品不足，用商品库补充
        if len(candidates) < 5:
            logger.info(f"小类 {subcategory} 候选商品不足({len(candidates)})，使用商品库补充")
            try:
                if hasattr(self.df, 'columns') and 'level1' in self.df.columns:
                    subcat_products = self.df[self.df['level1'].str.lower() == subcategory.strip().lower()]
                    subcat_products = subcat_products[subcat_products['List Price'] <= budget * 1.2]
                    subcat_products = subcat_products[subcat_products['List Price'] >= budget * 0.01]
                    
                    existing_products = {(c['name'], c.get('owner_id', '')) for c in candidates}
                    for _, item in subcat_products.head(15).iterrows():
                        product_id = item.get("product_id", "") or self.find_product_id_by_name(item["Product Name"], self.df)
                        owner_id = item.get("owner_id", "") or item.get("company_id", "")
                        product_key = (item["Product Name"], owner_id)
                        if product_key not in existing_products:
                            # 🆕 查询实时价格
                            real_time_price = self._get_real_time_price(
                                product_id=product_id,
                                product_name=item["Product Name"],
                                owner_id=owner_id
                            )
                            # 如果查询失败，使用CSV价格作为fallback
                            price = real_time_price if real_time_price is not None else item["List Price"]
                            
                            candidates.append({
                                "name": item["Product Name"],
                                "price": price,  # ✅ 使用实时价格
                                "product_id": product_id,
                                "owner_id": owner_id,  # 🆕 添加公司ID
                                "source_level2": "fallback"
                            })
                            if len(candidates) >= 15:
                                break
            except Exception as e:
                logger.warning(f"小类 {subcategory} 商品库补充失败: {e}")
        
        # 第5步：控制最终候选数量（基于topn和预算）
        # 总候选数应该在 topn 到 min(80, topn*3) 之间
        max_total_candidates = min(80, max(topn, int(budget / 5)))
        
        if len(candidates) > max_total_candidates:
            # 如果超出上限，按价格多样性选择
            candidates = sorted(candidates, key=lambda x: (abs(x['price'] - budget/10), x['price']))[:max_total_candidates]
        
        logger.info(
            f"小类 {subcategory} 收集到 {len(candidates)} 个候选商品 "
            f"(来自 {len(selected_level2)} 个level2子类)"
        )
        return candidates
    
    def _collect_candidates_fallback(self, category: str, subcategory: str, 
                                     budget: float, topn: int) -> List[Dict]:
        """
        原始的候选商品收集方案（作为fallback）
        """
        candidates = []
        query_text = f"{subcategory}"
        
        # 使用向量检索
        try:
            products = self._search_products_sync(query=query_text, top_k=topn * 3)
            price_range_min = budget * 0.01
            price_range_max = budget * 0.8
            
            for product in products:
                if pd.isna(product.price) or product.price <= 0:
                    continue
                candidate = {
                    "name": product.name,
                    "price": float(product.price),
                    "product_id": getattr(product, 'product_id', ''),
                    "owner_id": getattr(product, 'owner_id', '')  # 🆕 添加公司ID
                }
                if price_range_min <= product.price <= price_range_max:
                    candidates.append(candidate)
                if len(candidates) >= topn:
                    break
                    
        except Exception as e:
            logger.warning(f"小类 {subcategory} 向量检索失败: {e}")
        
        # 如果候选商品不足，用商品库补充
        if len(candidates) < 5:
            try:
                if hasattr(self.df, 'columns') and 'level1' in self.df.columns:
                    subcat_products = self.df[self.df['level1'].str.lower() == subcategory.strip().lower()]
                    subcat_products = subcat_products[subcat_products['List Price'] <= budget * 1.2]
                    subcat_products = subcat_products[subcat_products['List Price'] >= budget * 0.01]
                    
                    existing_products = {(c['name'], c.get('owner_id', '')) for c in candidates}
                    for _, item in subcat_products.head(15).iterrows():
                        product_id = item.get("product_id", "") or self.find_product_id_by_name(item["Product Name"], self.df)
                        owner_id = item.get("owner_id", "") or item.get("company_id", "")
                        product_key = (item["Product Name"], owner_id)
                        if product_key not in existing_products:
                            # 🆕 查询实时价格
                            real_time_price = self._get_real_time_price(
                                product_id=product_id,
                                product_name=item["Product Name"],
                                owner_id=owner_id
                            )
                            # 如果查询失败，使用CSV价格作为fallback
                            price = real_time_price if real_time_price is not None else item["List Price"]
                            
                            candidates.append({
                                "name": item["Product Name"],
                                "price": price,  # ✅ 使用实时价格
                                "product_id": product_id,
                                "owner_id": owner_id  # 🆕 添加公司ID
                            })
                            if len(candidates) >= 15:
                                break
            except Exception as e:
                logger.warning(f"小类 {subcategory} 商品库补充失败: {e}")
        
        logger.info(f"小类 {subcategory} 收集到 {len(candidates)} 个候选商品（fallback方案）")
        return candidates[:topn]
    
    def _enrich_candidates_with_attributes(self, all_candidates: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        为所有候选商品添加属性值信息（营养值或满意度属性）
        
        功能：
            - 遍历所有候选商品，通过 attribute_manager 获取每个商品的属性信息
            - 为食品商品添加营养值（carbohydrate_g, protein_g, fat_g, water_g等）
            - 为非食品商品添加满意度属性（functional_satisfaction等）
            - 将属性信息存储在候选商品的 'attributes' 字段中
        
        输入：
            all_candidates: Dict[str, List[Dict]] - 候选商品字典
                - key: subcategory (小类名称)
                - value: List[Dict] - 候选商品列表，每个商品包含 name, price, product_id 等
        
        输出：
            Dict[str, List[Dict]] - 增强后的候选商品字典
                - 每个候选商品新增 'attributes' 字段，包含属性信息
                - 格式：{'attributes': {'is_food': bool, 'nutrition': {...} 或 'satisfaction': {...}}}
        
        示例：
            输入: {'food': [{'name': 'Apple', 'price': 5.0, 'product_id': 'xxx'}]}
            输出: {'food': [{'name': 'Apple', 'price': 5.0, 'product_id': 'xxx', 
                            'attributes': {'is_food': True, 'nutrition': {'carbohydrate_g': 25.0, ...}}}]}
        """
        if not self.attribute_manager:
            return all_candidates
        
        enriched = {}
        for subcategory, candidates in all_candidates.items():
            enriched_list = []
            for candidate in candidates:
                enriched_candidate = candidate.copy()
                product_id = candidate.get('product_id') or self.find_product_id_by_name(
                    candidate.get('name', ''), self.df
                ) if hasattr(self, 'df') else None
                
                if product_id:
                    attrs = self.attribute_manager.get_product_attributes(product_id)
                    if attrs:
                        is_food = attrs.get('is_food', False)
                        enriched_candidate['attributes'] = {'is_food': is_food}
                        if is_food:
                            enriched_candidate['attributes']['nutrition'] = attrs.get('nutrition_supply', {})
                        else:
                            enriched_candidate['attributes']['satisfaction'] = attrs.get('satisfaction_attributes', {})
                            enriched_candidate['attributes']['duration_months'] = attrs.get('duration_months')
                    else:
                        # 🔧 修复：记录未找到属性的商品（用于调试）
                        product_name = candidate.get('name', 'Unknown')
                        logger.debug(f"商品 {product_name} (ID: {product_id}) 未找到属性信息")
                else:
                    # 🔧 修复：记录没有 product_id 的商品
                    product_name = candidate.get('name', 'Unknown')
                    logger.debug(f"商品 {product_name} 没有 product_id，跳过属性获取")
                
                enriched_list.append(enriched_candidate)
            enriched[subcategory] = enriched_list
        
        return enriched
    
    def _build_batch_product_selection_prompt(self, category: str, subcategory_budgets: Dict[str, float],
                                            all_candidates: Dict[str, List[Dict]], family_profile: str, 
                                            current_month: int, attribute_gaps: Dict[str, float] = None) -> str:
        """构建批量商品选择的prompt"""
        
        # 🔧 修复：去除 Family Profile 中的重复内容（就业状态和税收信息）
        import re
        pattern = r'(=== Current Household Employment Status ===.*?=== Please consider.*?===)'
        matches = list(re.finditer(pattern, family_profile, re.DOTALL))
        if len(matches) > 1:
            # 移除重复部分，只保留第一个
            for match in matches[1:]:
                family_profile = family_profile.replace(match.group(0), '', 1)
        
        prompt = f"""You are helping a family select products for the category "{category}" across multiple subcategories.

Family Profile: {family_profile}
Current Month: {current_month}
Category: {category}

"""
        
        # 添加属性需求信息
        if attribute_gaps:
            urgent_attributes = [attr for attr, gap in attribute_gaps.items() if gap > 5]
            if urgent_attributes:
                prompt += f"""
Family Attribute Needs Analysis:
- Current attribute gaps: {attribute_gaps}
- Urgent attributes (gap > 5): {urgent_attributes}

IMPORTANT: Each product below shows its attribute values in [Attrs: ...] format.
Please prioritize products with HIGH values in the urgent attributes listed above.
For example, if hunger_satisfaction gap is high, choose products with high hunger_satisfaction values.
The attribute values indicate how much each product can contribute to satisfying the family's needs.

"""
        
        prompt += """Please select appropriate products and quantities for each subcategory below. Each subcategory must reach at least 80% of its budget.

"""
        
        # 添加每个小类的详细信息
        for subcategory, budget in subcategory_budgets.items():
            candidates = all_candidates.get(subcategory, [])
            if not candidates:
                continue
                
            prompt += f"""
Subcategory: {subcategory}
Budget: ${budget:.2f}
Minimum spend: ${budget * 0.8:.2f}

Available products:
"""
            for i, candidate in enumerate(candidates, 1):
                attrs_str = ""
                attrs = candidate.get('attributes', {})
                if attrs:
                    if attrs.get('is_food'):
                        # 显示营养值
                        nutr = attrs.get('nutrition', {})
                        parts = [f"{k.replace('_g', '')}:{v:.1f}g" for k, v in nutr.items() 
                                if k.endswith('_g') and v > 0][:4]
                        attrs_str = f" [Nutr: {', '.join(parts)}]" if parts else ""
                    else:
                        # 显示满意度属性（每月提供的满意度）和持续时间（能提供几个月）
                        satis = attrs.get('satisfaction', {})
                        duration = attrs.get('duration_months')
                        
                        # 🔧 修复：从 satisfaction_attributes 中正确提取 monthly_supply 值
                        # JSON 结构：{"functional_utility": {"monthly_supply": 0.72, "reasoning": "..."}}
                        parts = []
                        for attr_key in ['functional_utility', 'aesthetic_utility', 'symbolic_utility', 'social_utility', 'growth_utility']:
                            attr_data = satis.get(attr_key, {})
                            if isinstance(attr_data, dict):
                                monthly_supply = attr_data.get('monthly_supply', 0)
                                if monthly_supply > 0:
                                    # 简化显示名：functional_utility -> functional
                                    display_name = attr_key.replace('_utility', '')
                                    parts.append(f"{display_name}:{monthly_supply:.2f}")
                        
                        # 添加持续时间信息
                        if duration:
                            parts.append(f"Duration:{duration}mo")
                        
                        attrs_str = f" [Attrs: {', '.join(parts)}]" if parts else ""
                
                owner_id = candidate.get('owner_id', 'N/A')
                prompt += f"{i}. {candidate['name']} - ${candidate['price']:.2f} (Company: {owner_id}){attrs_str}\n"
        
        prompt += f"""
⚠️ IMPORTANT: The same product name may be produced by different companies with different prices, quality, and attributes. You need to carefully compare and choose the best option based on price, quality, and family needs.

Respond with ONLY a JSON object in this exact format:
{{
"""
        
        # 添加示例格式
        first = True
        for subcategory in subcategory_budgets.keys():
            if all_candidates.get(subcategory):
                if not first:
                    prompt += ",\n"
                prompt += f'  "{subcategory}": [\n'
                prompt += f'    {{"name": "Product Name", "price": 10.50, "quantity": 2, "owner_id": "company_123"}}\n'  # 🆕 添加owner_id示例
                prompt += f'  ]'
                first = False
        
        prompt += """
}

⚠️ CRITICAL REQUIREMENTS:
1. Each subcategory reaches at least 80% of its budget
2. Use exact product names from the lists above
3. You MUST include "owner_id" (company ID) for each selected product
4. If multiple companies produce the same product, compare their prices, quality, and attributes, then select the best option
5. Choose realistic quantities for monthly family consumption
6. Total spending per subcategory should not exceed the budget
"""
        
        return prompt
    
    def _process_batch_product_results(self, category: str, subcategory_budgets: Dict[str, float],
                                     all_candidates: Dict[str, List[Dict]], 
                                     batch_selections: Dict) -> Dict[str, List[Dict]]:
        """处理批量商品选择的结果"""
        final_results = {}
        
        try:
            for subcategory, budget in subcategory_budgets.items():
                candidates = all_candidates.get(subcategory, [])
                if not candidates:
                    final_results[subcategory] = []
                    continue
                    
                # 检查LLM是否返回了该小类的结果
                if subcategory in batch_selections and isinstance(batch_selections[subcategory], list):
                    selected_products = []
                    
                    for item in batch_selections[subcategory]:
                        if isinstance(item, dict) and 'name' in item and 'price' in item and 'quantity' in item:
                            # 🆕 优先通过 (name, owner_id) 匹配
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
                                quantity = max(1, min(20, int(item['quantity'])))
                                price = float(item['price'])
                                
                                # 🆕 优先从LLM返回中获取owner_id，其次从候选商品中获取
                                result_owner_id = item.get('owner_id') or matching_candidate.get('owner_id', '')
                                
                                # 🔧 修复：直接从 matching_candidate 获取 product_id，避免重复查找
                                product_id = matching_candidate.get('product_id') or matching_candidate.get('id')
                                
                                # 如果候选商品中没有 product_id，再尝试查找
                                if not product_id:
                                    product_id = self.find_product_id_by_name(item['name'], self.df)
                                
                                # 🆕 如果仍然没有owner_id，尝试通过product_id查找（作为最后手段）
                                if not result_owner_id:
                                    result_owner_id = self.find_firm_id_by_name(product_id) if product_id else None
                                
                                selected_products.append({
                                    'name': item['name'],
                                    'price': price,
                                    'quantity': quantity,
                                    'total_spent': round(price * quantity, 2),
                                    'product_id': product_id,
                                    'owner_id': result_owner_id  # 🆕 使用获取到的owner_id
                                })
                    
                    # 记录选择结果（不再进行小类级补充，等待全局补充）
                    total_spent = sum(p['total_spent'] for p in selected_products)
                    utilization_rate = total_spent / budget if budget > 0 else 0
                    final_results[subcategory] = selected_products
                    logger.info(f"小类 {subcategory}: LLM选择完成, 预算 ${budget:.2f}, 花费 ${total_spent:.2f}, 利用率 {utilization_rate:.1%}")
                else:
                    logger.warning(f"小类 {subcategory}: LLM未返回结果")
                    final_results[subcategory] = []
            
            return final_results
            
        except Exception as e:
            # 🔍 调试信息：处理结果时的错误
            logger.error(f"🔍 处理批量结果时出错 (大类 {category}):")
            logger.error(f"🔍   异常类型: {type(e).__name__}")
            logger.error(f"🔍   异常信息: {str(e)}")
            logger.error(f"🔍   当前处理的小类: {subcategory if 'subcategory' in locals() else 'unknown'}")
            logger.error(f"🔍   batch_selections类型: {type(batch_selections)}")
            logger.error(f"🔍   batch_selections keys: {list(batch_selections.keys()) if isinstance(batch_selections, dict) else 'not a dict'}")
            if isinstance(batch_selections, dict) and 'subcategory' in locals():
                logger.error(f"🔍   当前小类在batch_selections中: {subcategory in batch_selections}")
                if subcategory in batch_selections:
                    logger.error(f"🔍   该小类的值类型: {type(batch_selections[subcategory])}")
                    logger.error(f"🔍   该小类的值: {batch_selections[subcategory]}")
            logger.error(f"🔍   all_candidates keys: {list(all_candidates.keys())}")
            logger.error(f"🔍   subcategory_budgets: {subcategory_budgets}")
            raise
    
    def _apply_global_supplement(self, final_results: Dict, subcategory_budget: Dict, 
                                 family_profile: str) -> Dict:
        """
        应用全局补充策略（简化调用接口）
        
        Args:
            final_results: LLM选择的商品结果
            subcategory_budget: 预算分配
            family_profile: 家庭画像
            
        Returns:
            补充后的结果
        """
        try:
            # 1. 提取家庭信息
            profile_dict = self._extract_family_profile_dict(family_profile)
            family_size = profile_dict.get('family_size', 1)
            
            # 2. 获取家庭规模系数
            size_key = str(int(family_size)) if family_size <= 5 else "6+"
            size_coefficients = {'1': 1.0, '2': 0.85, '3': 0.75, '4': 0.70, '5': 0.65, '6+': 0.60}
            family_coefficient = size_coefficients.get(size_key, 1.0)
            
            # 3. 营养配置（从配置文件读取，如果失败则使用默认值）
            nutrition_config = {
                'carbohydrate_g_per_month': 9000,
                'protein_g_per_month': 1800,
                'fat_g_per_month': 2100,
                'water_g_per_month': 15000  # 🔧 优化：降低水消耗标准
            }
            
            # 4. 获取商品属性映射文件路径
            attr_config_file = os.path.join(os.path.dirname(__file__), 'family_attribute_config.json')
            product_attr_file = ''
            
            if os.path.exists(attr_config_file):
                import json
                try:
                    with open(attr_config_file, 'r') as f:
                        attr_config = json.load(f)
                        
                        # 读取营养标准
                        if 'nutrition_reference' in attr_config:
                            nutrition_ref = attr_config['nutrition_reference']
                            nutrition_config['carbohydrate_g_per_month'] = nutrition_ref.get('carbohydrate_g_per_month', 9000)
                            nutrition_config['protein_g_per_month'] = nutrition_ref.get('protein_g_per_month', 1800)
                            nutrition_config['fat_g_per_month'] = nutrition_ref.get('fat_g_per_month', 2100)
                            nutrition_config['water_g_per_month'] = nutrition_ref.get('water_g_per_month', 15000)
                        
                        # 读取商品属性文件路径
                        product_attr_file = attr_config.get('product_attribute_file', '')
                except Exception as e:
                    logger.debug(f"从配置文件读取信息失败，使用默认值: {e}")
            else:
                logger.debug("未找到属性配置文件")
                return final_results
            
            # 5. 调用全局补充
            if product_attr_file and os.path.exists(product_attr_file):
                return self._global_attribute_supplement(
                    final_results, 
                    subcategory_budget,
                    family_size,
                    nutrition_config,
                    family_coefficient,
                    product_attr_file
                )
            else:
                logger.debug("未找到商品属性映射文件")
                return final_results
                
        except Exception as e:
            # 打印详细调试信息
            logger.warning(f"全局补充执行失败（非致命错误）: {e}")
            logger.warning(f"错误类型: {type(e).__name__}")
            logger.warning(f"错误详情: {str(e)}")
            import traceback
            logger.warning(f"完整堆栈:\n{traceback.format_exc()}")
            
            # 打印 final_results 的结构
            logger.warning(f"final_results 的键: {list(final_results.keys())}")
            for category, category_data in final_results.items():
                if isinstance(category_data, dict):
                    logger.warning(f"  {category} 的子类: {list(category_data.keys())}")
                    for subcategory, products in category_data.items():
                        if isinstance(products, list):
                            logger.warning(f"    {subcategory} 有 {len(products)} 个产品")
                            if products:
                                # 打印前2个产品的键
                                for i, p in enumerate(products[:2]):
                                    if isinstance(p, dict):
                                        logger.warning(f"      产品{i}的键: {list(p.keys())}")
                                    else:
                                        logger.warning(f"      产品{i}不是字典: type={type(p)}, value={p}")
            return final_results
    
    def _global_attribute_supplement(self, final_results: Dict, subcategory_budget: Dict, 
                                     family_size: float, nutrition_config: Dict, family_coefficient: float,
                                     product_attr_file: str) -> Dict:
        """
        全局属性补充：根据所有已选商品的属性，判断是否需要补充
        
        Args:
            final_results: 所有大类的商品选择结果
            subcategory_budget: 原始预算分配
            family_size: 家庭规模
            nutrition_config: 营养配置
            family_coefficient: 家庭规模系数
            product_attr_file: 商品属性映射文件路径
            
        Returns:
            补充后的结果
        """
        import json
        
        # 加载商品属性映射
        try:
            with open(product_attr_file, 'r') as f:
                attr_data = json.load(f)
                # 添加安全检查，确保 item 有 'product_name' 键
                product_mappings = {}
                for item in attr_data.get('product_mappings', []):
                    if isinstance(item, dict) and 'product_name' in item:
                        product_mappings[item['product_name']] = item
        except Exception as e:
            logger.warning(f"无法加载商品属性映射，跳过全局补充: {e}")
            return final_results
        
        # 计算本月需求（使用传入的nutrition_config，已经包含最新的标准）
        monthly_needs = {
            'carbohydrate_g': nutrition_config.get('carbohydrate_g_per_month', 9000) * family_size * family_coefficient,
            'protein_g': nutrition_config.get('protein_g_per_month', 1800) * family_size * family_coefficient,
            'fat_g': nutrition_config.get('fat_g_per_month', 2100) * family_size * family_coefficient,
            'water_g': nutrition_config.get('water_g_per_month', 15000) * family_size * family_coefficient  # 🔧 使用新标准
        }
        
        # 计算已选商品提供的属性
        provided = {'carbohydrate_g': 0, 'protein_g': 0, 'fat_g': 0, 'water_g': 0}
        
        for category, category_data in final_results.items():
            if isinstance(category_data, dict):
                for subcategory, products in category_data.items():
                    if isinstance(products, list):
                        for idx, product in enumerate(products):
                            # 确保 product 是字典类型
                            if not isinstance(product, dict):
                                continue
                                
                            product_name = product.get('name', '')
                            if not product_name:
                                continue
                            
                            quantity = product.get('quantity', 1)
                            
                            # 获取商品属性
                            if product_name in product_mappings:
                                mapping = product_mappings[product_name]
                                if mapping.get('is_food', False):
                                    nutrition = mapping.get('nutrition_supply', {})
                                    for key in provided.keys():
                                        provided[key] += nutrition.get(key, 0) * quantity
        
        # 🔧 优化：计算缺口和过剩情况
        gaps = {}
        over_supplied = {}
        satisfaction_rates = {}
        
        for key, need in monthly_needs.items():
            provided_val = provided.get(key, 0)
            rate = (provided_val / need * 100) if need > 0 else 0
            satisfaction_rates[key] = rate
            
            if rate < 80:  # 低于80%才补充
                gap = need - provided_val
                gaps[key] = gap
            elif rate > 150:  # 超过150%记录为过剩
                over_supplied[key] = rate
        
        if not gaps:
            logger.info("✅ 属性已满足，无需全局补充")
            logger.info(f"   满足率: " + ", ".join([f"{k}={v:.0f}%" for k, v in satisfaction_rates.items()]))
            return final_results
        
        logger.info(f"🔍 检测到属性缺口: " + ", ".join([f"{k}={satisfaction_rates[k]:.0f}% (缺{v:.0f})" for k, v in gaps.items()]))
        if over_supplied:
            logger.info(f"⚠️ 已过剩属性: " + ", ".join([f"{k}={v:.0f}%" for k, v in over_supplied.items()]))
        
        # 🔧 优化：计算剩余预算
        food_budget = subcategory_budget.get('food_expenditure', {})
        if isinstance(food_budget, dict):
            total_food_budget = sum(food_budget.values())
        else:
            total_food_budget = float(food_budget) if food_budget else 0
        
        # 计算LLM已花费金额
        llm_spent = 0.0
        if 'food_expenditure' in final_results and isinstance(final_results['food_expenditure'], dict):
            for subcategory, products in final_results['food_expenditure'].items():
                if isinstance(products, list):
                    for p in products:
                        if isinstance(p, dict):
                            llm_spent += p.get('total_spent', 0)
        
        # 计算剩余预算（最多用剩余预算的50%进行补充）
        remaining_budget = total_food_budget - llm_spent
        max_supplement_budget = remaining_budget * 0.8
        
        logger.info(f"💰 预算情况: 总预算=${total_food_budget:.2f}, LLM花费=${llm_spent:.2f}, 剩余=${remaining_budget:.2f}")
        logger.info(f"💰 补充预算上限: ${max_supplement_budget:.2f} (剩余预算的80%)")
        
        if max_supplement_budget <= 0:
            logger.info("⚠️ 剩余预算不足，跳过补充")
            return final_results
        
        # 🔧 优化：从全局商品库收集候选商品（而不是从已选商品）
        food_candidates = []
        
        # 获取已选商品名称（用于去重）
        selected_product_names = set()
        if 'food_expenditure' in final_results and isinstance(final_results['food_expenditure'], dict):
            for subcategory, products in final_results['food_expenditure'].items():
                if isinstance(products, list):
                    for p in products:
                        if isinstance(p, dict) and 'name' in p:
                            selected_product_names.add(p.get('name', ''))
        
        # 从全局商品库筛选食品类商品
        try:
            food_df = self.df[self.df['level1'].str.lower() == 'food']
            for _, row in food_df.iterrows():
                product_name = row.get('Product Name', '')
                if not product_name or product_name in selected_product_names:
                    continue  # 跳过已选商品
                
                if product_name in product_mappings:
                    mapping = product_mappings[product_name]
                    if mapping.get('is_food', False):
                        nutrition = mapping.get('nutrition_supply', {})
                        
                        # 🔧 优化：只选择能帮助填补缺口的商品
                        can_help = False
                        for key in gaps.keys():
                            if nutrition.get(key, 0) > 0:
                                can_help = True
                                break
                        
                        if can_help:
                            product_id = row.get('Product ID', '') or self.find_product_id_by_name(product_name, self.df)
                            # 🆕 查询实时价格
                            real_time_price = self._get_real_time_price(
                                product_id=product_id,
                                product_name=product_name,
                                owner_id=None
                            )
                            # 如果查询失败，使用CSV价格作为fallback
                            price = real_time_price if real_time_price is not None else row.get('List Price', 0)
                            
                            food_candidates.append({
                                'name': product_name,
                                'price': price,  # ✅ 使用实时价格
                                'product_id': product_id,
                                'owner_id': None  # 需要后续查找
                            })
            
            logger.info(f"📦 从全局商品库收集到 {len(food_candidates)} 个候选商品（已排除{len(selected_product_names)}个已选商品）")
        except Exception as e:
            logger.warning(f"从全局商品库收集候选商品失败: {e}，回退到已选商品")
            # 回退：从已选商品中提取
            if 'food_expenditure' in final_results and isinstance(final_results['food_expenditure'], dict):
                for subcategory, products in final_results['food_expenditure'].items():
                    if isinstance(products, list):
                        for p in products:
                            if isinstance(p, dict) and 'name' in p:
                                p_name = p.get('name', '')
                                if p_name and p_name in product_mappings:
                                    food_candidates.append(p)
        
        # 🔧 优化：按性价比排序（属性价值/价格），同时避免过剩营养素
        def calc_value_score(product_name):
            if product_name not in product_mappings:
                return 0
            mapping = product_mappings[product_name]
            if not mapping.get('is_food', False):
                return 0
            nutrition = mapping.get('nutrition_supply', {})
            
            # 修复：使用正确的列名 'Product Name' 和 'List Price'
            try:
                matched = self.df[self.df['Product Name'] == product_name]
                if len(matched) > 0:
                    # 优先使用 'List Price'，如果没有则尝试 'price'
                    if 'List Price' in matched.columns:
                        price = matched['List Price'].iloc[0]
                    elif 'price' in matched.columns:
                        price = matched['price'].iloc[0]
                    else:
                        price = 1
                else:
                    price = 1
            except:
                price = 1
            
            # 🔧 新增：检查是否会加剧过剩
            penalty = 0
            for over_key in over_supplied.keys():
                over_val = nutrition.get(over_key, 0)
                if over_val > 0:
                    # 如果商品提供过剩营养素，给予惩罚（惩罚值与提供量成正比）
                    penalty += over_val * 2.0  # 惩罚系数
            
            # 计算能填补的缺口价值
            value = 0
            for key, gap in gaps.items():
                provided_val = nutrition.get(key, 0)
                if provided_val > 0:
                    weight = min(gap / 1000, 3.0)  # 缺口越大权重越高
                    value += provided_val * weight
            
            # 最终得分 = (填补价值 - 过剩惩罚) / 价格
            final_value = max(0, value - penalty)
            return final_value / price if price > 0 else 0
        
        # 去重并排序（添加安全检查）
        unique_products = {p.get('name', ''): p for p in food_candidates if 'name' in p and p.get('name')}.values()
        sorted_candidates = sorted(unique_products, key=lambda x: calc_value_score(x.get('name', '')), reverse=True)
        
        # 🔧 优化：智能补充商品（根据缺口大小动态调整数量，严格控制预算）
        supplement_products = []
        remaining_gaps = gaps.copy()
        supplement_budget = 0
        max_supplement_items = 8  # 最大补充商品种类数
        
        logger.info(f"📦 开始智能补充，候选商品数: {len(sorted_candidates)}")
        
        for idx, candidate in enumerate(sorted_candidates[:max_supplement_items * 2]):  # 扩大搜索范围
            product_name = candidate.get('name', '')
            if not product_name or product_name not in product_mappings:
                continue
            
            mapping = product_mappings[product_name]
            nutrition = mapping.get('nutrition_supply', {})
            
            # 检查是否能帮助填补缺口（且不会加剧过剩）
            can_help = False
            will_worsen_oversupply = False
            
            for key in gaps.keys():
                if nutrition.get(key, 0) > 0 and remaining_gaps.get(key, 0) > 0:
                    can_help = True
                    break
            
            for over_key in over_supplied.keys():
                if nutrition.get(over_key, 0) > 50:  # 如果商品提供大量过剩营养素
                    will_worsen_oversupply = True
                    break
            
            if not can_help or will_worsen_oversupply:
                continue
            
            # 🔧 新增：根据缺口大小和剩余预算动态计算数量
            price = candidate.get('price', 0)
            if price <= 0:
                continue
            
            # 🔧 关键：检查预算限制
            if supplement_budget >= max_supplement_budget:
                logger.info(f"   ⚠️ 已达补充预算上限 ${max_supplement_budget:.2f}，停止补充")
                break
            
            # 计算该商品能填补的最大缺口
            max_gap_ratio = 0
            for key, gap in remaining_gaps.items():
                provided_val = nutrition.get(key, 0)
                if provided_val > 0 and gap > 0:
                    ratio = gap / provided_val
                    max_gap_ratio = max(max_gap_ratio, ratio)
            
            # 数量 = min(根据缺口计算, 根据预算计算, 10)
            quantity_by_gap = min(int(max_gap_ratio * 0.5) + 1, 10)  # 填补50%的最大缺口
            quantity_by_budget = int((max_supplement_budget - supplement_budget) / price)  # 预算允许的数量
            quantity = min(quantity_by_gap, quantity_by_budget)
            quantity = max(1, quantity)  # 至少1个
            
            # 再次检查是否会超预算
            if supplement_budget + price * quantity > max_supplement_budget:
                # 调整数量以不超预算
                quantity = int((max_supplement_budget - supplement_budget) / price)
                if quantity < 1:
                    logger.info(f"   ⚠️ 剩余预算不足以购买 {product_name}，停止补充")
                    break
            
            supplement_products.append({
                'name': product_name,
                'price': price,
                'quantity': quantity,
                'total_spent': price * quantity,
                'product_id': candidate.get('product_id', ''),
                'owner_id': candidate.get('owner_id', '')
            })
            supplement_budget += price * quantity
            
            # 更新剩余缺口
            for key in remaining_gaps:
                provided_val = nutrition.get(key, 0) * quantity
                remaining_gaps[key] = max(0, remaining_gaps[key] - provided_val)
            
            logger.info(f"   补充商品 {len(supplement_products)}: {product_name} x{quantity} (${price * quantity:.2f}), 累计花费: ${supplement_budget:.2f}/{max_supplement_budget:.2f}")
            
            # 如果主要缺口已填补到90%，停止
            if all(remaining_gaps[key] < gaps[key] * 0.1 for key in gaps.keys()):
                logger.info(f"   ✅ 缺口已基本填补，停止补充")
                break
            
            # 限制补充商品种类数
            if len(supplement_products) >= max_supplement_items:
                logger.info(f"   ⚠️ 已达最大补充商品种类数 {max_supplement_items}，停止补充")
                break
        
        if supplement_products:
            # 将补充的商品添加到food_expenditure的food小类
            if 'food_expenditure' not in final_results:
                final_results['food_expenditure'] = {}
            if not isinstance(final_results['food_expenditure'], dict):
                final_results['food_expenditure'] = {}
            if 'food' not in final_results['food_expenditure']:
                final_results['food_expenditure']['food'] = []
            
            final_results['food_expenditure']['food'].extend(supplement_products)
            
            # 🔧 新增：计算补充后的满足率
            final_provided = provided.copy()
            for product in supplement_products:
                product_name = product.get('name', '')
                quantity = product.get('quantity', 1)
                if product_name in product_mappings:
                    mapping = product_mappings[product_name]
                    if mapping.get('is_food', False):
                        nutrition = mapping.get('nutrition_supply', {})
                        for key in final_provided.keys():
                            final_provided[key] += nutrition.get(key, 0) * quantity
            
            final_rates = {}
            for key, need in monthly_needs.items():
                rate = (final_provided[key] / need * 100) if need > 0 else 0
                final_rates[key] = rate
            
            logger.info(f"✅ 全局补充完成: 添加{len(supplement_products)}种商品, 总计{sum(p['quantity'] for p in supplement_products)}件, 金额${supplement_budget:.2f}")
            logger.info(f"   补充后满足率: " + ", ".join([f"{k}={v:.0f}%" for k, v in final_rates.items()]))
        else:
            logger.info("⚠️ 无合适商品用于补充（可能因为会加剧过剩）")
        
        return final_results
    
    def _generate_fallback_selection(self, subcategory: str, budget: float, candidates: List[Dict], llm_selected: List[Dict] = None) -> List[Dict]:
        """为单个小类生成备用商品选择，可选择在LLM选择基础上进行贪心补充"""
        if not candidates:
            return []
        
        # 如果提供了LLM选择结果，则在其基础上进行补充
        if llm_selected:
            selected = llm_selected.copy()  # 保留LLM的选择
            remaining_budget = budget - sum(p['total_spent'] for p in llm_selected)
            logger.info(f"小类 {subcategory}: 基于LLM选择进行贪心补充，已花费 ${sum(p['total_spent'] for p in llm_selected):.2f}，剩余预算 ${remaining_budget:.2f}")
            
            # 从候选商品中排除LLM已选的商品，避免重复选择
            llm_selected_names = {p['name'] for p in llm_selected}
            remaining_candidates = [c for c in candidates if c['name'] not in llm_selected_names]
            logger.info(f"小类 {subcategory}: 排除LLM已选商品，剩余候选商品 {len(remaining_candidates)} 个")
        else:
            selected = []
            remaining_budget = budget
            remaining_candidates = candidates
            logger.info(f"小类 {subcategory}: 使用纯贪心算法选择商品")
        
        target_utilization = budget * 0.8
        
        # 简单的贪心选择（在剩余候选商品中选择）
        candidates_sorted = sorted(remaining_candidates, key=lambda x: x['price'])
        
        for candidate in candidates_sorted:
            if remaining_budget >= candidate['price']:
                max_qty = min(8, int(remaining_budget / candidate['price']))
                if max_qty > 0:
                    quantity = max(1, min(max_qty, int(target_utilization / (candidate['price'] * len(candidates_sorted)))))
                    
                    # 🔧 修复：优先使用 candidate 中的 product_id
                    product_id = candidate.get('product_id') or candidate.get('id')
                    if not product_id:
                        product_id = self.find_product_id_by_name(candidate['name'], self.df)
                    firm_id = self.find_firm_id_by_name(product_id, self.economic_center) if product_id else None
                    
                    selected.append({
                        'name': candidate['name'],
                        'price': candidate['price'],
                        'quantity': quantity,
                        'total_spent': round(candidate['price'] * quantity, 2),
                        'product_id': product_id,
                        'owner_id': firm_id
                    })
                    remaining_budget -= candidate['price'] * quantity
                    
                    if budget - remaining_budget >= target_utilization:
                        break
        
        # 计算最终利用率
        final_spent = sum(p['total_spent'] for p in selected)
        final_utilization = final_spent / budget if budget > 0 else 0
        
        if llm_selected:
            logger.info(f"小类 {subcategory}: 贪心补充完成，总花费 ${final_spent:.2f}，利用率 {final_utilization:.1%}")
        else:
            logger.info(f"小类 {subcategory}: 纯贪心选择完成，总花费 ${final_spent:.2f}，利用率 {final_utilization:.1%}")
        
        return selected
    
    async def _fallback_individual_product_selection(self, category: str, subcategory_budgets: Dict[str, float],
                                             family_profile: str, current_month: int, topn: int, family_id: str = None) -> Dict[str, List[Dict]]:
        """回退到单独处理每个小类，如果LLM失败则使用纯贪心算法"""
        results = {}
        
        for subcategory, budget in subcategory_budgets.items():
            candidates = self._collect_candidates_for_subcategory(category, subcategory, budget, topn, family_id=family_id)
            
            try:
                # 使用原有的LLM单独选择逻辑
                selected = await llm_utils.llm_score_products(
                    candidates, budget, subcategory, family_profile=family_profile
                )
                
                # 处理结果格式
                processed_selection = []
                for item in selected:
                    if isinstance(item, dict) and 'name' in item and 'price' in item and 'quantity' in item:
                        # 🆕 优先通过 (name, owner_id) 匹配
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
                            product_id = matching_candidate.get('product_id') or matching_candidate.get('id')
                        else:
                            product_id = None
                        
                        # 如果没有找到，再尝试通过名称查找
                        if not product_id:
                            product_id = self.find_product_id_by_name(item['name'], self.df)
                        
                        # 🆕 优先从LLM返回中获取owner_id，其次从候选商品中获取
                        result_owner_id = item.get('owner_id') or (matching_candidate.get('owner_id', '') if matching_candidate else '')
                        
                        # 🆕 如果仍然没有owner_id，尝试通过product_id查找（作为最后手段）
                        if not result_owner_id:
                            result_owner_id = self.find_firm_id_by_name(product_id, self.economic_center) if product_id else None
                        
                        processed_selection.append({
                            'name': item['name'],
                            'price': item['price'],
                            'quantity': item['quantity'],
                            'total_spent': round(item['price'] * item['quantity'], 2),
                            'product_id': product_id,
                            'owner_id': result_owner_id  # 🆕 使用获取到的owner_id
                        })
                
                results[subcategory] = processed_selection
                logger.info(f"小类 {subcategory}: LLM单独处理成功")
                
            except Exception as e:
                logger.error(f"小类 {subcategory} LLM单独处理失败: {e}，使用纯贪心算法")
                # LLM失败时，使用纯贪心算法（不传递LLM结果）
                results[subcategory] = self._generate_fallback_selection(subcategory, budget, candidates)
        
        return results

    async def allocate_subcategory_budget_to_products_hierarchical_batch(self, subcategory_budget: Dict[str, Union[float, Dict[str, float]]], 
                                                                 family_profile: str, current_month: int, topn=20, max_workers=64, ex_info=None, family_id: str = None) -> Dict[str, Union[float, Dict[str, List[Dict]]]]:
        """
        【方案A：分层批量处理】将小类预算分配到具体商品
        - 同一大类内的小类使用批量LLM处理
        - 不同大类之间使用并发处理
        - 避免信息丢失，提高处理速度
        
        Args:
            subcategory_budget: 小类预算分配，格式为 {category: {subcategory: budget}} 或 {category: budget}
            family_profile: 家庭画像信息
            current_month: 当前月份（1-12）
            topn: 每个小类的候选商品数量
            max_workers: 最大工作线程数
            
        Returns:
            Dict[str, Union[float, Dict[str, List[Dict]]]]: 商品分配结果
        """
        if not family_profile:
            raise ValueError("family_profile must be provided!")

        if ex_info:
            family_profile = ex_info + "\n " + family_profile
        
        if not subcategory_budget:
            return {}
        
        logger.info(f"开始分层批量处理小类预算到商品分配，共{len(subcategory_budget)}个大类")
        
        # 第1步：按大类分组，区分有二级子类和无二级子类的情况
        category_groups = {}  # 有二级子类的大类
        no_subcat_results = {}  # 无二级子类的大类（直接返回预算）
        
        for category, allocation in subcategory_budget.items():
            if isinstance(allocation, dict):
                # 有二级子类的情况
                if allocation:  # 确保不是空字典
                    category_groups[category] = allocation
            else:
                # 没有二级子类的情况，直接返回预算金额
                no_subcat_results[category] = allocation
                logger.info(f"大类 {category} 无二级子类，直接分配预算 ${allocation:.2f}")
        
        if not category_groups:
            logger.info("所有大类都无二级子类，直接返回预算分配")
            return no_subcat_results
        
        # 第2步：并发处理每个大类
        async def process_one_category(category_data):
            category, subcategory_budgets = category_data
            try:
                logger.info(f"开始处理大类 {category}，包含 {len(subcategory_budgets)} 个小类")
                
                # 为该大类的所有小类进行批量商品选择
                category_results = await self._batch_select_products_for_category(
                    category=category,
                    subcategory_budgets=subcategory_budgets,
                    family_profile=family_profile,
                    current_month=current_month,
                    topn=topn,
                    family_id=family_id
                )
                
                return (category, category_results, True)
                
            except Exception as e:
                logger.error(f"大类 {category} 分层批量处理失败: {e}")
                return (category, {}, False)
        
        # 使用并发处理所有大类
        final_results = {}
        successful_categories = 0
        failed_categories = 0
        
        # 使用asyncio并发处理所有大类（不再限制并发数，由全局LLM信号量控制）
        # 创建并发任务
        category_tasks = [
            process_one_category((category, budgets)) 
            for category, budgets in category_groups.items()
        ]
        
        # 并发执行所有任务
        category_results = await asyncio.gather(*category_tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(category_results):
            category = list(category_groups.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"大类 {category} 处理时发生异常: {result}")
                final_results[category] = {}
                failed_categories += 1
            else:
                try:
                    category_name, results, success = result
                    final_results[category_name] = results
                    
                    if success:
                        successful_categories += 1
                        logger.info(f"大类 {category_name} 分层批量处理成功")
                    else:
                        failed_categories += 1
                        logger.warning(f"大类 {category_name} 分层批量处理失败")
                        
                except Exception as e:
                    logger.error(f"解析大类 {category} 结果时发生异常: {e}")
                    final_results[category] = {}
                    failed_categories += 1
        
        # 第3步：合并无二级子类的结果
        final_results.update(no_subcat_results)
        
        # 第4步：统计和日志
        total_products = 0
        total_spending = 0.0
        
        for category, category_data in final_results.items():
            if isinstance(category_data, dict) and category_data:
                for subcategory, products in category_data.items():
                    if isinstance(products, list):
                        total_products += len(products)
                        total_spending += sum(p.get('total_spent', 0) for p in products)
        
        logger.info(f"LLM选择完成 - 成功: {successful_categories}, 失败: {failed_categories}, 总商品数: {total_products}, 总花费: ${total_spending:.2f}")
        
        # 第5步：全局属性补充
        final_results = self._apply_global_supplement(final_results, subcategory_budget, family_profile)
        
        # 重新统计（补充后）
        total_products_final = 0
        total_spending_final = 0.0
        
        for category, category_data in final_results.items():
            if isinstance(category_data, dict) and category_data:
                for subcategory, products in category_data.items():
                    if isinstance(products, list):
                        total_products_final += len(products)
                        total_spending_final += sum(p.get('total_spent', 0) for p in products)
        
        logger.info(f"分层批量处理完成 - 最终商品数: {total_products_final}, 最终花费: ${total_spending_final:.2f}")
        
        return final_results

    
    async def generate_current_month_products(self, family_id: str, current_month: int, 
                                      subcategory_budget: Dict[str, Union[float, Dict[str, float]]],
                                      family_profile: str = None) -> Dict[str, Union[float, Dict[str, List[Dict]]]]:
        """
        为当前月小类生成商品的专用函数，参考build_monthly_shopping_plan函数的逻辑
        
        Args:
            family_id: 家庭ID
            current_month: 当前月份（1-12）
            subcategory_budget: 小类预算分配，格式为 {category: {subcategory: budget}} 或 {category: budget}
            family_profile: 家庭画像信息，如果为None则自动获取
            
        Returns:
            Dict[str, Union[float, Dict[str, List[Dict]]]]: 商品分配结果
                - 有二级子类的大类: {category: {subcategory: [product_list]}}
                - 没有二级子类的大类: {category: budget}
        """
        try:
            # 如果没有提供家庭画像，则自动获取
            if family_profile is None:
                family_profile = self._get_family_profile_for_budget_calculation(family_id)
            
            logger.info(f"开始为家庭{family_id}第{current_month}月生成商品清单")
            
            # 调用allocate_subcategory_budget_to_products函数
            result = await self.allocate_subcategory_budget_to_products(
                subcategory_budget=subcategory_budget,
                family_profile=family_profile,
                current_month=current_month,
                topn=20,
                max_workers=32
            )
            
            logger.info(f"家庭{family_id}第{current_month}月商品清单生成完成")
            return result
            
        except Exception as e:
            logger.error(f"为家庭{family_id}第{current_month}月生成商品清单失败: {e}")
            # 返回空结果
            return {}

    async def allocate_subcategory_budget_to_products(self, subcategory_budget: Dict[str, Union[float, Dict[str, float]]], 
                                              family_profile: str, current_month: int, topn=20, max_workers=64) -> Dict[str, Union[float, Dict[str, List[Dict]]]]:
        """
        将小类预算分配到具体商品，参考build_monthly_shopping_plan函数的逻辑
        
        Args:
            subcategory_budget: 小类预算分配，格式为 {category: {subcategory: budget}} 或 {category: budget}
            family_profile: 家庭画像信息
            current_month: 当前月份（1-12）
            topn: 检索候选商品数量
            max_workers: 最大工作线程数
            
        Returns:
            Dict[str, Union[float, Dict[str, List[Dict]]]]: 商品分配结果
                - 有二级子类的大类: {category: {subcategory: [product_list]}}
                - 没有二级子类的大类: {category: budget}
        """
        if not family_profile:
            raise ValueError("family_profile must be provided!")
        
        if not subcategory_budget:
            return {}
        
        logger.info(f"开始将小类预算分配到具体商品，共{len(subcategory_budget)}个类别")
        
        # 构建任务列表
        tasks = []
        for category, allocation in subcategory_budget.items():
            if isinstance(allocation, dict):
                # 有二级子类的情况
                for subcategory, budget in allocation.items():
                    if budget > 0:
                        tasks.append((category, subcategory, budget))
            else:
                # 没有二级子类的情况，跳过（直接返回预算金额）
                continue
        
        def adjust_selection_to_budget(selected_items: list, budget: float) -> list:
            """
            严格预算调整器：确保总花费不超过预算。
            如果超支，会按性价比（优先保留低价商品）调整数量或移除商品。
            """
            total_spent = sum(item.get('price', 0) * item.get('quantity', 1) for item in selected_items)
            if total_spent <= budget:
                return selected_items

            logger.info(f"[预算调整] 开始调整，当前花费 {total_spent:.2f} > 预算 {budget:.2f}")
            
            # 按价格从高到低排序，优先调整最贵的商品
            sorted_items = sorted(selected_items, key=lambda x: x.get('price', 0), reverse=True)
            
            while total_spent > budget and sorted_items:
                item_to_adjust = sorted_items[0]
                price = item_to_adjust.get('price', 0)
                
                # 如果减少一个数量后仍在预算内，则减少数量
                if total_spent - price <= budget:
                    item_to_adjust['quantity'] -= 1
                    if item_to_adjust['quantity'] <= 0:
                        sorted_items.pop(0) # 如果数量为0，则移除
                else:
                    # 否则直接移除最贵的商品
                    sorted_items.pop(0)
                
                total_spent = sum(item.get('price', 0) * item.get('quantity', 1) for item in sorted_items)

            logger.info(f"[预算调整] 调整后花费: {total_spent:.2f}")
            return sorted_items

        async def process_one_subcategory(args):
            category, subcategory, budget = args
            logger.info(f"正在处理: 大类: {category} - 小类: {subcategory} - 预算: {budget:.2f}...")
            
            # 如果预算太小，跳过
            if budget < 10:
                logger.info(f"[跳过] 小类{subcategory}预算过小({budget:.2f})，跳过处理")
                return (category, subcategory, [])
            
            query_text = f"{subcategory}"
            candidates = []
            
            # 方案1: 使用ProductMarket进行向量检索（本地同步调用）
            try:
                # 直接使用ProductMarket的search_products方法（本地同步调用）
                products = self._search_products_sync(
                    query=query_text, 
                    top_k=topn * 5
                    # must_contain=subcategory
                )
                
                # 转换为候选商品格式，并进行价格过滤
                price_range_min = budget * 0.01  # 最小单价：预算的1%
                price_range_max = budget * 0.8   # 最大单价：预算的80%
                
                reasonable_candidates = []
                other_candidates = []
                
                for product in products:
                    # 检查价格是否有效
                    if pd.isna(product.price) or product.price <= 0:
                        continue
                        
                    candidate = {
                        "name": product.name, 
                        "price": float(product.price),
                        "product_id": getattr(product, 'product_id', '')
                    }
                    if price_range_min <= product.price <= price_range_max:
                        reasonable_candidates.append(candidate)
                    else:
                        other_candidates.append(candidate)
                
                # 优先使用合理价格范围的商品，不足时补充其他商品
                candidates = reasonable_candidates[:20] + other_candidates[:10]
                candidates = candidates[:25]  # 最多25个候选商品
                            
                logger.info(f"[向量检索] 小类{subcategory}找到{len(candidates)}个候选商品(合理价格:{len(reasonable_candidates)}, 其他:{len(other_candidates)})")
            except Exception as e:
                logger.warning(f"[向量检索异常] {e}")
                candidates = []
            
            # 方案2: 直接用商品库过滤（作为补充或备用）
            if len(candidates) < 8:
                logger.info(f"[备用方案] 小类{subcategory}向量检索候选商品不足({len(candidates)})，尝试直接用商品库过滤...")
                try:
                    if hasattr(self.df, 'columns') and 'level1' in self.df.columns:
                        # 精确匹配
                        subcat_products = self.df[self.df['level1'].str.lower() == subcategory.strip().lower()]
                        
                        # 如果精确匹配不够，尝试模糊匹配
                        if len(subcat_products) < 15:
                            fuzzy_products = self.df[
                                self.df['level1'].str.lower().str.contains(subcategory.strip().lower(), na=False) |
                                self.df['level1'].str.lower().str.contains(subcategory.strip().lower().replace(' ', ''), na=False)
                            ]
                            subcat_products = pd.concat([subcat_products, fuzzy_products]).drop_duplicates()
                    else:
                        subcat_products = self.df
                    
                    # 价格过滤 - 更宽松的价格范围
                    subcat_products = subcat_products[subcat_products['List Price'] <= budget * 1.2]
                    subcat_products = subcat_products[subcat_products['List Price'] >= budget * 0.005]  # 避免过于便宜的商品
                    subcat_products = subcat_products[subcat_products['List Price'] > 0]
                    
                    # 补充候选商品
                    existing_names = {c['name'] for c in candidates}
                    for _, item in subcat_products.head(30).iterrows():
                        if item["Product Name"] not in existing_names:
                            # 获取product_id，优先从数据中读取，如果没有则通过名称查找
                            product_id = item.get("product_id", "") or self.find_product_id_by_name(item["Product Name"], self.df)
                            owner_id = item.get("owner_id", "") or item.get("company_id", "")
                            # 🆕 查询实时价格
                            real_time_price = self._get_real_time_price(
                                product_id=product_id,
                                product_name=item["Product Name"],
                                owner_id=owner_id
                            )
                            # 如果查询失败，使用CSV价格作为fallback
                            price = real_time_price if real_time_price is not None else item["List Price"]
                            
                            candidates.append({
                                "name": item["Product Name"], 
                                "price": price,  # ✅ 使用实时价格
                                "product_id": product_id,
                                "owner_id": owner_id  # 🆕 添加公司ID
                            })
                            if len(candidates) >= 30:  # 最多30个候选商品
                                break
                                
                    logger.info(f"[商品库补充] 小类{subcategory}现有{len(candidates)}个候选商品")
                except Exception as e:
                    logger.warning(f"[商品库过滤异常] {e}")
                    pass
            
            # 方案3: 最后的备用方案 - 从同一大类下的其他小类借用商品
            if len(candidates) < 5:
                logger.info(f"[最后备用] 小类{subcategory}候选商品仍不足({len(candidates)})，从同大类其他小类借用...")
                try:
                    same_category_subcats = BudgetConfig.BUDGET_TO_WALMART_MAIN.get(category, [])
                    for other_subcat in same_category_subcats:
                        if other_subcat != subcategory and len(candidates) < 15:
                            other_products = self.df[self.df['level1'].str.lower() == other_subcat.strip().lower()]
                            other_products = other_products[other_products['List Price'] <= budget * 1.2]
                            other_products = other_products[other_products['List Price'] >= budget * 0.01]
                            existing_names = {c['name'] for c in candidates}
                            for _, item in other_products.head(8).iterrows():
                                if item["Product Name"] not in existing_names:
                                    # 获取product_id，优先从数据中读取，如果没有则通过名称查找
                                    product_id = item.get("product_id", "") or self.find_product_id_by_name(item["Product Name"], self.df)
                                    owner_id = item.get("owner_id", "") or item.get("company_id", "")
                                    # 🆕 查询实时价格
                                    real_time_price = self._get_real_time_price(
                                        product_id=product_id,
                                        product_name=item["Product Name"],
                                        owner_id=owner_id
                                    )
                                    # 如果查询失败，使用CSV价格作为fallback
                                    price = real_time_price if real_time_price is not None else item["List Price"]
                                    
                                    candidates.append({
                                        "name": item["Product Name"], 
                                        "price": price,  # ✅ 使用实时价格
                                        "product_id": product_id,
                                        "owner_id": owner_id  # 🆕 添加公司ID
                                    })
                                    if len(candidates) >= 15:
                                        break
                    logger.info(f"[同类借用] 小类{subcategory}现有{len(candidates)}个候选商品")
                except Exception as e:
                    logger.warning(f"[同类借用异常] {e}")
                    pass
            
            if not candidates:
                logger.warning(f"[警告] 小类{subcategory}最终无候选商品，跳过。")
                return (category, subcategory, [])
            
            # 按价格多样性排序候选商品，确保有不同价位的选择
            candidates = sorted(candidates, key=lambda x: (abs(x['price'] - budget/10), x['price']))[:20]
            logger.info(f"[最终] 小类{subcategory}准备送LLM，候选商品数: {len(candidates)}, 预算: {budget:.2f}")
            
            selected = []
            # LLM 挑选商品和数量
            
            selected = await llm_utils.llm_score_products(
                candidates, budget, subcategory, family_profile=family_profile
            )
            
            # 增强的备用方案：确保预算利用率
            if not selected:
                logger.info(f"[LLM失败备用] 为小类{subcategory}启用增强贪心策略")
                selected = []
                remaining_budget = budget
                target_utilization = budget * 0.8  # 目标80%利用率
                
                # 选择多个不同价位的商品
                candidates_by_price = sorted(candidates, key=lambda x: x['price'])
                low_price = candidates_by_price[:len(candidates_by_price)//3]    # 低价位
                mid_price = candidates_by_price[len(candidates_by_price)//3:2*len(candidates_by_price)//3]  # 中价位
                high_price = candidates_by_price[2*len(candidates_by_price)//3:]  # 高价位
                
                # 分配策略：先选一些中价位商品作为主力，再补充低价位商品
                for candidate_group in [mid_price, low_price, high_price]:
                    for candidate in candidate_group[:3]:  # 每个价位最多3个商品
                        if remaining_budget >= candidate['price']:
                            max_qty = min(12, int(remaining_budget / candidate['price']))
                            if max_qty > 0:
                                # 根据预算和价格确定合理数量
                                if candidate['price'] < budget * 0.1:  # 便宜商品买多点
                                    quantity = min(max_qty, max(2, int(budget * 0.15 / candidate['price'])))
                                else:  # 贵商品买少点
                                    quantity = min(max_qty, max(1, int(budget * 0.25 / candidate['price'])))
                                
                                selected.append({
                                    'name': candidate['name'],
                                    'price': candidate['price'],
                                    'quantity': quantity
                                })
                                remaining_budget -= candidate['price'] * quantity
                                
                                # 如果已达到目标利用率，可以停止
                                current_spent = budget - remaining_budget
                                if current_spent >= target_utilization:
                                    break
                    
                    # 检查是否已达到目标
                    current_spent = budget - remaining_budget
                    if current_spent >= target_utilization:
                        break
            
            # --- 核心修改：增加严格的预算后处理器 ---
            final_selection = adjust_selection_to_budget(selected, budget)

            result = []
            for item in final_selection:
                if isinstance(item, dict):
                    price = item.get('price', 0)
                    quantity = item.get('quantity', 1) if 'quantity' in item else 1
                    name = item.get('name', '')
                elif isinstance(item, str):
                    match = next((c for c in candidates if c['name'] == item), None)
                    if not match:
                        match = next((c for c in candidates if item.lower() in c['name'].lower()), None)
                    price = match['price'] if match else 0
                    quantity = 1
                    name = item
                else:
                    price = 0
                    quantity = 1
                    name = str(item)
                
                if price > 0 and quantity > 0:  # 只添加有效商品
                    total_spent = round(price * quantity, 2)
                    
                    # 🔧 修复：优先从候选列表中获取 product_id
                    matching_candidate = next((c for c in candidates if c['name'] == name), None)
                    if matching_candidate:
                        product_id = matching_candidate.get('product_id') or matching_candidate.get('id')
                    else:
                        product_id = None
                    
                    # 如果没有找到，再尝试通过名称查找
                    if not product_id:
                        product_id = self.find_product_id_by_name(name, self.df)
                    firm_id = self.find_firm_id_by_name(product_id, self.economic_center) if product_id else None
                    
                    result.append({
                        'name': name,
                        'price': price,
                        'quantity': quantity,
                        'total_spent': total_spent,
                        'product_id': product_id,  # 新增：添加product_id字段
                        'owner_id': firm_id # 新增：添加owner_id字段
                    })
            
            actual_spent = sum(x['total_spent'] for x in result)
            utilization_rate = actual_spent / budget if budget > 0 else 0
            logger.info(f"完成: 大类: {category} - 小类: {subcategory} (预算: {budget:.2f}, 实际花费: {actual_spent:.2f}, 利用率: {utilization_rate:.1%}, 商品数: {len(result)})")
            return (category, subcategory, result)

        # 使用asyncio并发处理
        result_products = {}
        
        # 限制并发数量
        semaphore = asyncio.Semaphore(min(max_workers, 64))
        
        async def limited_subcategory_task(args):
            async with semaphore:
                return await process_one_subcategory(args)
        
        # 创建并发任务
        subcategory_tasks = [limited_subcategory_task(task) for task in tasks]
        
        # 并发执行所有任务
        results = await asyncio.gather(*subcategory_tasks, return_exceptions=True)
        
        # 整理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                category, subcategory, budget = task
                logger.error(f"处理小类 {category}-{subcategory} 时发生异常: {result}")
                if category not in result_products:
                    result_products[category] = {}
                result_products[category][subcategory] = []
            else:
                category, subcategory, selected = result
                if category not in result_products:
                    result_products[category] = {}
                result_products[category][subcategory] = selected
        
        # 处理没有二级子类的情况
        for category, allocation in subcategory_budget.items():
            if not isinstance(allocation, dict):
                # 没有二级子类的情况，直接返回预算金额
                result_products[category] = allocation
                logger.info(f"类别 {category} 没有二级子类，直接分配预算 ${allocation:.2f}")
        
        logger.info(f"小类预算分配完成，共处理 {len(result_products)} 个类别")
        return result_products
    
    def find_firm_id_by_name(self, product_id: str, economic_center=None) -> str:
        """
        通过product_id在商品库中精确匹配对应的firm_id
        
        竞争市场模式：
        - 如果有多个供应商，选择价格最低的
        - 价格相同则随机选择
        
        Args:
            product_id: 商品ID
            economic_center: EconomicCenter实例（用于查询价格，竞争模式需要）
        
        Returns:
            选定的 firm_id，匹配失败返回 None
        """
        try:
            matched = self.pro_firm_df[self.pro_firm_df['product_id'] == product_id]['company_id'].values
            if len(matched) == 0:
                logger.warning(f"🔍 未找到product_id={product_id}对应的firm_id，返回None")
                return None
            
            # 只有一个供应商，直接返回
            if len(matched) == 1:
                return matched[0]
            
            # 🔥 多个供应商（竞争模式）：根据价格选择
            if economic_center is None:
                # 如果没有提供economic_center，随机选择（兼容旧代码）
                import random
                return random.choice(matched)
            
            # 查询各供应商的价格和库存
            available_suppliers = []
            for company_id in matched:
                try:
                    # 从economic_center查询该供应商的商品信息
                    price = ray.get(economic_center.query_price.remote(company_id, product_id))
                    if price > 0:
                        available_suppliers.append({
                            'company_id': company_id,
                            'price': price
                        })
                except Exception as e:
                    logger.warning(f"查询供应商 {company_id} 价格失败: {e}")
            
            if not available_suppliers:
                # 所有供应商都缺货，返回第一个
                return matched[0]
            
            # 80%概率选择价格低的供应商，20%概率选择价格高的供应商
            import random
            if random.random() < 0.8:
                min_price = min(s['price'] for s in available_suppliers)
                best_suppliers = [s for s in available_suppliers if s['price'] == min_price]
            else:
                max_price = max(s['price'] for s in available_suppliers)
                best_suppliers = [s for s in available_suppliers if s['price'] == max_price]
            
            selected = random.choice(best_suppliers)
            
            return selected['company_id']
            
        except Exception as e:
            logger.error(f"🔍 查找firm_id时出错:")
            logger.error(f"🔍   product_id: {product_id}")
            logger.error(f"🔍   异常类型: {type(e).__name__}")
            logger.error(f"🔍   异常信息: {str(e)}")
            logger.error(f"🔍   pro_firm_df shape: {self.pro_firm_df.shape if hasattr(self.pro_firm_df, 'shape') else 'unknown'}")
            return None
    
    def _extract_family_profile_dict(self, family_profile: Union[str, Dict]) -> Dict:
        """
        从family_profile提取字典格式
        
        Args:
            family_profile: 家庭画像（字符串或字典）
            
        Returns:
            Dict: 包含family_size等信息的字典
        """
        if isinstance(family_profile, dict):
            return family_profile
        
        # 如果是字符串，尝试提取family_size
        try:
            if family_profile and "family_size" in str(family_profile):
                import re
                match = re.search(r'family[_\s]size[:\s]+(\d+)', str(family_profile), re.IGNORECASE)
                if match:
                    return {'family_size': int(match.group(1))}
        except Exception as e:
            logger.debug(f"Failed to extract family_size from profile: {e}")
        
        # 默认返回单人家庭
        return {'family_size': 3}

    def _calculate_and_save_attributes(
        self, 
        family_id: str, 
        current_month: int, 
        shopping_plan: Dict[str, Any],
        family_profile: Union[str, Dict]
    ):
        """
        【已废弃】根据购物计划计算属性值，并更新保存家庭属性
        
        注意：此方法已废弃，属性更新逻辑已迁移到 Household 类中。
        现在由 household.py 的 update_attributes_after_purchase() 方法完成。
        保留此方法仅用于向后兼容。
        
        Args:
            family_id: 家庭ID
            current_month: 当前月份
            shopping_plan: 购物计划（包含所有待购商品）
            family_profile: 家庭画像
        """
        logger.warning("⚠️ _calculate_and_save_attributes 已废弃，属性更新应由 Household 完成")
        return  # 直接返回，不执行
        try:
            if not self.attribute_manager:
                logger.warning("属性管理器未初始化，跳过属性计算")
                return
            
            # 1. 获取家庭月初属性值（上个月末的值）
            # 注意：第1个月必须从0开始，第N个月(N>1)从第N-1个月末开始
            if current_month == 1:
                # 第1个月：初始化为0
                base_attrs = self.attribute_manager.config.get("base_consumption", {})
                current_attributes = {attr: 0.0 for attr in base_attrs.keys()}
            else:
                # 第N个月(N>1)：获取第N-1个月末的属性值
                previous_month = current_month - 1
                current_attributes = self.attribute_manager.get_family_current_attributes(
                    family_id, str(previous_month)
                )
            
            # 2. 解析家庭画像
            profile_dict = self._extract_family_profile_dict(family_profile)
            
            # 3. 计算月度消耗
            monthly_consumption = self.attribute_manager.calculate_family_attribute_needs(
                family_id, profile_dict, str(current_month)
            )
            
            # 4. 计算本月商品供给的属性值
            monthly_supply = {attr: 0.0 for attr in current_attributes.keys()}
            
            # 遍历购物计划，累加每个商品的属性值
            product_count = 0
            for category, subcategories in shopping_plan.items():
                if not isinstance(subcategories, dict):
                    continue
                    
                for subcategory, products in subcategories.items():
                    if not isinstance(products, list):
                        continue
                    
                    for product_item in products:
                        if not isinstance(product_item, dict):
                            continue
                        
                        product_id = product_item.get("product_id", "")
                        product_name = product_item.get("name", "")
                        quantity = float(product_item.get("quantity", 0))
                        
                        if quantity <= 0:
                            continue
            
            # 5. 计算新的属性值 = 当前值 + 供给 - 消耗
            new_attributes = {}
            for attr in current_attributes.keys():
                current_value = current_attributes.get(attr, 0.0)
                supply_value = monthly_supply.get(attr, 0.0)
                consumption_value = monthly_consumption.get(attr, 0.0)
                new_value = max(0.0, current_value + supply_value - consumption_value)
                new_attributes[attr] = new_value
            
            # 6. 保存到文件
            self.attribute_manager._save_family_attributes(
                family_id=family_id,
                current_month=str(current_month),
                new_attributes=new_attributes,
                monthly_consumption=monthly_consumption,
                product_supply=monthly_supply,
                family_profile=profile_dict
            )
            
            # 7. 记录日志
            total_supply = sum(monthly_supply.values())
            total_consumption = sum(monthly_consumption.values())
            logger.info(
                f"家庭 {family_id} 第 {current_month} 月属性更新完成 | "
                f"商品数: {product_count}, 供给: {total_supply:.2f}, 消耗: {total_consumption:.2f}"
            )
            
        except Exception as e:
            logger.error(f"计算和保存属性值时出错: {e}")
            import traceback
            traceback.print_exc()

    
    def find_classification_by_product_id(self, product_id: str) -> str:
        """
        通过product_id在商品库中精确匹配对应的classification（daily_cate），匹配失败返回None
        """
        try:
            if hasattr(self.df, 'columns') and 'daily_cate' in self.df.columns:
                matches = self.df[self.df['Uniq Id'] == product_id]
                if not matches.empty:
                    classification = matches.iloc[0]['daily_cate']
                    return classification if pd.notna(classification) else None
            return None
        except Exception as e:
            print(f"查找商品分类失败 (product_id={product_id}): {e}")
            return None
    
    async def batch_allocate(
        self, 
        household_contexts: List[Dict], 
        current_month: int,
        batch_size: int = 20
    ) -> Dict[str, Dict]:
        """
        ✨ 批量预算分配：将多个家庭的预算请求合并处理
        
        Args:
            household_contexts: 家庭上下文信息列表，每项包含：
                - household_id: 家庭ID
                - balance: 余额
                - last_month_income: 上月收入
                - ex_info: 就业信息
                - family_profile: 家庭画像
            current_month: 当前月份
            batch_size: 每批处理的家庭数量
            
        Returns:
            Dict[household_id, budget_result]: 每个家庭的预算分配结果
        """
        import asyncio
        from agentsociety_ecosim.consumer_modeling import llm_utils
        
        results = {}
        total_contexts = len(household_contexts)
        
        # 分批处理
        for batch_start in range(0, total_contexts, batch_size):
            batch_end = min(batch_start + batch_size, total_contexts)
            batch_contexts = household_contexts[batch_start:batch_end]
            batch_num = batch_start//batch_size + 1
            total_batches = (total_contexts + batch_size - 1)//batch_size
            
            batch_timer = time.time()
            print(f"   批次 {batch_num}/{total_batches} (家庭 {batch_start+1}-{batch_end})...", end=" ", flush=True)
            
            # 构建批量请求的prompt
            batch_prompt = self._build_batch_budget_prompt(batch_contexts, current_month)
            prompt_length = len(batch_prompt)
            print(f"[Prompt: {prompt_length} chars]", end=" ", flush=True)
            
            try:
                # 使用全局LLM信号量控制并发
                llm_semaphore = self.get_global_llm_semaphore()
                async with llm_semaphore:
                    # 调用LLM进行批量预算分配（增加超时时间）
                    import os
                    from openai import AsyncOpenAI
                    
                    # 创建临时客户端，使用更长的超时时间
                    batch_client = AsyncOpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                        base_url=os.getenv("BASE_URL", ""),
                        timeout=120.0  # 120秒超时，适配批量请求
                    )
                    
                    llm_response = await batch_client.chat.completions.create(
                        model=os.getenv("MODEL", ""),
                        messages=[
                            {"role": "system", "content": "You are a professional financial planner. Process multiple household budgets efficiently."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        temperature=0.1,
                        stream=False
                    )
                    response = llm_response.choices[0].message.content.strip()
                    
                    batch_duration = time.time() - batch_timer
                    print(f"✅ {batch_duration:.1f}秒", flush=True)
                    
                    # 解析批量响应
                    batch_results = self._parse_batch_budget_response(response, batch_contexts)
                    
                    # 为每个家庭生成完整的预算结果
                    for ctx in batch_contexts:
                        household_id = ctx["household_id"]
                        budget_data = batch_results.get(household_id, {})
                        
                        if budget_data:
                            # 生成shopping_plan（简化版，不再调用LLM）
                            shopping_plan = await self._generate_shopping_plan_from_budget(
                                budget_data.get("category_budget", {}),
                                ctx
                            )
                            
                            results[household_id] = {
                                "category_budget": budget_data.get("category_budget", {}),
                                "shopping_plan": shopping_plan,
                                "total_budget": budget_data.get("total_budget", 0),
                                "batch_mode": True
                            }
                        else:
                            # 失败的家庭使用默认预算
                            results[household_id] = self._get_default_budget(ctx)
                            
            except Exception as e:
                batch_duration = time.time() - batch_timer
                print(f"❌ 失败 ({batch_duration:.1f}秒): {str(e)[:50]}", flush=True)
                logger.error(f"批量预算分配失败 (批次 {batch_num}): {e}")
                # 失败时为该批次所有家庭使用默认预算
                for ctx in batch_contexts:
                    results[ctx["household_id"]] = self._get_default_budget(ctx)
        
        return results
    
    def _build_batch_budget_prompt(self, batch_contexts: List[Dict], current_month: int) -> str:
        """构建批量预算分配的prompt（简化版，减少token）"""
        
        # 简化类别列表，只显示关键类别
        key_categories = ["food_expenditure", "housing_expenditure", "transportation_expenditure", 
                         "utilities_expenditure", "healthcare_expenditure", "clothing_expenditure"]
        
        prompt = f"""Allocate budgets for {len(batch_contexts)} households. Categories: {', '.join(key_categories)} + others.

Data (Balance|Income|Size|Kids):
"""
        
        # 大幅简化每个家庭的信息（一行显示）
        for i, ctx in enumerate(batch_contexts, 1):
            hid = ctx["household_id"]
            bal = ctx.get("balance", 0)
            inc = ctx.get("last_month_income", 0)
            profile = ctx.get("family_profile", {})
            size = profile.get('family_size', 2)
            kids = profile.get('num_children', 0)
            
            prompt += f"{i}.{hid}|${bal:.0f}|${inc:.0f}|{size}p|{kids}k\n"
        
        prompt += f"""
Return JSON array [{{"household_id":"...","total_budget":0,"category_budget":{{"food_expenditure":0,...}}}},...]. ONLY JSON, no text.
"""
        return prompt
    
    def _parse_batch_budget_response(self, response: str, batch_contexts: List[Dict]) -> Dict[str, Dict]:
        """解析批量预算响应"""
        try:
            # 清理响应内容
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # 解析JSON数组
            budget_array = json.loads(cleaned_response)
            
            # 转换为字典
            results = {}
            for budget_data in budget_array:
                household_id = budget_data.get("household_id")
                if household_id:
                    results[household_id] = budget_data
            
            return results
            
        except Exception as e:
            logger.error(f"解析批量预算响应失败: {e}")
            return {}
    
    async def _generate_shopping_plan_from_budget(self, category_budget: Dict, context: Dict) -> List[Dict]:
        """
        根据预算生成购物计划（简化版，基于规则而非LLM）
        
        Returns:
            List[Dict]: 格式为 [{"category": "...", "products": []}, ...]
                       符合 execute_budget_based_purchases 的期望格式
        """
        shopping_plan = []
        
        # 为每个有预算的类别生成简单的商品列表
        for category, budget in category_budget.items():
            if budget > 0:
                shopping_plan.append({
                    "category": category,
                    "subcategory": category,  # 简化：子类别与类别相同
                    "budget": budget,
                    "products": []  # 空列表，购买阶段会根据预算搜索商品
                })
        
        return shopping_plan
    
    def _get_default_budget(self, context: Dict) -> Dict:
        """获取默认预算（当LLM失败时使用）"""
        balance = context.get("balance", 0)
        income = context.get("last_month_income", 0)
        
        # 使用简单的80/20规则
        total_budget = max(0, balance * 0.8 + income * 0.2)
        
        # 使用固定比例分配
        default_ratios = {
            "food_expenditure": 0.25,
            "housing_expenditure": 0.30,
            "transportation_expenditure": 0.15,
            "utilities_expenditure": 0.08,
            "healthcare_expenditure": 0.07,
            "clothing_expenditure": 0.05,
            "education_expenditure": 0.05,
            "other_recreation_expenditure": 0.05
        }
        
        category_budget = {
            cat: total_budget * ratio
            for cat, ratio in default_ratios.items()
        }
        
        return {
            "category_budget": category_budget,
            "shopping_plan": {},
            "total_budget": total_budget,
            "default_mode": True
        }


# ---------------- Example usage ----------------
async def main():
    """异步主函数"""
    # 配置日志
    logger.basicConfig(
        level=logger.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logger.StreamHandler(),  # 输出到终端
            logger.FileHandler("consumer_decision.log", encoding='utf-8')  # 输出到文件
        ]
    )

    product_market = None
    allocator = None
    
    try:
        # ========== 🚀 启用向量检索（Level2改进方案完整版）==========
        print("\n" + "=" * 80)
        print(" " * 20 + "🚀 初始化向量检索系统")
        print("=" * 80)
        
        # 1. 初始化Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            print("✅ Ray初始化完成")
        else:
            print("✅ Ray已经初始化")
        
        # 2. 创建或复用ProductMarket实例（使用Named Actor）
        try:
            from agentsociety_ecosim.center.assetmarket import ProductMarket
            
            # 尝试获取已存在的 Actor
            try:
                product_market = ray.get_actor("product_market_instance")
                print("✅ 复用已有的 ProductMarket Actor")
            except ValueError:
                # 不存在，创建新的命名 Actor
                product_market = ProductMarket.options(
                    name="product_market_instance",
                    lifetime="detached"  # 脱离创建进程，可跨进程复用
                ).remote()
                print("✅ 创建新的 ProductMarket Actor")
            
            print("   - 向量库地址: http://localhost:6333")
            print("   - 集合名称: part_products")
            print("   - 向量数量: 29120")
            print("   - Embedding模型: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"❌ ProductMarket初始化失败: {e}")
            print("⚠️  将使用Fallback方案（直接从商品库匹配）")
            product_market = None
        
        # 3. 创建BudgetAllocator，传入product_market
        allocator = BudgetAllocator(product_market=product_market)
        print("✅ BudgetAllocator初始化完成（已启用向量检索）")
        print("=" * 80 + "\n")

        print("=" * 60)
        print("开始批量测试：家庭1-10，月份1-12；总余额=2021年年消费总额；月收入=年消费/12")
        print("=" * 60)

        for fid in range(1,3):
            try:
                family_info = get_family_consumption_and_profile_by_id(fid)
                if not family_info:
                    print(f"家庭{fid} 数据缺失，跳过")
                    continue
                consumption = family_info.get("consumption", {}) or {}
                year_key = "2021"
                if year_key in consumption:
                    year_data = consumption.get(year_key, {}) or {}
                else:
                    # 回退到最近一年
                    year_data = get_latest_expenditures_by_family_id(fid) or {}
                    print(f"家庭{fid} 未找到2021年数据，使用最近一年消费代替")

                total_expenditure = 0.0
                for v in year_data.values():
                    try:
                        total_expenditure += float(v)
                    except Exception:
                        continue

                total_balance = total_expenditure
                current_income = total_expenditure / 12.0 if total_expenditure > 0 else 0.0
                family_profile = family_info.get("family_profile")

                print(f"家庭{fid}: 年消费={total_expenditure:.2f} | 月收入={current_income:.2f}")

                for m in range(1, 5):
                    try:
                        print(f"  -> 运行 月份{m} ...")
                        _ = await allocator.allocate_with_metrics(
                            family_id=str(fid),
                            current_month=m,
                            current_income=current_income,
                            total_balance=total_balance,
                            family_profile=family_profile,
                            max_workers=32,
                        )
                    except Exception as e:
                        print(f"  月份{m} 运行失败: {e}")
            except Exception as e:
                print(f"家庭{fid} 处理失败: {e}")

        print("测试完成。")
    
    finally:
        # 清理资源
        print("\n" + "=" * 80)
        print(" " * 25 + "🧹 清理资源")
        print("=" * 80)
        
        # 注意：使用 Named Actor 且 lifetime="detached" 时，
        # Actor 会在 Ray 集群中持久化，可以被后续调用复用。
        # 如果这是最后一次运行，需要手动清理，可以取消下面的注释：
        
        # if product_market:
        #     try:
        #         ray.kill(product_market)
        #         print("✅ ProductMarket Actor 已清理")
        #     except Exception as e:
        #         print(f"⚠️  清理 ProductMarket 时出错: {e}")
        
        # 如果需要完全关闭 Ray（测试脚本结束时）：
        # ray.shutdown()
        # print("✅ Ray 已关闭")
        
        print("ℹ️  ProductMarket Actor 保持运行中（可复用）")
        print("ℹ️  如需完全清理，请手动执行: ray.shutdown()")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
