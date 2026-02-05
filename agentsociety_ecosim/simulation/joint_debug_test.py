#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentSociety经济仿真系统主运行脚本
直接运行经济仿真，不进行测试
"""

# 🔥 关键：必须在导入torch之前设置CUDA设备
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 使用空闲的GPU 4（GPU 0-3已满载）

# 加载环境变量 - 必须在其他导入之前
from dotenv import load_dotenv
load_dotenv()

import asyncio
import time
import json
# import psutil
import ray
import sys
import os
from typing import List, Dict, Any, Optional, DefaultDict
from dataclasses import dataclass, asdict
from datetime import datetime, date
import pytz
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentsociety_ecosim.utils.log_utils import setup_global_logger
from agentsociety_ecosim.simulation.monthly_visualization import MonthlyVisualization
from agentsociety_ecosim.simulation.industry_competition_analyzer import IndustryCompetitionAnalyzer
from agentsociety_ecosim.simulation.innovation_exporter import InnovationDataExporter
from agentsociety_ecosim.agent.firm import Firm
from agentsociety_ecosim.agent.government import Government
from agentsociety_ecosim.agent.household import Household
from agentsociety_ecosim.agent.bank import Bank
from agentsociety_ecosim.center.ecocenter import EconomicCenter
from agentsociety_ecosim.center.assetmarket import ProductMarket
from agentsociety_ecosim.center.jobmarket import LaborMarket
from agentsociety_ecosim.center.model import Job, TaxPolicy
from agentsociety_ecosim.utils.data_loader import *
from agentsociety_ecosim.utils.select_firms import reduce_products_and_update_map

from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import torch

# 为 MCP 服务器设置：如果环境变量 MCP_MODE 存在，强制使用 CPU
if os.getenv('MCP_MODE'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
    device = torch.device("cpu")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 使用GPU 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))
model = AutoModel.from_pretrained(os.getenv("MODEL_PATH")).to(device)

# 注意：Qdrant 客户端由 ProductMarket Actor 管理，主进程不需要初始化
# 如果需要在主进程中使用向量搜索，请使用远程 Qdrant 服务器
print("Using local Qdrant storage: /home/chenjiaju/AgentEconomist/agentsociety_ecosim/data/qdrant_data")

# 设置日志
logger = setup_global_logger(name="economic_simulation", log_dir="logs", level="INFO")

std_job = load_jobs()
job_dis = load_job_dis()

@dataclass
class SimulationConfig:
    """仿真配置类"""
    # 系统规模配置 - 初始化所有企业，不限制数量
    num_households: int = 100      # 测试：5个家庭
    num_iterations: int = 12     # 测试：4个月
    experiment_name: Optional[str] = None  # 允许外部指定实验目录名称
    experiment_output_dir: Optional[str] = None  # 记录实验输出目录（可选）
    
    # 再分配策略配置
    redistribution_strategy: str = "none"  # 可选: "none", "equal", "income_proportional", "poverty_focused", "unemployment_focused", "family_size", "mixed"
    redistribution_poverty_weight: float = 0.3  # 贫困权重 (0-1)
    redistribution_unemployment_weight: float = 0.2  # 失业权重 (0-1)
    redistribution_family_size_weight: float = 0.1  # 家庭规模权重 (0-1)
    
    # 性能配置
    max_concurrent_tasks: int = 100
    
    max_llm_concurrent: int = 400  
    
    # ✨ 批量LLM优化配置
    use_batch_budget_allocation: bool = False  # ❌ 关闭批量模式（实测：10家庭82秒，比并发更慢）
    batch_size: int = 10  # 每批处理的家庭数量（减小以避免超时，建议5-10）
    batch_llm_timeout: int = 120  # 批量LLM请求的超时时间（秒）
    
    # 企业处理并发限制
    max_firm_concurrent: int = 50
    
    # 税率配置（与Government的TaxPolicy保持一致）
    income_tax_rate: float = 0.225  # 22.5% 个人所得税
    vat_rate: float = 0.08  # 8% 消费税（增值税）
    corporate_tax_rate: float = 0.21  # 21% 企业所得税（用于参考）
    
    # 🏭 生产与补货配置
    # ✨ 新版：基于利润和成本的生产系统
    profit_to_production_ratio: float = 0.9  # 利润转化为生产预算的比例（70%）
    min_production_per_product: float = 5.0  # 每个商品最小生产量
    
    # 劳动力生产函数参数（柯布-道格拉斯生产函数）
    # Q = A × L^α (Q=产出, A=全要素生产率, L=劳动力, α=劳动力弹性)
    labor_productivity_factor: float = 200.0  # A: 全要素生产率/基础效率因子
    labor_elasticity: float = 0.7  # α: 劳动力弹性系数 (0-1之间，越接近1劳动力影响越大)
    
    # [已废弃] 旧版基础生产参数（保留以防回退）
    base_production_rate: float = 100.0  # 每个产品基础补货量（单位/月）
    high_demand_multiplier: float = 1.5  # 高需求商品补货倍数
    low_demand_multiplier: float = 0.7  # 低需求商品补货倍数
    
    # 监控配置
    monitor_interval: float = 5.0  # 5秒监控一次
    enable_monitoring: bool = False
    
    # 辞退系统配置
    dismissal_rate: float = 0.1  # 每月辞退比例 (10%)
    enable_dismissal: bool = False  # ✅ 启用辞退功能
    
    # 工作发布配置
    enable_dynamic_job_posting: bool = True  # ✅ 启用动态招聘
    first_month_job_rate: float = 0.9  # 第一个月发布工作比例
    unemployment_threshold: float = 0.4  # 失业率阈值，超过此值时发布新工作
    job_posting_multiplier: float = 0.1  # 工作发布倍数，基于失业人数

    # 公司-商品数量配置
    min_per_cat: int = 20        # 每类最少20个商品
    multiplier: int = 12     # 使用全部商品：实际将加载所有可用商品（约29,000个）
    random_state: int = 42
    amount: Dict[str, float] = None

    
    # 🔥 企业竞争模式配置（创新破坏理论）
    enable_competitive_market: bool = True  # 是否启用竞争市场模式（同类企业销售相同商品）

    # 💰 商品价格调整配置
    enable_price_adjustment: bool = False # 是否启用价格根据销量自动调整
    price_adjustment_rate: float = 0.1    # 价格调整幅度 (10%)
    
    # 🛒 固有市场配置 (解决商品积压问题)
    enable_inherent_market: bool = True  # 是否启用固有市场
    inherent_market_consumption_rate: float = 0.30  # 每月消耗商品的比例 (30%)
    inherent_market_focus_new_products: bool = True  # 是否优先消耗新生产的商品
    
    # 💰 商品毛利率配置 (基于Daily Category的12个大类)
    # 毛利率 = (售价 - 成本) / 售价 × 100%
    # 以下配置将用于计算企业的成本和利润
    category_profit_margins: Dict[str, float] = None  # 各大类的毛利率配置，将在__post_init__中初始化
    
    # 创新部分
    enable_innovation_module: bool = True        # 是否启用创新模块
    innovation_gamma: float = 1.3                 # 创新成功后的质量/产量阶梯 γ
    policy_encourage_innovation: bool = True     # 政策是否鼓励创新
    innovation_lambda: float = 0.9              # 单位创新到达强度 λ
    innovation_concavity_beta: float = 0.6        # 研发有效劳动凹性 β (0<β≤1)
    innovation_research_share: float = 0.1        # 鼓励创新企业的研发投入比例（10%利润用于研发）

    def __post_init__(self):
        """初始化后处理，设置默认的毛利率配置"""
        if self.category_profit_margins is None:
            # 基于Daily Category的12个大类的毛利率配置（单位：%）
            # 由GPT-5生成，基于行业实际情况和市场竞争程度
            self.category_profit_margins = {
                "Beverages": 25.0,                              # 饮料
                "Confectionery and Snacks": 32.0,               # 糖果和零食
                "Dairy Products": 15.0,                         # 乳制品
                "Furniture and Home Furnishing": 30.0,          # 家具和家居装饰
                "Garden and Outdoor": 28.0,                     # 园艺和户外
                "Grains and Bakery": 18.0,                      # 谷物和烘焙
                "Household Appliances and Equipment": 30.0,     # 家用电器和设备
                "Meat and Seafood": 16.0,                       # 肉类和海鲜
                "Personal Care and Cleaning": 40.0,             # 个人护理和清洁
                "Pharmaceuticals and Health": 45.0,             # 药品和健康
                "Retail and Stores": 25.0,                      # 零售和商店
                "Sugars, Oils, and Seasonings": 20.0,           # 糖类、油类和调料
            }
        """初始化后处理，设置默认的商品数量配置"""
        if self.amount is None:
            self.amount = {
                'food_amount': 800,
                'non_food_amount': 400
            }
    
@dataclass
class SystemMetrics:
    """系统指标类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float

@dataclass
class SimulationMetrics:
    """仿真指标类"""
    iteration: int
    timestamp: float
    active_firms: int
    active_households: int
    total_jobs_posted: int
    total_jobs_matched: int
    total_consumption: float
    total_income: float
    iteration_duration: float

@dataclass
class HouseholdMonthlyMetrics:
    """家庭月度指标类"""
    household_id: str
    month: int
    monthly_income: float
    monthly_redistribution_amount:float
    monthly_expenditure: float
    savings_rate: float
    consumption_structure: Dict[str, float]
    household_labor_hours: int
    household_employees: int
    current_savings: float
    income_change_rate: float = 0.0

    
@dataclass
class FirmMonthlyMetrics:
    """企业月度指标类"""
    company_id: str  # 统一使用 company_id，与 Firm 类保持一致
    month: int
    monthly_revenue: float
    monthly_expenses: float
    monthly_profit: float
    current_employees: int
    job_postings: int
    successful_hires: int
    recruitment_success_rate: float

@dataclass
class PerformanceMetrics:
    """性能监控指标类"""
    timestamp: float
    operation_type: str
    agent_id: str
    duration: float
    
@dataclass
class LLMMetrics:
    """LLM调用指标类"""
    timestamp: float
    agent_type: str
    input_tokens: int
    output_tokens: int
    api_call_duration: float
    success: bool

class EconomicSimulation:
    """经济仿真主类"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.economic_center = None
        self.product_market = None
        self.labor_market = None
        self.government = None
        self.bank = None
        self.households = []
        self.firms = []
        self.metrics_history = []
        self.economic_metrics_history = []  # 新增：经济指标历史记录
        self.household_monthly_metrics = {}  # 家庭月度指标
        self.firm_monthly_metrics = []  # 企业月度指标
        self.performance_metrics = []  # 性能监控指标
        self.llm_metrics = []  # LLM调用指标
        self.initial_household_savings = {}  # 记录初始储蓄用于财富差距分析
        self.monthly_dismissal_stats = {}  # 月度辞退统计
        self.monitoring_task = None
        self.is_monitoring = False
        
        self.current_month:int = 0
        self._wrapper = None  # 将由wrapper设置，用于钩子调用
        # 新增：基尼系数和平均工资历史
        self.gini_history = []  # 存储每月的基尼系数数据
        self.wage_history = []  # 存储每月的平均工资数据
        
        # 新增：月度详细统计数据
        self.monthly_unemployment_stats = {}  # 每月失业人员统计
        self.monthly_vacant_jobs = {}  # 每月空缺岗位统计
        self.monthly_firm_revenue = {}  # 每月企业收入
        self.monthly_product_sales = {}  # 每月商品销量
        self.monthly_product_inventory = {}  # 每月商品库存数量
        self.monthly_product_prices = {}  # 每月商品价格
        self.monthly_firm_operation_rate = {}  # 每月企业营业率
        self.monthly_supply_demand = {}  # 每月供需数据
        self.monthly_product_restock = {}  # 每月商品补货数据
        self.household_purchase_records = {}  # 家庭每月购买记录 {month: [purchase_record, ...]}
        self.monthly_production_stats = {}  # 每月生产统计数据 {month: production_stats}
        
        # 实验名称（允许通过配置覆盖，否则基于家庭数、月数和时间生成）
        if getattr(self.config, "experiment_name", None):
            self.experiment_name = self.config.experiment_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"exp_{self.config.num_households}h_{self.config.num_iterations}m_{timestamp}"
            self.config.experiment_name = self.experiment_name
        
        # 统一的输出目录（允许配置覆盖，以便在 YAML 中记录）
        configured_output_dir = getattr(self.config, "experiment_output_dir", None)
        if configured_output_dir:
            self.experiment_output_dir = configured_output_dir.rstrip("/\\")
        else:
            self.experiment_output_dir = f"output/{self.experiment_name}"
        self.config.experiment_output_dir = self.experiment_output_dir
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "RAY_DEBUG": "1",
                        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
                                        "BASE_URL": os.getenv("BASE_URL", ""),
                        "MODEL": os.getenv("MODEL", ""),
                    }
                }
            )

        # 🆕 初始化行业竞争分析器（输出到实验目录）
        # 注意：experiment_name 在初始化时已确定，但输出目录会在首次使用时创建
        self.competition_analyzer = None  # 延迟初始化，在首次使用时设置正确的输出目录

        # 🆕 初始化创新数据导出器（输出到实验目录）
        self.innovation_exporter = None  # 延迟初始化，在首次使用时设置正确的输出目录

        logger.info("经济仿真系统初始化完成")
    
    def get_profit_margin_by_category(self, category: str) -> float:
        """
        根据商品大类获取毛利率（仅用于利润计算）
        
        Args:
            category: 商品大类名称（daily_cate）
            
        Returns:
            毛利率（百分比，如25.0表示25%）
        """
        # 如果配置中有该大类，返回配置的毛利率
        if category in self.config.category_profit_margins:
            return self.config.category_profit_margins[category]
        
        # 如果找不到该大类，返回默认毛利率25%
        logger.warning(f"未找到大类 '{category}' 的毛利率配置，使用默认值25%")
        return 25.0
    
    def calculate_profit_from_revenue(self, revenue: float, category: str) -> float:
        """
        根据销售收入和商品大类计算利润
        
        公式：利润 = 销售收入 × 毛利率
        
        Args:
            revenue: 销售收入（售价 × 销量）
            category: 商品大类名称（daily_cate）
            
        Returns:
            利润金额
        """
        margin_rate = self.get_profit_margin_by_category(category) / 100.0
        profit = revenue * margin_rate
        return profit
    
    def set_wrapper(self, wrapper):
        """设置包装器引用"""
        self._wrapper = wrapper
    
    async def setup_simulation_environment(self):
        """设置仿真环境"""
        logger.info("开始设置仿真环境...")
        
        try:
            # 初始化核心组件（传入税率配置）
            self.economic_center = EconomicCenter.remote(
                income_tax_rate=self.config.income_tax_rate,
                vat_rate=self.config.vat_rate,
                corporate_tax_rate=self.config.corporate_tax_rate,
                category_profit_margins=self.config.category_profit_margins
            )
            self.product_market = ProductMarket.remote()
            self.labor_market = LaborMarket.remote()
            
            # 初始化政府（从config创建TaxPolicy）
            tax_policy = TaxPolicy(
                income_tax_rate=self.config.income_tax_rate,
                corporate_tax_rate=self.config.corporate_tax_rate,
                vat_rate=self.config.vat_rate
            )
            self.government = Government.remote(
                government_id="gov_main_simulation",
                initial_budget=10000000.0,
                tax_policy=tax_policy,
                economic_center=self.economic_center
            )
            await self.government.initialize.remote()
            
            # 初始化银行
            self.bank = Bank.remote(
                bank_id="bank_main_simulation",
                initial_capital=1000000.0,
                economic_center=self.economic_center
            )
            await self.bank.initialize.remote()
            logger.info("银行系统初始化完成")
            
            # 加载数据
            logger.info("加载仿真数据...")
            
            # 设置全局LLM并发限制（在创建家庭之前）
            from agentsociety_ecosim.consumer_modeling.consumer_decision import BudgetAllocator
            BudgetAllocator.set_global_llm_limit(self.config.max_llm_concurrent)
            
            # 创建家庭
            await self._create_households()
            
            # 创建企业
            await self._create_firms()
            
            # 验证创建结果
            if len(self.households) == 0:
                logger.error("没有成功创建任何家庭")
                return False
            
            if len(self.firms) == 0:
                logger.error("没有成功创建任何企业")
                return False
            
            return True

        except Exception as e:
            logger.error(f"仿真环境设置失败: {e}")
            return False
    
    async def _create_households(self):
        """创建仿真家庭"""
        logger.info("创建仿真家庭...")
        
        households_dict = load_households()
        household_keys = list(households_dict.keys())[:self.config.num_households]
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def create_household(key):
            async with semaphore:
                try:
                    household_id = key
                    labor_hours = load_lh(household_id, households_dict[key])
                    
                    household = Household(
                        household_id=household_id,
                        economic_center=self.economic_center,
                        labor_hour=labor_hours,
                        labormarket=self.labor_market,
                        product_market=self.product_market,
                        income_tax_rate=self.config.income_tax_rate,
                        vat_rate=self.config.vat_rate
                    )
                    
                    await household.initialize()
                    return household
                except Exception as e:
                    logger.warning(f"创建家庭 {key} 失败: {e}")
                    return None
        
        tasks = [create_household(key) for key in household_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.households = [h for h in results if h is not None]
        logger.info(f"成功创建 {len(self.households)} 个家庭")
        
        # 记录初始储蓄
        print("📊 正在记录家庭初始储蓄...")
        for household in self.households:
            try:
                household_id = household.household_id
                initial_savings = await household.get_balance_ref()
                self.initial_household_savings[household_id] = initial_savings
                # print(f"   家庭 {household_id}: 初始储蓄 ${initial_savings:.2f}")
            except Exception as e:
                logger.warning(f"获取家庭 {household_id if 'household_id' in locals() else '未知'} 初始储蓄失败: {e}")
        
        print(f"✅ 已记录 {len(self.initial_household_savings)} 个家庭的初始储蓄")

    async def _create_firms(self):
        """创建仿真企业（优化版：支持缓存复用）"""
        logger.info("创建仿真企业...")
        
        import os
        
        # 🔄 确定使用的文件名
        if self.config.enable_competitive_market:
            map_file = 'data/company_product_map_competitive.csv'
            mode_name = "竞争市场"
        else:
            map_file = 'data/company_product_map.csv'
            mode_name = "独占市场"
        
        reduced_map_file = 'data/company_product_map_rescaled.csv'
        products_file = 'data/products.csv'
        
        # 🔍 检查是否所有必要文件都存在
        all_files_exist = all(os.path.exists(f) for f in [map_file, reduced_map_file, products_file])
        
        if all_files_exist:
            # ✅ 所有文件都存在，直接读取复用
            logger.info(f"✅ 发现缓存文件，直接复用 ({mode_name}模式)")
            logger.info(f"   - {map_file}")
            logger.info(f"   - {reduced_map_file}")
            logger.info(f"   - {products_file}")
            
            new_map_reduced = pd.read_csv(reduced_map_file)
            products = pd.read_csv(products_file)
            firms_df = load_firms_df()
            
            logger.info(f"📊 已加载: {len(new_map_reduced)} 个企业-商品映射, {len(products)} 个商品")
        else:
            # ⚙️ 文件不存在，执行完整初始化流程
            logger.info(f"⚙️  缓存文件不完整，执行完整初始化流程 ({mode_name}模式)")
            
            products = load_products()
            firms_df = load_firms_df()

            # 🔥 根据配置选择商品分配模式
            if self.config.enable_competitive_market:
                from agentsociety_ecosim.utils.data_loader import allocate_products_competitive
                logger.info("🔥 使用竞争市场模式：同类企业销售相同商品（创新破坏理论）")
                new_map = allocate_products_competitive(products, firms_df, self.config.random_state)
            else:
                logger.info("📦 使用独占市场模式：不同企业销售不同商品")
                new_map = allocate_products(products, firms_df, self.config.random_state)

            _, new_map_reduced, _ = reduce_products_and_update_map(
                products=products,
                new_map=new_map,
                households=self.config.num_households,
                category_col="daily_cate",
                price_col="price",        
                min_per_cat=self.config.min_per_cat,
                multiplier=self.config.multiplier,
                random_state=self.config.random_state,
            )
            
            # 获取有效商品
            valid_pids = set(new_map_reduced['product_id'].unique())
            products = products[products['Uniq Id'].isin(valid_pids)].copy()
            
            # 过滤价格为0的商品
            if 'List Price' in products.columns:
                products = products[products['List Price'] > 0].copy()
            if 'price' in products.columns:
                products = products[products['price'] > 0].copy()
            
            logger.info(f"过滤后剩余有效商品: {len(products)} 个")
            
            # 💾 保存所有文件供下次复用
            products.to_csv(products_file, index=False)
            logger.info(f"💾 已保存缓存文件:")
            logger.info(f"   - {reduced_map_file}")
            logger.info(f"   - {products_file}")
            logger.info(f"   提示: 删除这些文件可重新生成配置")
        
        # 📦 基于 reduced map 初始化企业
        firm2product = (
            new_map_reduced.groupby('company_id')['product_id']
               .apply(list)
               .to_dict()
        )
        available_company_ids = set(firm2product.keys())
        
        # 如果 firms_df 还未加载（缓存路径），加载它
        if 'firms_df' not in locals():
            firms_df = load_firms_df()
        
        firms_df = firms_df[firms_df['factset_entity_id'].isin(available_company_ids)].copy()

        logger.info(f"初始化有效企业: {len(firms_df)} 家")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def create_firm(record):
            async with semaphore:
                cid = record.get('factset_entity_id')
                try:
                    # 先检查产品，避免创建没有产品的公司
                    prod_ids = firm2product.get(cid, [])
                    if not prod_ids:
                        logger.warning(f"[跳过] 公司 {cid} 未分配到任何产品")
                        return None

                    firm_products = products[products['Uniq Id'].isin(prod_ids)].copy()
                    if firm_products.empty:
                        logger.warning(f"[跳过] 公司 {cid} 的产品在 products 中不存在（可能列名或ID不一致）")
                        return None
                    
                    # 产品检查通过后才创建和初始化公司
                    kwargs = Firm.parse_dicts(record)
                    firm = Firm(**kwargs, 
                              economic_center=self.economic_center, 
                              product_market=self.product_market) 
                    
                    await firm.initialize()

                    # 加载企业产品（不再需要 client 参数，由 ProductMarket Actor 管理）
                    await load_products_firm(firm, firm_products, firm2product, 
                                     self.config.amount, self.economic_center, self.product_market, 
                                     model, tokenizer)  
                    
                    return firm
                except Exception as e:
                    logger.warning(f"创建企业失败: {e}")
                    return None
        
        records = firms_df.to_dict(orient='records')
        tasks = [create_firm(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.firms = [f for f in results if f is not None]
        logger.info(f"成功创建 {len(self.firms)} 家企业")

        # 🆕 初始化行业竞争分析器（如果还未初始化）
        if self.competition_analyzer is None:
            competition_output_dir = os.path.join(self.experiment_output_dir, "industry_competition")
            self.competition_analyzer = IndustryCompetitionAnalyzer(
                output_dir=competition_output_dir,
                economic_center=self.economic_center,
                use_timestamp=False  # 不使用时间戳，使用实验名称
            )

        # 🆕 注册行业-企业映射关系（用于竞争分析）
        self.competition_analyzer.register_industry_firms(self.firms)
        logger.info("✅ 行业竞争分析器注册完成")


        await self._assign_innovation_strategies()
        logger.info("✅ 创新策略分配完成")



    async def _assign_innovation_strategies(self):
        """
        为每个行业的两家竞争企业分配创新策略

        🔬 创新破坏理论实验设置:
        - 两家企业都鼓励创新,形成真正的创新竞争
        - 可以观察创新到达对市场份额的影响
        - 创新成功的企业应该在下个月获得更大的市场份额
        """
        logger.info("🔬 开始分配创新策略...")

        # 按行业分组企业
        industry_firms = DefaultDict(list)
        for firm in self.firms:
            industry = firm.main_business  # 使用 main_business 作为行业分类
            industry_firms[industry].append(firm)

        encouraged_count = 0

        # 为每个行业分配策略
        if self.config.policy_encourage_innovation == True:
            for industry, firms_list in industry_firms.items():
                if len(firms_list) != 2:
                    # 跳过非竞争行业
                    logger.debug(f"行业 {industry} 只有 {len(firms_list)} 家企业，跳过创新策略分配")
                    continue

                # 从配置中获取 research_share
                fund_share = self.config.innovation_research_share if hasattr(self.config, 'innovation_research_share') else 0.1

                # 🆕 两家企业都鼓励创新,体现创新破坏理论
                firm1 = firms_list[0]
                firm2 = firms_list[1]

                # 注册到 EconomicCenter
                await self.economic_center.register_firm_innovation_config.remote(
                    firm1,
                    "encouraged",
                    self.config.labor_productivity_factor,
                    fund_share*2
                )
                await self.economic_center.register_firm_innovation_config.remote(
                    firm2,
                    "encouraged",
                    self.config.labor_productivity_factor,
                    fund_share*2
                )

                encouraged_count += 2

                logger.info(f"   🏭 【{industry}】")
                logger.info(f"      ✅ 鼓励创新: {firm1.company_id} (研发比例: {fund_share * 2:.1%})")
                logger.info(f"      ✅ 鼓励创新: {firm2.company_id} (研发比例: {fund_share:.1%})")

            logger.info(f"📊 创新策略分配完成: {encouraged_count} 家企业全部鼓励创新(创新破坏理论实验)")
            logger.info(f"💡 实验目的: 观察创新到达对市场份额的破坏性影响")

        else:
            for industry, firms_list in industry_firms.items():
                if len(firms_list) != 2:
                    # 跳过非竞争行业
                    logger.debug(f"行业 {industry} 只有 {len(firms_list)} 家企业，跳过创新策略分配")
                    continue
                
                # 从配置中获取 research_share
                fund_share = self.config.innovation_research_share if hasattr(self.config, 'innovation_research_share') else 0.1
                firm1 = firms_list[0]
                firm2 = firms_list[1]
                # 注册到 EconomicCenter
                await self.economic_center.register_firm_innovation_config.remote(
                    firm1,
                    "suppressed",
                    self.config.labor_productivity_factor,
                    fund_share
                )
                
                await self.economic_center.register_firm_innovation_config.remote(
                    firm2,
                    "suppressed",
                    self.config.labor_productivity_factor,
                    fund_share
                )
                
                logger.info(f"   🏭 【{industry}】")
                logger.info(f"      ✅ 抑制创新: {firm1.company_id} (研发比例: {fund_share:.1%})")
                logger.info(f"      ✅ 抑制创新: {firm2.company_id} (研发比例: {fund_share:.1%})")
                
            logger.info(f"📊 创新策略分配完成: {len(industry_firms)} 家企业全部抑制创新")
            logger.info(f"💡 实验目的: 观察创新到达对市场份额的破坏性影响")

    def _calculate_optimal_job_count(self, household_count: int, current_month: int, unemployment_data: Optional[Dict[str, Any]] = None) -> int:
        """
        根据家庭数量、当前月份和失业情况计算最优工作岗位数量
        
        Args:
            household_count: 家庭数量
            iteration: 当前迭代次数（从0开始）
            unemployment_data: 失业数据，包含失业人数等信息
        """
        
        if current_month == 1:
            # 第一个月：基于家庭数量计算岗位数量
            base_jobs = int(household_count * self.config.first_month_job_rate)
            logger.info(f"第1个月岗位数量计算: 家庭数={household_count}, 基础岗位={base_jobs}")
            return base_jobs
            
        else:
            # 第二个月开始：基于配置和失业人数动态调整
            if self.config.enable_dynamic_job_posting and unemployment_data and 'total_labor_force_unemployed' in unemployment_data:
                unemployed_count = unemployment_data['total_labor_force_unemployed']
                total_labor_force = unemployment_data.get('total_labor_force_available', unemployed_count)
                unemployment_rate = unemployed_count / total_labor_force if total_labor_force > 0 else 0.0
                
                # 检查是否达到失业率阈值
                if unemployment_rate >= self.config.unemployment_threshold:
                    # 根据失业人数和配置的倍数计算岗位数量
                    base_jobs = max(1, int(unemployed_count * self.config.job_posting_multiplier))
                    
                    # 限制岗位数量范围，避免过度招聘或招聘不足
                    min_jobs = max(1, int(unemployed_count * 0.05))  # 至少填补5%的失业
                    max_jobs = min(unemployed_count, household_count)  # 不超过失业人数和家庭数
                    
                    optimal_jobs = max(min_jobs, min(base_jobs, max_jobs))
                    
                    logger.info(f"第{current_month}个月动态岗位发布: 失业率={unemployment_rate:.1%} >= 阈值{self.config.unemployment_threshold:.1%}, "
                              f"失业人数={unemployed_count}, 发布倍数={self.config.job_posting_multiplier:.1%}, 最优岗位={optimal_jobs}")
                else:
                    # 失业率未达到阈值，不发布新工作
                    optimal_jobs = 0
                    logger.info(f"第{current_month}个月跳过动态岗位发布: 失业率={unemployment_rate:.1%} < 阈值{self.config.unemployment_threshold:.1%}")
                    
            elif self.config.enable_dynamic_job_posting:
                # 启用了动态发布但没有失业数据
                optimal_jobs = 0
                logger.info(f"第{current_month}个月跳过动态岗位发布: 缺少失业数据")
                
            else:
                return 0
        
        return optimal_jobs
    
    def get_beijing_time(self) -> str:
        """获取北京时间字符串"""
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.now(beijing_tz)
        return beijing_time.strftime('%Y-%m-%d %H:%M:%S')
    
    async def generate_consumption_budget_charts(self, current_month: int):
        """为每个家庭生成当月消费预算饼状图"""
        try:
            # 创建输出目录
            base_output_dir = os.path.join(self.experiment_output_dir, "output_fig")
            
            print(f"\n📊 正在生成第 {current_month} 月消费预算饼状图...")
            
            chart_count = 0
            for household_monthly_metric in self.household_monthly_metrics[current_month]:
                try:
                    household_id = household_monthly_metric.household_id
                    household_consumption_structure = household_monthly_metric.consumption_structure
                    filtered_budget = {k: v for k, v in household_consumption_structure.items() if isinstance(v, (int, float)) and v > 0}

                    if not filtered_budget:
                        print(f"   ⚠️  家庭 {household_id} 第 {current_month} 月无有效消费预算数据")
                        continue
                    
                    # 创建家庭专属目录
                    family_dir = os.path.join(base_output_dir, household_id)
                    os.makedirs(family_dir, exist_ok=True)
                    
                    # 生成饼状图
                    self._create_budget_pie_chart(
                        budget_data=filtered_budget,
                        household_id=household_id,
                        month=current_month,
                        output_path=os.path.join(family_dir, f"第{current_month}月消费预算分布.jpg")
                    )
                    
                    chart_count += 1
                    
                except Exception as e:
                    print(f"   ❌ 家庭 {household_id if 'household_id' in locals() else 'unknown'} 图表生成失败: {e}")
                    continue
            
            print(f"   ✅ 成功生成 {chart_count} 个家庭的消费预算饼状图")
            
        except Exception as e:
            print(f"❌ 消费预算图表生成过程出错: {e}")
    
    def _create_budget_pie_chart(self, budget_data: Dict, household_id: str, month: int, output_path: str):
        """创建消费预算饼状图"""
        try:
            # 准备数据
            categories = list(budget_data.keys())
            amounts = list(budget_data.values())
            total_budget = sum(amounts)
            
            # 创建图表
            plt.figure(figsize=(10, 8))
            
            # 生成颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            # 创建饼状图
            wedges, texts, autotexts = plt.pie(
                amounts, 
                labels=categories,
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%\n(${pct*total_budget/100:.0f})',
                startangle=90,
                textprops={'fontsize': 10}
            )
            
            # 设置标题
            plt.title(f'Household {household_id} - Month {month} Consumption Budget Distribution\nTotal Budget: ${total_budget:.2f}', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # 调整文本样式
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            # 添加图例
            plt.legend(wedges, [f'{cat}: ${amt:.2f}' for cat, amt in zip(categories, amounts)],
                      title="Consumption Categories",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
            
            # 确保饼图是圆形
            plt.axis('equal')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # print(f"   📊 已保存: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 创建饼状图失败 {household_id}: {e}")
            plt.close()  # 确保关闭图表
    
    async def start_monitoring(self):
        """开始系统监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_system())
        logger.info("系统监控已启动")
    
    async def stop_monitoring(self):
        """停止系统监控"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("系统监控已停止")
    
    async def _monitor_system(self):
        """系统监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                logger.info(f"系统状态: CPU={metrics.cpu_percent:.1f}%, "
                           f"内存={metrics.memory_percent:.1f}%, "
                           f"内存使用={metrics.memory_used_gb:.1f}GB")
                
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                await asyncio.sleep(self.config.monitor_interval)
    
    # def _collect_system_metrics(self) -> SystemMetrics:
    #     """收集系统指标"""
    #     cpu_percent = psutil.cpu_percent(interval=0.1)
    #     memory = psutil.virtual_memory()
        
    #     return SystemMetrics(
    #         timestamp=time.time(),
    #         cpu_percent=cpu_percent,
    #         memory_percent=memory.percent,
    #         memory_used_gb=memory.used / (1024**3)
    #     )
    
    def _record_performance_metric(self, operation_type: str, agent_id: str, duration: float):
        """记录性能指标"""
        metric = PerformanceMetrics(
            timestamp=time.time(),
            operation_type=operation_type,
            agent_id=agent_id,
            duration=duration
        )
        self.performance_metrics.append(metric)
        
    def _record_llm_metric(self, agent_type: str, input_tokens: int, output_tokens: int, 
                          duration: float, success: bool):
        """记录LLM调用指标"""
        metric = LLMMetrics(
            timestamp=time.time(),
            agent_type=agent_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_call_duration=duration,
            success=success
        )
        self.llm_metrics.append(metric)
    
    def _calculate_gini_coefficient(self, incomes: List[float]) -> float:
        """计算基尼系数 - 包含所有家庭（包括零收入）以准确反映不平等程度"""
        if not incomes or len(incomes) == 0:
            return 0.0
        
        # 处理负收入（转换为0），但保留零收入
        non_negative_incomes = [max(0.0, income) for income in incomes]
        
        # 排序
        sorted_incomes = sorted(non_negative_incomes)
        n = len(sorted_incomes)
        
        # 如果所有收入都是0，基尼系数为0（完全平等）
        total_income = sum(sorted_incomes)
        if total_income == 0:
            return 0.0
        
        # 计算基尼系数 - 使用标准公式
        cumsum = 0
        for i, income in enumerate(sorted_incomes):
            cumsum += (2 * (i + 1) - n - 1) * income
        
        gini = cumsum / (n * total_income)
        return max(0.0, min(1.0, gini))  # 确保在[0,1]范围内
    
    async def _collect_monthly_metrics(self, month: int, households: List, firms: List, job_postings: int):
        """收集月度指标数据"""
        logger.info(f"收集第 {month} 个月的指标数据...")
        
        try:
            # 收集家庭月度数据 - 并行处理
            print(f"📊 开始并行收集 {len(households)} 个家庭的月度数据...")
            
            async def collect_household_monthly_data(household):
                try:
                    # 并行获取当月和上月数据
                    current_month_task = self.economic_center.compute_household_monthly_stats.remote(
                        household.household_id, month
                    )
                    balance_task = household.get_balance_ref()
                    
                    tasks = [current_month_task, balance_task]
                    
                    # 如果不是第一个月，获取上月数据
                    if month > 1:
                        prev_month_task = self.economic_center.compute_household_monthly_stats.remote(
                            household.household_id, month - 1
                        )
                        tasks.append(prev_month_task)
                    
                    # 并行执行所有任务
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 🔧 修复：解析结果 - compute_household_monthly_stats 返回 (income, expense, balance) 三个值
                    if not isinstance(results[0], Exception) and len(results[0]) >= 2:
                        monthly_income, monthly_expenditure, _ = results[0]
                    else:
                        monthly_income, monthly_expenditure = 0, 0
                    
                    current_balance = results[1] if not isinstance(results[1], Exception) else 0
                    
                    # 计算储蓄率
                    savings_rate = (monthly_income - monthly_expenditure) / monthly_income if monthly_income > 0 else 0
                    
                    # 计算收入变化率
                    income_change_rate = 0.0
                    if month > 1 and len(results) > 2 and not isinstance(results[2], Exception):
                        # 🔧 修复：results[2] 也是 (income, expense, balance) 三个值
                        if len(results[2]) >= 2:
                            prev_income, prev_expense, _ = results[2]
                        else:
                            prev_income = 0
                        
                        if prev_income > 0:
                            income_change_rate = (monthly_income - prev_income) / prev_income
                    
                    # 使用实际的消费预算数据
                    consumption_structure = {}
                    try:
                        # 获取household的实际消费预算数据
                        consume_budget_data = household.get_consume_budget_data()
                        if month in consume_budget_data:
                            consumption_structure = consume_budget_data[month]
                        else:
                            # 如果没有实际数据，使用简化的消费结构作为备选
                            consumption_structure = {
                                "food": monthly_expenditure * 0.25,
                                "housing": monthly_expenditure * 0.30,
                                "transportation": monthly_expenditure * 0.15,
                                "entertainment": monthly_expenditure * 0.10,
                                "clothing": monthly_expenditure * 0.08,
                                "healthcare": monthly_expenditure * 0.07,
                                "education": monthly_expenditure * 0.05
                            }
                    except Exception as e:
                        logger.warning(f"获取家庭 {household.household_id} 第{month}月消费预算失败: {e}")
                        # 使用简化的消费结构作为备选
                        consumption_structure = {
                            "food": monthly_expenditure * 0.25,
                            "housing": monthly_expenditure * 0.30,
                            "transportation": monthly_expenditure * 0.15,
                            "entertainment": monthly_expenditure * 0.10,
                            "clothing": monthly_expenditure * 0.08,
                            "healthcare": monthly_expenditure * 0.07,
                            "education": monthly_expenditure * 0.05
                        }
                    
                    # 计算家庭就业人数
                    household_labor_hours = len(household.labor_hours)
                    household_employees = 0
                    for lh in household.labor_hours:
                        if not lh.is_valid:
                            household_employees += 1

                    # 创建家庭月度指标
                    return HouseholdMonthlyMetrics(
                        household_id=household.household_id,
                        month=month,
                        monthly_income=monthly_income,
                        monthly_expenditure=monthly_expenditure,
                        savings_rate=savings_rate,
                        consumption_structure=consumption_structure,
                        income_change_rate=income_change_rate,
                        household_labor_hours=household_labor_hours,
                        household_employees=household_employees
                    )
                    
                except Exception as e:
                    logger.warning(f"收集家庭 {household.household_id} 月度数据失败: {e}")
                    return None
            
            # 并行收集所有家庭数据
            household_data_tasks = [collect_household_monthly_data(h) for h in households]
            household_metrics = await asyncio.gather(*household_data_tasks, return_exceptions=True)
            
            # 添加有效的指标到列表
            valid_metrics = [metric for metric in household_metrics if metric is not None and not isinstance(metric, Exception)]
            self.household_monthly_metrics[month].extend(valid_metrics)
            
            print(f"✅ 家庭月度数据收集完成: {len(valid_metrics)}/{len(households)} 个家庭数据收集成功")
            
            # 收集企业月度数据
            for firm in firms:
                try:
                    # 获取企业销售收入（这里需要根据实际的企业数据获取方法）
                    # 暂时使用占位数据
                    monthly_revenue = 0.0  # 需要实现获取企业收入的方法
                    
                    # 获取当前员工数
                    current_employees = 0
                    if hasattr(firm, 'employees'):
                        current_employees = len(firm.employees) if firm.employees else 0
                    
                    # 统计本月成功招聘数量
                    successful_hires = 0
                    company_id = self._get_consistent_firm_id(firm)
                    
                    # 从雇佣确认结果中统计该企业的成功招聘数量
                    if hasattr(self, 'confirmed_hires_for_month') and month in self.confirmed_hires_for_month:
                        confirmed_hires = self.confirmed_hires_for_month[month]
                        for hire in confirmed_hires:
                            if hire.get("company_id") == company_id:
                                successful_hires += 1
                    
                    # 如果没有雇佣确认数据，尝试从劳动力市场获取
                    if successful_hires == 0 and hasattr(self, 'labor_market'):
                        try:
                            # 获取该企业的已匹配工作数量（只统计本月的）
                            matched_jobs = await self.labor_market.get_matched_jobs_for_firm.remote(company_id)
                            if matched_jobs:
                                # 只统计本月的匹配工作，避免累积数据
                                current_month_matches = [job for job in matched_jobs if hasattr(job, 'month') and job.month == month]
                                successful_hires = len(current_month_matches) if current_month_matches else 0
                        except Exception as e:
                            logger.debug(f"获取企业 {company_id} 匹配工作数量失败: {e}")
                    
                    # 统计本月实际发布的岗位数量
                    job_postings = 0
                    if hasattr(self, 'labor_market'):
                        try:
                            # 获取该企业的所有开放岗位
                            open_jobs = await self.labor_market.query_jobs.remote(company_id)
                            if open_jobs:
                                # 统计岗位的总可用数量
                                for job in open_jobs:
                                    if hasattr(job, 'positions_available'):
                                        job_postings += job.positions_available
                                    else:
                                        job_postings += 1  # 如果没有positions_available属性，默认为1
                        except Exception as e:
                            logger.debug(f"获取企业 {company_id} 开放岗位数量失败: {e}")
                            # 如果获取失败，使用默认值
                            job_postings = 1
                    else:
                        # 如果没有劳动力市场，使用默认值
                        job_postings = 1
                    
                    recruitment_success_rate = (successful_hires / job_postings) if job_postings > 0 else 0

                    # 创建企业月度指标
                    firm_metric = FirmMonthlyMetrics(
                        company_id=company_id,
                        month=month,
                        monthly_revenue=monthly_revenue,
                        current_employees=current_employees,
                        job_postings=job_postings,  # 使用实际统计的岗位数量
                        successful_hires=successful_hires,
                        recruitment_success_rate=recruitment_success_rate
                    )
                    
                    self.firm_monthly_metrics.append(firm_metric)
                    
                except Exception as e:
                    logger.warning(f"收集企业 {firm.company_id} 月度数据失败: {e}")
            
            logger.info(f"第 {month} 个月指标数据收集完成: {len(households)} 个家庭, {len(firms)} 家企业")
            
        except Exception as e:
            logger.error(f"收集月度指标数据失败: {e}")
    
    async def update_deposit(self):
        for household in self.households:
            savings = await household.get_balance_ref()
            await self.bank.update_deposit.remote(household.household_id, savings)

    async def handle_dismissal(self):
        """基于企业利润的智能辞退逻辑"""
        dismissal_start = time.time()

        if self.config.enable_dismissal and self.current_month > 1:  # 从第二个月开始执行辞退
            print(f"\n🔥 ===== 第 {self.current_month} 月智能辞退阶段 =====")
            try:
                # 1. 收集所有企业的利润数据
                firm_profits = []
                for firm in self.firms:
                    try:
                        monthly_financials = await self.economic_center.query_firm_monthly_financials.remote(firm.company_id, self.current_month - 1)
                        profit = monthly_financials.get("monthly_profit", 0.0)
                        firm_profits.append({
                            'company_id': firm.company_id,
                            'firm': firm,
                            'profit': profit,
                            'employees': getattr(firm, 'employees', 0)
                        })
                    except Exception as e:
                        logger.warning(f"获取企业 {firm.company_id} 利润数据失败: {e}")
                        firm_profits.append({
                            'company_id': firm.company_id,
                            'firm': firm,
                            'profit': 0.0,
                            'employees': getattr(firm, 'employees', 0)
                        })
                
                # 2. 按利润排序：负利润优先，然后按利润从低到高
                firm_profits.sort(key=lambda x: (x['profit'] >= 0, x['profit']))
                
                # 3. 执行辞退策略
                dismissed_count = 0
                firms_to_dismiss = []
                
                # 策略1：优先辞退负利润企业
                negative_profit_firms = [f for f in firm_profits if f['profit'] < 0 and f['employees'] > 0]
                for firm_data in negative_profit_firms:
                    firms_to_dismiss.append(firm_data)
                    dismissed_count += 1
                    print(f"   📉 负利润企业 {firm_data['company_id']}: 利润${firm_data['profit']:.2f}, 员工{firm_data['employees']}人 → 裁员1人")
                
                # 策略2：如果所有企业都盈利，辞退利润最低的5家
                if not negative_profit_firms:
                    positive_profit_firms = [f for f in firm_profits if f['profit'] >= 0 and f['employees'] > 0]
                    # 取利润最低的5家
                    lowest_profit_firms = positive_profit_firms[:5]
                    for firm_data in lowest_profit_firms:
                        firms_to_dismiss.append(firm_data)
                        dismissed_count += 1
                        print(f"   📊 低利润企业 {firm_data['company_id']}: 利润${firm_data['profit']:.2f}, 员工{firm_data['employees']}人 → 裁员1人")
                
                # 4. 执行实际辞退
                if firms_to_dismiss:
                    print(f"\n🔄 开始执行辞退...")
                    dismissal_result = await self.labor_market.dismiss_workers_by_firm.remote(
                        firms_to_dismiss,
                        month=self.current_month
                    )
                    
                    # 处理企业员工数量更新
                    if 'firm_updates' in dismissal_result and dismissal_result['firm_updates']:
                        for company_id, update_info in dismissal_result['firm_updates'].items():
                            firm = self._find_firm_by_id(company_id)
                            if firm:
                                firm.remove_employees(update_info['count'])
                                print(f"   ✅ 企业 {company_id}: 减少 {update_info['count']} 名员工")
                    
                    # 处理家庭状态同步
                    if 'dismissed_workers' in dismissal_result and dismissal_result['dismissed_workers']:
                        print(f"🔄 开始同步 {len(dismissal_result['dismissed_workers'])} 个家庭的辞退状态...")
                        
                        for worker_info in dismissal_result['dismissed_workers']:
                            household_id = worker_info['household_id']
                            lh_type = worker_info['lh_type']
                            company_id = worker_info['company_id']
                            job_soc = worker_info['job_SOC']
                            
                            household = self._find_household_by_id(household_id)
                            
                            # 更新labor_hour状态
                            labor_hour_updated = False
                            for labor_hour in household.labor_hours:
                                if (labor_hour.lh_type == lh_type and 
                                    labor_hour.job_SOC == job_soc and 
                                    labor_hour.company_id == company_id and
                                    not labor_hour.is_valid):
                                    
                                    labor_hour.is_valid = True
                                    labor_hour.company_id = None
                                    labor_hour.job_title = None
                                    labor_hour.job_SOC = None
                                    labor_hour_updated = True
                                    break
                                    
                            # 更新head_job/spouse_job状态
                            if lh_type == 'head':
                                household.head_job = None
                            elif lh_type == 'spouse':
                                household.spouse_job = None
                            
                            if labor_hour_updated:
                                print(f"   ✅ 同步成功: 家庭 {household_id} ({lh_type}) 状态已更新")
                            else:
                                print(f"   ⚠️  同步警告: 家庭 {household_id} ({lh_type}) 未找到匹配的labor_hour")
                        
                        print(f"🔄 家庭状态同步完成")
                    
                    # 打印辞退统计
                    print(f"\n📊 智能辞退统计:")
                    print(f"   目标辞退: {dismissed_count} 人")
                    print(f"   实际辞退: {dismissal_result.get('dismissed_count', 0)} 人")
                    print(f"   重新开放岗位: {dismissal_result.get('jobs_reopened', 0)} 个")
                    
                    # 记录辞退信息
                    self.monthly_dismissal_stats[self.current_month] = {
                        'dismissed_count': dismissal_result.get('dismissed_count', 0),
                        'jobs_reopened': dismissal_result.get('jobs_reopened', 0),
                        'firm_updates': dismissal_result.get('firm_updates', {}),
                        'dismissal_strategy': 'profit_based'
                    }
                else:
                    print(f"📊 无需辞退：所有企业都盈利且员工充足")
                    self.monthly_dismissal_stats[self.current_month] = {
                        'dismissed_count': 0,
                        'jobs_reopened': 0,
                        'firm_updates': {},
                        'dismissal_strategy': 'no_dismissal_needed'
                    }
                
            except Exception as e:
                print(f"❌ 智能辞退过程出错: {e}")
                logger.error(f"智能辞退失败: {e}")

        else:
            if not self.config.enable_dismissal:
                print(f"📊 第 {self.current_month} 月跳过辞退（辞退功能已禁用）")
            else:
                print(f"📊 第 {self.current_month} 月跳过辞退（首月不执行辞退）")
        
        unemployment_data = None
        if self.current_month > 1:  # 第二个月开始收集失业数据
            try:
                # 使用现有的家庭统计逻辑收集失业数据
                unemployment_data = await self._collect_unemployment_data(self.current_month)
                if unemployment_data:
                    unemployed_count = unemployment_data['total_labor_force_unemployed']
                    total_labor_force = unemployment_data.get('total_labor_force_available', unemployed_count)
                    unemployment_rate = unemployed_count / total_labor_force if total_labor_force > 0 else 0.0
                    logger.info(f"第{self.current_month}个月失业统计: 失业人数={unemployment_data['total_labor_force_unemployed']}, 失业率={unemployment_rate:.1%}")
            except Exception as e:
                logger.warning(f"收集失业数据失败: {e}")
                unemployment_data = None

        dismissal_duration = time.time() - dismissal_start
        print(f"✅ 辞退阶段完成 (耗时: {dismissal_duration:.3f}秒)\n")

        return unemployment_data, dismissal_duration

    async def post_jobs(self, unemployment_data):
        """企业发布工作机会，返回(使用的企业列表, 成功发布岗位总数, 耗时秒)"""
        optimal_job_count = self._calculate_optimal_job_count(len(self.households), self.current_month, unemployment_data)
        logger.info(f"仿真迭代 {self.current_month}: 企业发布工作机会，目标岗位数: {optimal_job_count}")

        # 修复整数除法问题：确保岗位能够合理分配
        if optimal_job_count >= len(self.firms):
            # 岗位数 >= 企业数：正常分配
            jobs_per_firm = optimal_job_count // len(self.firms)
            remaining_jobs = optimal_job_count % len(self.firms)
            firms_to_post = self.firms
        else:
            # 岗位数 < 企业数：随机选择部分企业发布岗位
            jobs_per_firm = 1
            remaining_jobs = 0
            # 随机选择要发布岗位的企业
            import random
            firms_to_post = random.sample(self.firms, optimal_job_count)
            logger.info(f"岗位数({optimal_job_count}) < 企业数({len(self.firms)})，随机选择 {len(firms_to_post)} 家企业发布岗位")

        # 计算实际分配情况
        total_allocated_jobs = 0
        firms_with_extra = 0
        for i, firm in enumerate(firms_to_post):
            base_jobs = jobs_per_firm
            if i < remaining_jobs:
                base_jobs += 1
                firms_with_extra += 1
            total_allocated_jobs += base_jobs
        
        logger.info(f"岗位分配方案: 目标{optimal_job_count}个岗位分配给{len(firms_to_post)}家企业")
        logger.info(f"  - {firms_with_extra}家企业各发布{jobs_per_firm + 1}个岗位")
        logger.info(f"  - {len(firms_to_post) - firms_with_extra}家企业各发布{jobs_per_firm}个岗位")
        logger.info(f"  - 总计分配: {total_allocated_jobs}个岗位")

        job_posting_start = time.time()
        print(f"🏢 开始并行处理企业岗位发布...")

        job_posting_semaphore = asyncio.Semaphore(self.config.max_firm_concurrent)

        # 为剩余岗位创建分配方案
        firm_job_counts = {}
        for i, firm in enumerate(firms_to_post):
            base_jobs = jobs_per_firm
            # 前remaining_jobs家企业额外获得1个岗位
            if i < remaining_jobs:
                base_jobs += 1
            firm_job_counts[firm.company_id] = base_jobs

        async def post_jobs_with_limit(firm):
            async with job_posting_semaphore:
                try:
                    t0 = time.time()
                    actual_jobs = firm_job_counts.get(firm.company_id, jobs_per_firm)
                    await firm.define_job_openings(job_dis, std_job, self.labor_market, actual_jobs)
                    duration = time.time() - t0
                    self._record_performance_metric("job_posting", firm.company_id, duration)
                    return actual_jobs
                except Exception as e:
                    logger.warning(f"企业 {firm.company_id} 发布工作失败: {e}")
                    return 0

        posting_tasks = [post_jobs_with_limit(firm) for firm in firms_to_post]
        posting_results = await asyncio.gather(*posting_tasks, return_exceptions=True)

        total_job_postings = sum(r for r in posting_results if isinstance(r, int))
        job_posting_duration = time.time() - job_posting_start
        
        # 验证分配是否正确
        if total_job_postings == optimal_job_count:
            print(f"✅ 岗位发布完成: {len(firms_to_post)} 家企业参与发布, 总计 {total_job_postings} 个岗位 (目标: {optimal_job_count})")
        else:
            print(f"⚠️  岗位发布完成: {len(firms_to_post)} 家企业参与发布, 实际 {total_job_postings} 个岗位 (目标: {optimal_job_count}, 差异: {total_job_postings - optimal_job_count})")
        
        # 显示分配详情
        if remaining_jobs > 0:
            print(f"   📊 分配详情: {firms_with_extra}家企业×{jobs_per_firm + 1}岗位 + {len(firms_to_post) - firms_with_extra}家企业×{jobs_per_firm}岗位 = {total_job_postings}岗位")

        return firms_to_post, total_job_postings, job_posting_duration

    async def match_jobs(self):
        """家庭匹配工作，生成期望薪资，返回(总申请数, 耗时秒)"""
        job_matching_start = time.time()
        logger.info(f"仿真迭代 {self.current_month}: 家庭寻找工作...")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        async def find_jobs(household):
            async with semaphore:
                try:
                    job_search_start = time.time()
                    result = await household.find_jobs()
                    job_search_duration = time.time() - job_search_start
                    
                    # 记录性能指标
                    self._record_performance_metric(
                        "job_search", 
                        household.household_id, 
                        job_search_duration
                    )
                    
                    return result
                except Exception as e:
                    logger.warning(f"家庭 {household.household_id} 找工作失败: {e}")
                    return ([], [])  # 返回空的(head_apps, spouse_apps)元组
        
        tasks = [find_jobs(h) for h in self.households]
        all_matched_jobs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计匹配结果 - 现在find_jobs返回的是(head_applications, spouse_applications)
        total_applications = 0
        print(f"\n{'='*60}")
        print(f"📋 第 {self.current_month} 月工作申请统计")
        print(f"{'='*60}")
        
        for i, result in enumerate(all_matched_jobs):
            if isinstance(result, tuple) and len(result) == 2:
                head_apps, spouse_apps = result
                # 使用更安全的方式获取household_id，避免索引不匹配的问题
                household = self.households[i] if i < len(self.households) else None
                household_id = household.household_id if household else f"household_{i}"
                
                head_count = len(head_apps) if isinstance(head_apps, list) else 0
                spouse_count = len(spouse_apps) if isinstance(spouse_apps, list) else 0
                
                # if head_count > 0 or spouse_count > 0:
                #     print(f"  🏠 家庭 {household_id}: 户主申请 {head_count} 个工作, 配偶申请 {spouse_count} 个工作")
                    
                #     # 显示具体申请的工作详情
                #     if head_count > 0:
                #         for app in head_apps:
                #             print(f"    👨 户主申请: Job-{app.job_id[:8]}... 期望薪资: ${app.expected_wage:.2f}/小时")
                    
                #     if spouse_count > 0:
                #         for app in spouse_apps:
                #             print(f"    👩 配偶申请: Job-{app.job_id[:8]}... 期望薪资: ${app.expected_wage:.2f}/小时")
                
                total_applications += head_count + spouse_count
        
        job_matching_duration = time.time() - job_matching_start
        print(f"\n📊 总计: {total_applications} 个工作申请已提交")
        logger.info(f"本月提交工作申请数量: {total_applications}")
        
        return total_applications, job_matching_duration

    async def process_firm_hiring_decisions(self, firms_to_post_jobs):
        """企业处理工作申请并做出招聘决策，返回(招聘决策列表, job_offers, 耗时秒)"""
        firm_decisions_start = time.time()
        print(f"\n{'='*60}")
        print(f"🏢 企业招聘决策阶段")
        print(f"{'='*60}")
        
        # 初始化本月备用候选人相关数据结构
        if not hasattr(self, 'backup_candidates_history'):
            self.backup_candidates_history = {}
        if not hasattr(self, 'monthly_backup_stats'):
            self.monthly_backup_stats = {}
        
        # 清空上个月的备用候选人数据，确保数据准确性
        if self.current_month in self.backup_candidates_history:
            del self.backup_candidates_history[self.current_month]
        if self.current_month in self.monthly_backup_stats:
            del self.monthly_backup_stats[self.current_month]
        
        logger.info("企业处理工作申请...")
        
        # 批量并行处理企业招聘决策
        print(f"🤔 开始批量并行处理企业招聘决策...")
        
        # ✅ 修复逻辑缺陷：不仅处理本月发布岗位的企业，还要处理之前发布但未成功招聘的企业
        # 获取所有有开放岗位的企业（包括本月发布的和之前发布但还有空缺的）
        firms_with_open_jobs = []
        
        # 从劳动力市场获取所有有开放岗位的企业
        try:
            all_open_jobs = await self.labor_market.get_open_jobs.remote()  # 获取所有开放岗位
            company_ids_with_jobs = set()
            for job in all_open_jobs:
                company_ids_with_jobs.add(job.company_id)
            
            # 找到对应的企业对象
            for firm in self.firms:
                if firm.company_id in company_ids_with_jobs:
                    firms_with_open_jobs.append(firm)
            
            print(f"📊 发现 {len(firms_with_open_jobs)} 家企业有开放岗位需要处理申请")
            print(f"   其中本月新发布岗位的企业: {len(firms_to_post_jobs)} 家")
            print(f"   包含之前发布但未满员的企业: {len(firms_with_open_jobs) - len([f for f in firms_with_open_jobs if f in firms_to_post_jobs])} 家")
        except Exception as e:
            logger.warning(f"获取开放岗位企业列表失败，回退到只处理本月发布岗位的企业: {e}")
            firms_with_open_jobs = firms_to_post_jobs
        
        # 动态调整批大小：根据企业数量和系统资源
        total_firms = len(firms_with_open_jobs)
        batch_size = min(10, max(3, total_firms // 4))  # 3-10家企业/批，根据企业总数动态调整
        all_hiring_decisions = []
        
        for i in range(0, len(firms_with_open_jobs), batch_size):
            batch_firms = firms_with_open_jobs[i:i+batch_size]
            print(f"  📦 处理第 {i//batch_size + 1} 批企业 ({len(batch_firms)} 家)...")
            
            batch_tasks = []
            for firm in batch_firms:
                try:
                    print(f"    🏭 企业 {firm.company_id} 开始处理工作申请...")
                    batch_tasks.append(
                    self.labor_market.process_job_applications_for_firm.remote(firm.company_id, self.current_month)
                        )
                except Exception as e:
                    logger.warning(f"企业 {firm.company_id} 处理申请失败: {e}")
                    print(f"    ❌ 企业 {firm.company_id} 处理申请失败: {e}")
            
            if batch_tasks:
                print(f"    ⏳ 等待批次内 {len(batch_tasks)} 家企业完成招聘决策...")
                firm_decisions = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 处理批次结果
                for j, decisions in enumerate(firm_decisions):
                    if isinstance(decisions, list):
                        firm = batch_firms[j] if j < len(batch_firms) else None
                        firm_id = self._get_consistent_firm_id(firm) if firm else f"firm_{j}"
                        print(f"    ✅ 企业 {firm_id} 完成决策: {len(decisions)} 个招聘决策")
                        
                        # 显示每个决策的详情 - 支持主要候选人和备选候选人
                        for decision in decisions:
                            job_title = decision.get("job_title", "未知职位")
                            primary_candidates = decision.get("primary_candidates", [])
                            backup_candidates = decision.get("backup_candidates", [])
                            
                            # 兼容旧格式
                            if not primary_candidates and "selected_candidates" in decision:
                                primary_candidates = decision["selected_candidates"]
                            
                            total_candidates = decision.get("total_candidates", 0)
                            total_selected = len(primary_candidates) + len(backup_candidates)
                            
                            # print(f"      📋 职位 '{job_title}': 从 {total_candidates} 个候选人中选择了 {total_selected} 人")
                            # print(f"          主要候选人: {len(primary_candidates)} 个, 备选候选人: {len(backup_candidates)} 个")
                            
                            # 显示主要候选人
                            for candidate in primary_candidates:
                                household_id = candidate.get("household_id")
                                raw_final_wage = candidate.get("final_wage_offer", 0)
                                # 确保工资数据是数字类型
                                try:
                                    final_wage = float(raw_final_wage) if raw_final_wage else 0
                                except (ValueError, TypeError):
                                    final_wage = 0
                                role = candidate.get("lh_type", "未知")
                                # print(f"        🎯 主要: 家庭 {household_id} ({role}) - 薪资: ${final_wage:.2f}/小时")
                            
                            # 显示备选候选人
                            for candidate in backup_candidates:
                                household_id = candidate.get("household_id")
                                raw_final_wage = candidate.get("final_wage_offer", 0)
                                # 确保工资数据是数字类型
                                try:
                                    final_wage = float(raw_final_wage) if raw_final_wage else 0
                                except (ValueError, TypeError):
                                    final_wage = 0
                                role = candidate.get("lh_type", "未知")
                                priority = candidate.get("priority_rank", "未知")
                                print(f"        🔄 备选{priority}: 家庭 {household_id} ({role}) - 薪资: ${final_wage:.2f}/小时")
                        
                        all_hiring_decisions.extend(decisions)
                    elif isinstance(decisions, Exception):
                        firm = batch_firms[j] if j < len(batch_firms) else None
                        firm_id = self._get_consistent_firm_id(firm) if firm else f"firm_{j}"
                        print(f"    ❌ 企业 {firm_id} 决策失败: {decisions}")
        
        firm_decisions_duration = time.time() - firm_decisions_start
        print(f"\n📊 招聘决策汇总: {len(all_hiring_decisions)} 个职位完成决策")
        
        # 确认招聘决策 - 企业发出job offers
        job_offers_start = time.time()
        print(f"\n{'='*60}")
        print(f"📧 企业发出Job Offers")
        print(f"{'='*60}")
        
        logger.info("企业发出工作offer...")
        job_offers = []
        if all_hiring_decisions:
            print(f"🔄 正在处理 {len(all_hiring_decisions)} 个招聘决策...")
            job_offers = await self.labor_market.finalize_hiring_decisions.remote(all_hiring_decisions)
            
            job_offers_duration = time.time() - job_offers_start
            print(f"✨ Job offers发送完成!")
        else:
            print(f"ℹ️  本月没有招聘决策需要处理")
            job_offers_duration = 0.0
        
        total_duration = time.time() - firm_decisions_start
        return all_hiring_decisions, job_offers, total_duration

    async def process_household_offer_evaluation(self, job_offers):
        """家庭评估job offers，处理备选候选人，返回(接受的offers列表, 耗时秒)"""
        household_evaluation_start = time.time()
        print(f"\n{'='*60}")
        print(f"🤔 家庭智能评估Job Offers")
        print(f"{'='*60}")
        
        accepted_offers = []
        if job_offers:
            print(f"📋 收到总计 {len(job_offers)} 个工作offer")
            
            # 按家庭分组offers
            household_offers = {}
            for offer in job_offers:
                household_id = offer.get("household_id")
                if household_id not in household_offers:
                    household_offers[household_id] = []
                household_offers[household_id].append(offer)
            
            print(f"📊 涉及 {len(household_offers)} 个家庭")
            
            # 每个家庭独立评估其offers
            evaluation_tasks = []
            for household_id, offers in household_offers.items():
                # 找到对应的家庭对象
                household = self._find_household_by_id(household_id, self.households)
                if household:
                    print(f"\n  🏠 家庭 {household_id} 收到 {len(offers)} 个offers")
                    evaluation_tasks.append(household.evaluate_job_offers(offers, std_job))
            
            # 并行处理所有家庭的offer评估
            if evaluation_tasks:
                household_decisions = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                
                # 收集所有接受的offers
                for decisions in household_decisions:
                    if isinstance(decisions, list):
                        accepted_offers.extend(decisions)
                    elif isinstance(decisions, Exception):
                        print(f"    ❌ 家庭评估失败: {decisions}")
            
            print(f"\n📈 评估结果汇总:")
            print(f"   收到offers: {len(job_offers)}")
            print(f"   接受offers: {len(accepted_offers)}")
            print(f"   拒绝offers: {len(job_offers) - len(accepted_offers)}")
            print(f"   接受率: {len(accepted_offers)/len(job_offers)*100:.1f}%")
        else:
            print(f"ℹ️  本月没有家庭收到工作offer")
        
        # 处理被拒绝的offers，激活备选候选人
        backup_offers = []
        if job_offers and len(accepted_offers) < len(job_offers):
            print(f"\n{'='*60}")
            print(f"🔄 激活备选候选人机制")
            print(f"{'='*60}")
            
            backup_offers = await self.labor_market.process_rejected_offers_and_activate_backups.remote(job_offers, accepted_offers)
            
            # 记录本月备用候选人统计
            self.monthly_backup_stats[self.current_month] = {
                'total_rejected_offers': len(job_offers) - len(accepted_offers),
                'backup_offers_activated': len(backup_offers) if backup_offers else 0,
                'backup_offers_accepted': 0,  # 稍后更新
                'backup_success_rate': 0.0     # 稍后更新
            }
            
            # 如果有备选candidates被激活，让他们也评估offers
            if backup_offers:
                print(f"\n🔄 备选候选人评估新offers...")
                
                # 按家庭分组backup offers
                backup_household_offers = {}
                for offer in backup_offers:
                    household_id = offer.get("household_id")
                    if household_id not in backup_household_offers:
                        backup_household_offers[household_id] = []
                    backup_household_offers[household_id].append(offer)
                
                # 每个家庭评估backup offers
                backup_evaluation_tasks = []
                for household_id, offers in backup_household_offers.items():
                    household = self._find_household_by_id(household_id, self.households)
                    if household:
                        print(f"  🏠 家庭 {household_id} 收到 {len(offers)} 个备选offers")
                        backup_evaluation_tasks.append(household.evaluate_job_offers(offers, std_job))
                
                # 并行处理备选offers评估
                if backup_evaluation_tasks:
                    backup_decisions = await asyncio.gather(*backup_evaluation_tasks, return_exceptions=True)
                    
                    # 收集备选offers的接受情况
                    backup_accepted_count = 0
                    # 创建已接受工作的家庭成员集合，防止重复接受
                    already_hired = set()
                    for offer in accepted_offers:
                        household_key = f"{offer['household_id']}_{offer['lh_type']}"
                        already_hired.add(household_key)
                    
                    for decisions in backup_decisions:
                        if isinstance(decisions, list):
                            # 检查备选候选人是否已经接受了其他工作
                            valid_backup_decisions = []
                            for decision in decisions:
                                household_key = f"{decision['household_id']}_{decision['lh_type']}"
                                if household_key not in already_hired:
                                    decision["offer_status"] = "backup_activated"
                                    valid_backup_decisions.append(decision)
                                    already_hired.add(household_key)  # 标记为已雇佣
                                else:
                                    print(f"    ⚠️  跳过重复接受: 家庭 {decision['household_id']} ({decision['lh_type']}) 已接受其他工作")
                            
                            accepted_offers.extend(valid_backup_decisions)
                            backup_accepted_count += len(valid_backup_decisions)
                        elif isinstance(decisions, Exception):
                            print(f"    ❌ 备选候选人评估失败: {decisions}")
                    
                    # 更新备用候选人统计
                    self.monthly_backup_stats[self.current_month]['backup_offers_accepted'] = backup_accepted_count
                    if backup_offers:
                        self.monthly_backup_stats[self.current_month]['backup_success_rate'] = backup_accepted_count / len(backup_offers)
                    
                    print(f"✅ 备选候选人评估完成: 新增 {backup_accepted_count} 个接受的backup offers")
                    print(f"📊 备选成功率: {backup_accepted_count}/{len(backup_offers)} = {self.monthly_backup_stats[self.current_month]['backup_success_rate']:.1%}")
            else:
                print(f"ℹ️  本月没有备选候选人被激活")
                # 更新统计：没有备选候选人被激活
                self.monthly_backup_stats[self.current_month]['backup_offers_activated'] = 0
                self.monthly_backup_stats[self.current_month]['backup_offers_accepted'] = 0
                self.monthly_backup_stats[self.current_month]['backup_success_rate'] = 0.0
        else:
            print(f"ℹ️  本月没有需要激活备选候选人的情况")
            # 记录本月统计：没有拒绝的offers
            self.monthly_backup_stats[self.current_month] = {
                'total_rejected_offers': 0,
                'backup_offers_activated': 0,
                'backup_offers_accepted': 0,
                'backup_success_rate': 0.0
            }
            print(f"📊 本月所有offers都被接受，无需激活备选候选人")
        
        household_evaluation_duration = time.time() - household_evaluation_start
        return accepted_offers, household_evaluation_duration

    async def process_hiring_confirmation(self, accepted_offers):
        """最终雇佣确认，更新家庭和企业状态，返回(确认的雇佣数, 耗时秒)"""
        hiring_confirmation_start = time.time()
        print(f"\n{'='*60}")
        print(f"✅ 最终雇佣确认")
        print(f"{'='*60}")
        
        confirmed_hires = []
        if accepted_offers:
            print(f"🔄 正在确认 {len(accepted_offers)} 个接受的offers (包含备选)...")
            confirmed_hires = await self.labor_market.process_job_acceptances.remote(accepted_offers)
            
            # 保存雇佣确认结果，供企业月度指标收集使用
            if not hasattr(self, 'confirmed_hires_for_month'):
                self.confirmed_hires_for_month = {}
            self.confirmed_hires_for_month[self.current_month] = confirmed_hires
            
            # 更新家庭的 income_this_period
            if confirmed_hires:
                print(f"💰 更新家庭收入预期...")
                household_income_updates = {}  # {household_id: total_monthly_income}
                
                for hire in confirmed_hires:
                    household_id = hire.get('household_id')
                    offered_wage = hire.get('offered_wage', 0.0)
                    # 假设全职工作，每月160小时（每周40小时 × 4周）
                    monthly_income = offered_wage * 160.0
                    
                    if household_id not in household_income_updates:
                        household_income_updates[household_id] = 0.0
                    household_income_updates[household_id] += monthly_income
                
                # 批量更新所有受影响的家庭
                updated_households = 0
                for household in self.households:
                    if household.household_id in household_income_updates:
                        household.income_this_period = household_income_updates[household.household_id]
                        updated_households += 1
                        logger.debug(f"家庭 {household.household_id} 预期月收入更新为: ${household.income_this_period:.2f}")
                
                print(f"✅ 已更新 {updated_households} 个家庭的预期月收入")
            
            print(f"✨ 最终雇佣确认完成!")
        else:
            print(f"ℹ️  没有需要确认的雇佣关系")
            # 即使没有雇佣，也要记录空结果
            if not hasattr(self, 'confirmed_hires_for_month'):
                self.confirmed_hires_for_month = {}
            self.confirmed_hires_for_month[self.current_month] = []
        
        total_aligned_job = len(confirmed_hires) if confirmed_hires else 0
        backup_hires = len([h for h in confirmed_hires if h.get("offer_status") == "backup_activated"]) if confirmed_hires else 0
        
        # 更新本月备用候选人最终统计
        if self.current_month in self.monthly_backup_stats:
            self.monthly_backup_stats[self.current_month]['final_backup_hires'] = backup_hires
            self.monthly_backup_stats[self.current_month]['total_hires'] = total_aligned_job
            if self.monthly_backup_stats[self.current_month]['backup_offers_activated'] > 0:
                self.monthly_backup_stats[self.current_month]['final_backup_success_rate'] = backup_hires / self.monthly_backup_stats[self.current_month]['backup_offers_activated']
            else:
                self.monthly_backup_stats[self.current_month]['final_backup_success_rate'] = 0.0
        
        print(f"\n🎉 第 {self.current_month} 月最终招聘结果:")
        print(f"   成功招聘总数: {total_aligned_job} 个职位")
        print(f"   主要候选人: {total_aligned_job - backup_hires} 个")
        print(f"   备选候选人: {backup_hires} 个")
        if backup_hires > 0 and self.current_month in self.monthly_backup_stats:
            backup_activated = self.monthly_backup_stats[self.current_month]['backup_offers_activated']
            if backup_activated > 0:
                success_rate = backup_hires / backup_activated * 100
                print(f"   备选成功率: {backup_hires}/{backup_activated} = {success_rate:.1f}%")
            else:
                print(f"   备选成功率: {backup_hires}/0 = 0.0%")
        
        # 显示本月备用候选人详细统计
        if self.current_month in self.monthly_backup_stats:
            stats = self.monthly_backup_stats[self.current_month]
            print(f"\n📊 第 {self.current_month} 月备用候选人统计:")
            print(f"   被拒绝offers: {stats['total_rejected_offers']} 个")
            print(f"   激活备选候选人: {stats['backup_offers_activated']} 个")
            print(f"   备选候选人接受: {stats['backup_offers_accepted']} 个")
            print(f"   最终备选雇佣: {stats.get('final_backup_hires', 0)} 个")
            print(f"   备选成功率: {stats.get('final_backup_success_rate', 0):.1%}")
        
        logger.info(f"仿真迭代 {self.current_month}: 成功招聘 {total_aligned_job} 个职位")
        
        # 更新家庭劳动力状态并显示详细信息
        if confirmed_hires:
            print(f"\n📋 成功招聘详情:")
            print(f"{'-'*50}")
            
            # 去重处理：确保每个家庭只被处理一次
            processed_households = set()
            
            # 统计每个企业的招聘数量
            company_hires = {}
            
            # 收集需要更新的家庭状态任务（并行执行）
            household_update_tasks = []
            
            for i, hire in enumerate(confirmed_hires, 1):
                household_id = hire.get("household_id")
                job_title = hire.get("job_title", "未知职位")
                raw_final_wage = hire.get("final_wage", 0)
                # 确保工资数据是数字类型
                try:
                    final_wage = float(raw_final_wage) if raw_final_wage else 0
                except (ValueError, TypeError):
                    final_wage = 0
                role = hire.get("lh_type", "未知")
                company_id = hire.get("company_id", "未知企业")
                
                # 统计企业招聘数量
                if company_id not in company_hires:
                    company_hires[company_id] = 0
                company_hires[company_id] += 1
                
                print(f"  {i:2d}. 🏠 家庭 {household_id} ({role})")
                print(f"      💼 职位: {job_title}")
                print(f"      🏢 企业: {company_id}")
                print(f"      💰 薪资: ${final_wage:.2f}/小时")
                print()
                
                # 检查是否已经处理过这个家庭的这个角色
                household_role_key = f"{household_id}_{role}"
                if household_role_key in processed_households:
                    print(f"      ⚠️  跳过重复处理: 家庭 {household_id} 的 {role} 已经被处理过")
                    continue
                
                # 找到对应的家庭对象并准备更新任务
                for household in self.households:
                    if household.household_id == household_id:
                        # 创建Job对象用于更新labor_hour状态
                        job_for_update = Job.create(
                            soc=hire.get("job_SOC", ""),
                            title=job_title,
                            wage_per_hour=final_wage,
                            company_id=company_id
                        )
                        
                        # 添加到并行更新任务列表
                        household_update_tasks.append(
                            household.update_labor_hours(job_for_update, role)
                        )
                        
                        # 标记为已处理
                        processed_households.add(household_role_key)
                        break
            
            # 并行执行所有家庭状态更新
            if household_update_tasks:
                print(f"\n🔄 并行更新 {len(household_update_tasks)} 个家庭的劳动力状态...")
                await asyncio.gather(*household_update_tasks, return_exceptions=True)
                print(f"✅ 家庭状态更新完成!")
            
            # 将雇佣事件下发到企业，添加员工详细信息
            print(f"\n🏢 同步员工入职到企业：")
            print(f"{'-'*30}")
            for hire in confirmed_hires:
                company_id = hire.get("company_id")
                household_id = hire.get("household_id")
                lh_type = hire.get("lh_type")
                job_title = hire.get("job_title", "")
                job_soc = hire.get("job_SOC", "")
                raw_final_wage = hire.get("final_wage", 0)
                try:
                    wage_per_hour = float(raw_final_wage) if raw_final_wage else 0.0
                except (ValueError, TypeError):
                    wage_per_hour = 0.0
                # 默认每期工时，如无法获取具体hours
                hours_per_period = 40
                # 从对应家庭的 labor_hours 对象补齐 skills/abilities
                skills = {}
                abilities = {}
                household = self._find_household_by_id(household_id, self.households)  # 修复拼写错误
                
                try:
                    if household:
                        labor_hours = getattr(household, 'labor_hours', []) or []
                        for lh in labor_hours:
                            if getattr(lh, 'lh_type', None) == lh_type:
                                # 优先从 labor_hour 取
                                skills = getattr(lh, 'skill_profile', None) or {}
                                abilities = getattr(lh, 'ability_profile', None) or {}
                                break
                except Exception:
                    skills, abilities = {}, {}
                employee_data = {
                    "household_id": str(household_id),
                    "lh_type": lh_type,
                    "job_title": job_title,
                    "job_soc": job_soc,
                    "wage_per_hour": wage_per_hour,
                    "hours_per_period": hours_per_period,
                    "skills": skills,
                    "abilities": abilities,
                    "hire_date": f"month_{self.current_month}"
                }
                company = self._find_firm_by_id(company_id, self.firms)
                if company:
                    try:
                        company.add_employee(employee_data)
                        print(f"  ✅ {company_id} <- {household_id}_{lh_type} 入职 {job_title} @ ${wage_per_hour:.2f}/h")
                    except Exception as e:
                        print(f"  ❌ 同步到企业失败 {company_id}: {e}")
                else:
                    print(f"  ❌ 未找到企业对象: {company_id}")

            # 显示企业员工数量状态 (员工数量已在add_employee中更新)
            print(f"\n🏢 企业员工数量状态:")
            print(f"{'-'*30}")
            for company_id, hire_count in company_hires.items():
                # 找到对应的企业对象
                company = self._find_firm_by_id(company_id, self.firms)
                if company:
                    current_count = company.get_employees()
                    print(f"  {company_id}: 当前员工数 {current_count} (本月新增 +{hire_count})")
                else:
                    print(f"  {company_id}: 企业对象未找到")
            
            # 统计信息 - 确保工资数据是数字类型
            total_wage_cost = 0
            for hire in confirmed_hires:
                raw_wage = hire.get("final_wage", 0)
                try:
                    wage = float(raw_wage) if raw_wage else 0
                    total_wage_cost += wage
                except (ValueError, TypeError):
                    continue
            
            avg_wage = total_wage_cost / len(confirmed_hires) if confirmed_hires else 0
            print(f"\n💡 招聘统计:")
            print(f"   总薪资成本: ${total_wage_cost:.2f}/小时")
            print(f"   平均薪资: ${avg_wage:.2f}/小时")
            print(f"   涉及家庭: {len(set(h.get('household_id') for h in confirmed_hires))} 个")
            print(f"   涉及企业: {len(company_hires)} 家")
        else:
            print(f"ℹ️  本月没有成功的招聘")
        
        hiring_confirmation_duration = time.time() - hiring_confirmation_start
        return total_aligned_job, hiring_confirmation_duration

    async def process_household_consumption(self):
        """
        家庭消费和商品价格更新，返回(成功消费家庭数, 耗时秒)
        
        ✨ 优化：使用批量LLM调用，大幅减少API请求次数和等待时间
        """
        household_consumption_start = time.time()
        logger.info(f"仿真迭代 {self.current_month}: 家庭进行消费...")
        
        # 🚀 方案2：批量LLM预算分配
        if self.config.use_batch_budget_allocation:
            print(f"\n🛒 开始批量处理家庭消费 (批量模式) - {len(self.households)}个家庭...")
            return await self._process_household_consumption_batch()
        
        # 原有的并行模式
        # print(f"\n🛒 开始并行处理家庭消费 (并行模式: {self.config.max_llm_concurrent}个并发)...")
        # print(f"   所有{len(self.households)}个家庭同时启动，LLM调用共享并发池")
        outer_semaphore = asyncio.Semaphore(100)
        
        async def consume_household(household):
            async with outer_semaphore:
                try:                            
                    # 记录消费预算计算和购买操作的耗时
                    consumption_start = time.time()
                    await household.consume(self.product_market, self.economic_center)
                    consumption_duration = time.time() - consumption_start
                    
                    # 记录性能指标
                    self._record_performance_metric(
                        "consumption", 
                        household.household_id, 
                        consumption_duration
                    )
                    
                    return household.household_id, consumption_duration
                except Exception as e:
                    logger.warning(f"家庭 {household.household_id} 消费失败: {e}")
                    return household.household_id, 0.0
        
        # 并行执行所有家庭的消费（无限制，由全局LLM信号量控制实际并发）
        consumption_tasks = [consume_household(household) for household in self.households]
        consumption_results = await asyncio.gather(*consumption_tasks, return_exceptions=True)
        
        # 统计消费结果
        successful_consumptions = 0
        total_consumption_time = 0.0
        for result in consumption_results:
            if isinstance(result, tuple) and len(result) == 2:
                household_id, duration = result
                if duration > 0:
                    successful_consumptions += 1
                    total_consumption_time += duration
        
        avg_consumption_time = total_consumption_time / successful_consumptions if successful_consumptions > 0 else 0
        print(f"✅ 家庭消费完成: {successful_consumptions}/{len(self.households)} 个家庭成功完成消费")
        
        household_consumption_duration = time.time() - household_consumption_start
        return successful_consumptions, household_consumption_duration
    
    async def _process_household_consumption_batch(self):
        """
        ✨ 批量LLM模式：将多个家庭的预算请求合并为批次处理
        
        流程：
        1. 收集所有家庭的上下文信息
        2. 批量调用LLM进行预算分配（一次API调用处理多个家庭）
        3. 分发预算结果给各个家庭
        4. 并行执行商品购买
        """
        from agentsociety_ecosim.consumer_modeling.consumer_decision import BudgetAllocator
        
        batch_start = time.time()
        successful_consumptions = 0
        
        # 步骤1：准备所有家庭的预算请求
        print(f"📋 步骤1/4: 收集{len(self.households)}个家庭的消费上下文...")
        prep_start = time.time()
        
        household_contexts = []
        for household in self.households:
            try:
                # 获取家庭基本信息
                balance = await self.economic_center.query_balance.remote(household.household_id)
                
                # 获取上个月收入
                last_month_income = 0
                if self.current_month > 1:
                    try:
                        last_month_income = await self.economic_center.query_income.remote(
                            household.household_id, self.current_month - 1
                        )
                    except:
                        pass
                
                # 🔧 修改：从经济中心获取当前月份的工作工资，而不是从轻量化家庭对象读取
                # 获取当前月份收入（工作工资）
                current_month_income = 0
                try:
                    current_month_income = await self.economic_center.query_income.remote(
                        household.household_id, self.current_month
                    )
                except Exception as e:
                    logger.debug(f"无法从经济中心获取家庭 {household.household_id} 当前月份收入: {e}")
                
                # 生成就业信息（基于经济中心的工资数据）
                ex_info = self._generate_employment_ex_info_from_center(
                    household, current_month_income, last_month_income
                )
                
                household_contexts.append({
                    "household": household,
                    "household_id": household.household_id,
                    "balance": balance,
                    "last_month_income": last_month_income,
                    "current_month_income": current_month_income,
                    "ex_info": ex_info,
                    "family_profile": household.family_profile or {}
                })
            except Exception as e:
                logger.warning(f"收集家庭 {household.household_id} 上下文失败: {e}")
        
        prep_duration = time.time() - prep_start
        print(f"   ✅ 上下文收集完成 ({prep_duration:.2f}秒)")
        
        # 步骤2：批量LLM预算分配
        total_batches = (len(household_contexts) + self.config.batch_size - 1) // self.config.batch_size
        print(f"\n🤖 步骤2/4: 批量LLM预算分配")
        print(f"   批次配置: 每批{self.config.batch_size}个家庭, 共{total_batches}批次")
        print(f"   超时设置: {self.config.batch_llm_timeout}秒")
        budget_start = time.time()
        
        # 初始化BudgetAllocator（如果需要）
        if not hasattr(self, '_batch_budget_allocator'):
            self._batch_budget_allocator = BudgetAllocator(
                product_market=self.product_market,
                economic_center=self.economic_center
            )
        
        # 批量分配预算
        budget_results = await self._batch_budget_allocator.batch_allocate(
            household_contexts,
            current_month=self.current_month,
            batch_size=self.config.batch_size
        )
        
        budget_duration = time.time() - budget_start
        print(f"   ✅ 预算分配完成 ({budget_duration:.2f}秒, 平均{budget_duration/len(household_contexts):.3f}秒/家庭)")
        
        # 步骤3：并行执行商品购买
        print(f"\n🛍️  步骤3/4: 并行执行商品购买...")
        purchase_start = time.time()
        
        async def execute_purchase(context, budget_result):
            try:
                household = context["household"]
                if budget_result and budget_result.get("shopping_plan"):
                    # 执行购买
                    total_spent, purchased_items = await household.execute_budget_based_purchases(
                        budget_result["shopping_plan"],
                        self.product_market
                    )
                    
                    # 更新属性
                    if purchased_items:
                        await household.update_attributes_after_purchase(
                            purchased_items,
                            budget_result.get("shopping_plan")
                        )
                    
                    return household.household_id, total_spent, True
                return household.household_id, 0.0, False
            except Exception as e:
                logger.warning(f"家庭 {context['household_id']} 购买失败: {e}")
                return context['household_id'], 0.0, False
        
        purchase_tasks = [
            execute_purchase(ctx, budget_results.get(ctx["household_id"]))
            for ctx in household_contexts
        ]
        purchase_results = await asyncio.gather(*purchase_tasks, return_exceptions=True)
        
        # 统计结果
        for result in purchase_results:
            if isinstance(result, tuple) and len(result) >= 3:
                household_id, spent, success = result
                if success:
                    successful_consumptions += 1
        
        purchase_duration = time.time() - purchase_start
        print(f"   ✅ 商品购买完成 ({purchase_duration:.2f}秒)")
        
        # 步骤4：统计和总结
        total_duration = time.time() - batch_start
        print(f"\n📊 批量消费完成统计:")
        print(f"   成功家庭: {successful_consumptions}/{len(self.households)}")
        print(f"   总耗时: {total_duration:.2f}秒")
        print(f"   阶段耗时:")
        print(f"     - 上下文收集: {prep_duration:.2f}秒 ({prep_duration/total_duration*100:.1f}%)")
        print(f"     - 批量预算: {budget_duration:.2f}秒 ({budget_duration/total_duration*100:.1f}%)")
        print(f"     - 商品购买: {purchase_duration:.2f}秒 ({purchase_duration/total_duration*100:.1f}%)")
        print(f"   平均每家庭: {total_duration/len(self.households):.3f}秒")
        
        return successful_consumptions, total_duration

    def _generate_employment_ex_info_from_center(
        self, 
        household, 
        current_month_income: float, 
        last_month_income: float = 0
    ) -> str:
        """
        从经济中心获取工作工资信息，生成就业状况ex_info
        
        Args:
            household: 家庭对象
            current_month_income: 当前月份收入（从经济中心获取）
            last_month_income: 上个月收入（可选）
            
        Returns:
            str: 格式化的就业状况和税率信息
        """
        try:
            # 获取税率信息（从家庭对象或使用默认值）
            income_tax_rate = getattr(household, 'income_tax_rate', 0.225)
            vat_rate = getattr(household, 'vat_rate', 0.08)
            
            # 计算月度工资（假设当前月份收入就是工作工资）
            monthly_salary = current_month_income if current_month_income > 0 else 0.0
            
            # 判断就业状态
            is_employed = monthly_salary > 0
            
            # 计算税后收入和购买力
            gross_income = monthly_salary
            after_tax_income = gross_income * (1 - income_tax_rate) if gross_income > 0 else 0.0
            effective_purchasing_power = after_tax_income / (1 + vat_rate) if after_tax_income > 0 else 0.0
            combined_burden = income_tax_rate + vat_rate
            
            # 构建ex_info文本（英文版，包含税率信息）
            if is_employed:
                employment_status = "Employed"
                job_info = f"Monthly salary: ${monthly_salary:.0f}"
            else:
                employment_status = "Unemployed"
                job_info = "No current employment"
            
            ex_info = f"""=== Current Household Employment Status ===
Labor Force Overview:
- Total household labor force: 1-2 people (estimated)
- Currently employed: {'1' if is_employed else '0'} person(s)
- Household employment rate: {'50-100%' if is_employed else '0%'}

Employment Details:
- Head: {employment_status} | {job_info}
- Spouse: Unknown (data from economic center)

Income Status:
- Total estimated monthly income: ${monthly_salary:.0f}
- Primary income source: {'Employment' if is_employed else 'No income'}
- Income structure: {'Single-income household' if is_employed else 'No-income household'}

=== Tax Environment ===
Tax Rates: Income {income_tax_rate:.1%} + Sales {vat_rate:.1%} = {combined_burden:.1%} burden
After-Tax: Gross ${gross_income:.0f} → Net ${after_tax_income:.0f} → Purchasing Power ${effective_purchasing_power:.0f}
Note: Product prices exclude {vat_rate:.1%} sales tax. Budget on net income ${after_tax_income:.0f}

=== Please consider employment status and tax impact in consumption decisions ==="""

            return ex_info
            
        except Exception as e:
            logger.warning(f"从经济中心生成家庭 {household.household_id} 就业ex_info失败: {e}")
            # 返回默认信息
            return """=== Current Household Employment Status ===
Failed to retrieve employment information from economic center, adopting conservative consumption strategy
=== Please consider employment status impact in consumption decisions ==="""

    async def update_product_prices(self):
        """更新商品市场价格，返回(更新的商品数, 耗时秒)"""
        price_update_start = time.time()
        logger.info(f"🔄 开始第 {self.current_month} 月的商品价格更新流程...")

        try:
            # 1. 收集销售统计数据
            logger.info("📊 收集销售统计数据...")
            sales_stats = await self.economic_center.collect_sales_statistics.remote(self.current_month)
            
            if not sales_stats:
                logger.warning("⚠️ 本月没有销售数据，跳过价格更新")
                updated_products = 0
            elif not self.config.enable_price_adjustment:
                # 价格调整功能已关闭
                logger.info("💰 价格自动调整功能已关闭，保持价格不变")
                updated_products = 0
            else:
                logger.info(f"📈 收集到 {len(sales_stats)} 个商品-企业组合的销售数据")
                
                # 2. 根据销量更新价格
                logger.info("💰 根据销量数据更新商品价格...")
                price_changes = await self.economic_center.update_product_prices_based_on_sales.remote(
                    sales_stats, self.config.price_adjustment_rate
                )
                
                if not price_changes:
                    logger.warning("⚠️ 没有商品价格需要更新")
                    updated_products = 0
                else:
                    logger.info(f"💵 更新了 {len(price_changes)} 个商品的价格")
                    
                    # 3. 同步价格变更到ProductMarket
                    logger.info("🔄 同步价格变更到ProductMarket...")
                    sync_success = await self.economic_center.sync_price_changes_to_market.remote(
                        self.product_market, price_changes
                    )
                    
                    if sync_success:
                        logger.info("✅ 价格同步成功")
                        updated_products = len(price_changes)
                    else:
                        logger.error("❌ 价格同步失败")
                        updated_products = 0
        except Exception as e:
            logger.error(f"❌ 价格更新过程中发生错误: {e}")
            updated_products = 0
        
        price_update_duration = time.time() - price_update_start
        return updated_products, price_update_duration

    async def process_wage_payment_and_tracking(self):
        """工资发放和工作追踪更新，返回(发放岗位数, 耗时秒)"""
        wage_processing_start = time.time()
        print(f"\n{'='*60}")
        print(f"💰 工资发放阶段")
        print(f"{'='*60}")
        
        logger.info(f"仿真迭代 {self.current_month}: 处理工资发放...")
        
        # 获取当前匹配的工作数量
        matched_jobs_count = len(await self.labor_market.query_matched_jobs.remote())
        
        if matched_jobs_count > 0:
            print(f"💼 准备为 {matched_jobs_count} 个工作岗位发放工资...")
            print(f"🔄 正在处理工资转账...")
            
        await self.labor_market.process_wages.remote(self.economic_center, self.current_month)
        
        print(f"✅ 工资发放完成!")
        print(f"📊 本月工资发放统计:")
        print(f"   发放岗位数: {matched_jobs_count}")
        print(f"   发放月份: 第 {self.current_month} 月")
        
        # ===== 更新家庭工作追踪记录 =====
        print(f"\n📝 更新家庭工作追踪记录...")
        job_tracking_start = time.time()
        
        # 获取当月的工作匹配信息，用于工资追踪
        matched_jobs = await self.labor_market.query_matched_jobs.remote()
        wage_info = {}  # {household_id: {lh_type: wage}}
        
        for matched_job in matched_jobs:
            household_id = matched_job.household_id
            lh_type = matched_job.lh_type
            wage = matched_job.average_wage
            
            if household_id not in wage_info:
                wage_info[household_id] = {}
            wage_info[household_id][lh_type] = wage
        
        # 为所有家庭更新当月的工作状态
        for household in self.households:
            try:
                # 更新基本工作状态
                household.update_monthly_job_status(self.current_month)
                
                # 添加工资信息到工作追踪记录
                if household.household_id in wage_info:
                    for lh_type, wage in wage_info[household.household_id].items():
                        household.add_wage_info_to_job_tracking(self.current_month, lh_type, wage)
                        
            except Exception as e:
                logger.warning(f"更新家庭 {household.household_id} 工作追踪失败: {e}")
        
        job_tracking_duration = time.time() - job_tracking_start
        print(f"✅ 工作追踪记录更新完成 (耗时: {job_tracking_duration:.3f}秒)")
        
        # 打印工作追踪统计
        total_head_employed = 0
        total_spouse_employed = 0
        total_both_employed = 0
        
        for household in self.households:
            monthly_status = household.get_monthly_job_status(self.current_month)
            head_employed = monthly_status.get('head', {}).get('employed', False)
            spouse_employed = monthly_status.get('spouse', {}).get('employed', False)
            
            if head_employed:
                total_head_employed += 1
            if spouse_employed:
                total_spouse_employed += 1
            if head_employed and spouse_employed:
                total_both_employed += 1
        
        print(f"📊 第 {self.current_month} 月就业统计:")
        print(f"   户主就业: {total_head_employed}/{len(self.households)} ({total_head_employed/len(self.households):.1%})")
        print(f"   配偶就业: {total_spouse_employed}/{len(self.households)} ({total_spouse_employed/len(self.households):.1%})")
        print(f"   双人就业: {total_both_employed}/{len(self.households)} ({total_both_employed/len(self.households):.1%})")
        
        wage_processing_duration = time.time() - wage_processing_start
        return matched_jobs_count, wage_processing_duration

    async def process_skill_enhancement(self):
        """技能和能力提升系统，返回(获得提升的劳动力数, 耗时秒)"""
        skill_enhancement_start = time.time()
        print(f"\n📚 ===== 第 {self.current_month} 月技能提升 =====")
        
        # 为所有有工作的家庭提升技能
        enhanced_households = 0
        total_skill_enhancements = 0
        
        for household in self.households:
            try:
                # 获取提升前的技能摘要
                before_summary = household.get_skill_development_summary()
                
                # 提升技能
                household.enhance_labor_skills(self.current_month, std_job)
                
                # 获取提升后的技能摘要
                after_summary = household.get_skill_development_summary()
                
                # 检查是否有技能提升 - 使用更宽松的判断标准
                for lh_type in ['head', 'spouse']:
                    if (lh_type in before_summary and lh_type in after_summary and
                        before_summary[lh_type]['employed'] and after_summary[lh_type]['employed']):

                        before_skill_avg = before_summary[lh_type]['skill_average']
                        after_skill_avg = after_summary[lh_type]['skill_average']
                        before_ability_avg = before_summary[lh_type]['ability_average']
                        after_ability_avg = after_summary[lh_type]['ability_average']

                        # 计算提升幅度（避免浮点数精度问题）
                        skill_improvement = after_skill_avg - before_skill_avg
                        ability_improvement = after_ability_avg - before_ability_avg

                        # 设置最小提升阈值（0.001），避免微小数值差异导致的问题
                        min_improvement_threshold = 0.001

                        # 只要有任何提升（包括很小的提升），就认为有技能提升
                        if (skill_improvement >= min_improvement_threshold or
                            ability_improvement >= min_improvement_threshold):
                            enhanced_households += 1
                            total_skill_enhancements += 1

                            # 记录详细的提升信息
                            logger.debug(f"家庭 {household.household_id} {lh_type} 技能提升: "
                                       f"技能 {before_skill_avg:.4f} -> {after_skill_avg:.4f} "
                                       f"(+{skill_improvement:.4f}), "
                                       f"能力 {before_ability_avg:.4f} -> {after_ability_avg:.4f} "
                                       f"(+{ability_improvement:.4f})")
                        elif skill_improvement > 0 or ability_improvement > 0:
                            # 有提升但低于阈值的情况
                            logger.debug(f"家庭 {household.household_id} {lh_type} 有微小提升但低于阈值: "
                                       f"技能 +{skill_improvement:.6f}, 能力 +{ability_improvement:.6f}")
                            
            except Exception as e:
                logger.warning(f"家庭 {household.household_id} 技能提升失败: {e}")
        
        skill_enhancement_duration = time.time() - skill_enhancement_start
        print(f"✅ 技能提升完成 (耗时: {skill_enhancement_duration:.3f}秒)")
        print(f"📊 技能提升统计:")
        print(f"   参与家庭: {len(self.households)} 个")
        print(f"   获得提升的劳动力: {total_skill_enhancements} 人")
        print(f"   提升成功率: {total_skill_enhancements/(len(self.households)*2):.1%} (基于{len(self.households)*2}个劳动力机会)")

        return total_skill_enhancements, skill_enhancement_duration

    async def process_bank_interest(self):
        """银行利息发放，返回(发放利息总额, 耗时秒)"""
        bank_interest_start = time.time()
        print(f"\n💰 ===== 第 {self.current_month} 月银行利息发放 =====")

        try:
            if hasattr(self, 'bank') and self.bank:
                total_interest = await self.bank.calculate_and_pay_monthly_interest.remote(self.current_month)
                print(f"✅ 第 {self.current_month} 月利息发放完成，总计发放利息${total_interest:.2f}")
            else:
                print(f"💰 银行系统未启用，跳过利息发放")
                total_interest = 0.0
        except Exception as e:
            print(f"❌ 银行利息发放处理失败: {e}")
            total_interest = 0.0
        
        bank_interest_duration = time.time() - bank_interest_start
        return total_interest, bank_interest_duration

    async def process_tax_redistribution(self):
        """税收再分配，返回(再分配总额, 耗时秒)"""
        tax_redistribution_start = time.time()
        print(f"\n🏛️ ===== 第 {self.current_month} 月税收再分配 =====")
        
        # 如果策略为 "none"，跳过再分配
        if self.config.redistribution_strategy == "none":
            print(f"ℹ️  当前策略设置为 'none'，跳过税收再分配")
            tax_redistribution_duration = time.time() - tax_redistribution_start
            return 0.0, tax_redistribution_duration
        
        try:
            redistribution_result = await self.economic_center.redistribute_monthly_taxes.remote(
                self.current_month, 
                strategy=self.config.redistribution_strategy,
                poverty_weight=self.config.redistribution_poverty_weight,
                unemployment_weight=self.config.redistribution_unemployment_weight,
                family_size_weight=self.config.redistribution_family_size_weight
            )
            
            print(f"📊 税收再分配详情:")
            tax_breakdown = redistribution_result.get('tax_breakdown', {})
            print(f"   消费税收入: ${tax_breakdown.get('consume_tax', 0):.2f}")
            print(f"   个人所得税收入: ${tax_breakdown.get('labor_tax', 0):.2f}")
            print(f"   企业所得税收入: ${tax_breakdown.get('corporate_tax', 0):.2f}")
            print(f"   税收总额: ${redistribution_result.get('total_tax_collected', 0):.2f}")
            print(f"✅ 再分配完成:")
            print(f"   受益劳动者数量: {redistribution_result.get('recipients', 0)} 个")
            print(f"   人均分配金额: ${redistribution_result.get('per_person', 0):.2f}")
            print(f"   总再分配金额: ${redistribution_result.get('total_redistributed', 0):.2f}")
            
            total_redistributed = redistribution_result.get('total_redistributed', 0)
        except Exception as e:
            print(f"❌ 税收再分配处理失败: {e}")
            total_redistributed = 0.0
        
        tax_redistribution_duration = time.time() - tax_redistribution_start
        return total_redistributed, tax_redistribution_duration

    async def process_production_restocking(self):
        """月度生产补货周期，返回(补货商品数, 耗时秒)"""
        production_start = time.time()
        print(f"\n🏭 ===== 第 {self.current_month} 月生产补货周期 =====")
        
        # 确保发生异常时也有默认值，避免后续引用未定义
        production_stats = {}
        try:
            production_stats = await self.economic_center.execute_monthly_production_cycle.remote(
                month=self.current_month,
                labor_market=self.labor_market,
                product_market=self.product_market,
                std_jobs=std_job,  # 传入标准工作数据
                firms=self.firms,  # 传入企业列表
                # ✨ 传入新版生产配置参数（基于利润和成本）
                production_config={
                    'profit_to_production_ratio': self.config.profit_to_production_ratio,
                    'min_production_per_product': self.config.min_production_per_product,
                    'labor_productivity_factor': self.config.labor_productivity_factor,
                    'labor_elasticity': self.config.labor_elasticity
                },
                innovation_config={
                    'enable_innovation_module': self.config.enable_innovation_module,
                    'innovation_gamma': self.config.innovation_gamma,
                    'policy_encourage_innovation': self.config.policy_encourage_innovation,
                    'innovation_lambda': self.config.innovation_lambda,
                    'innovation_concavity_beta': self.config.innovation_concavity_beta

                }
            )
            
            print(f"📊 生产统计:")
            print(f"   参与公司: {production_stats.get('total_companies', 0)} 家")
            print(f"   有工人公司: {production_stats.get('companies_with_workers', 0)} 家")
            print(f"   基础产出: {production_stats.get('base_production_total', 0):.2f} 单位")
            
            # 显示劳动力效率信息
            firm_labor_efficiency = production_stats.get('firm_labor_efficiency', {})
            if firm_labor_efficiency:
                print(f"\n💼 企业劳动力效率:")
                for firm_id, labor_info in firm_labor_efficiency.items():
                    if isinstance(labor_info, dict):
                        total_emp = labor_info.get('total_employees', 0)
                        effective_labor = labor_info.get('effective_labor', 0.0)
                        avg_match = labor_info.get('avg_match_score', 0.0)
                        print(f"   {firm_id}: {total_emp}名员工 → {effective_labor:.2f}有效劳动力 (平均匹配度: {avg_match:.2f})")
                    else:
                        print(f"   {firm_id}: {labor_info:.2f}有效劳动力")
            print(f"   劳动力产出: {production_stats.get('labor_production_total', 0):.2f} 单位")
            print(f"   补货商品: {production_stats.get('products_restocked', 0)} 种")
            
            # 获取详细统计
            detailed_stats = await self.economic_center.get_production_statistics.remote(self.current_month)
            print(f"   总库存: {detailed_stats.get('total_inventory', 0):.2f} 单位")
            print(f"   低库存商品: {len(detailed_stats.get('low_stock_products', []))} 种")
            print(f"   高库存商品: {len(detailed_stats.get('high_stock_products', []))} 种")
            
            # 打印创新相关记录
            innovation_events = await self.economic_center.query_all_firm_innovation_events.remote()
            if innovation_events:
                for event in innovation_events:
                    # FirmInnovationEvent 使用 company_id 字段
                    company_id = getattr(event, 'company_id', 'unknown')
                    innovation_type = getattr(event, 'innovation_type', 'N/A')
                    month = getattr(event, 'month', 'N/A')
                    print(f"   创新事件: {company_id} 类型={innovation_type} 月份={month}")

            print(f"✅ 第 {self.current_month} 月生产周期完成")
            
            # 保存生产统计数据
            if production_stats:
                self.monthly_production_stats[self.current_month] = production_stats
            
            products_restocked = production_stats.get('products_restocked', 0)
        except Exception as e:
            print(f"❌ 月度生产周期处理失败: {e}")
            # 确保异常时也有可返回的默认结构
            production_stats = {}
            products_restocked = 0
        
        production_duration = time.time() - production_start
        return products_restocked, production_stats, production_duration

    async def get_firms_inventory_value(self) -> Dict[str, float]:
        """统计每家企业所持有商品的总价值"""
        firms_inventory_value = {}
        
        try:
            for firm in self.firms:
                try:
                    # 获取企业产品信息
                    products = await self.economic_center.query_products.remote(firm.company_id)
                    total_value = 0.0
                    
                    if products:
                        for product in products:
                            # 商品总价值 = 数量 × 单价
                            product_value = product.amount * product.price
                            total_value += product_value
                    
                    firms_inventory_value[firm.company_id] = total_value
                    
                except Exception as e:
                    logger.warning(f"获取企业 {firm.company_id} 库存价值失败: {e}")
                    firms_inventory_value[firm.company_id] = 0.0
                    
        except Exception as e:
            logger.error(f"统计企业库存价值失败: {e}")
            
        return firms_inventory_value

    async def process_inherent_market(self):
        """
        固有市场机制：每月固定消耗一定比例的商品，让企业获取收益
        重点关注新生产的商品（库存量较高的商品）
        返回(消耗商品总价值, 消耗商品数量, 耗时秒)
        """
        if not self.config.enable_inherent_market:
            return 0.0, 0, 0.0
            
        inherent_market_start = time.time()
        print(f"\n🛒 ===== 第 {self.current_month} 月固有市场消耗 =====")
        
        total_value_consumed = 0.0
        total_quantity_consumed = 0
        
        try:
            # 1. 统计所有企业的商品库存
            all_products = []
            
            for firm in self.firms:
                try:
                    products = await self.economic_center.query_products.remote(firm.company_id)
                    if products:
                        for product in products:
                            if product.amount > 0:
                                all_products.append({
                                    'product': product,
                                    'company_id': firm.company_id,
                                    'value': product.amount * product.price
                                })
                except Exception as e:
                    logger.warning(f"获取企业 {firm.company_id} 产品失败: {e}")
            
            if not all_products:
                print("⚠️  没有可消耗的商品")
                return 0.0, 0, time.time() - inherent_market_start
            
            # 2. 如果启用优先消耗新生产商品，按库存量排序（库存高的优先）
            if self.config.inherent_market_focus_new_products:
                all_products.sort(key=lambda x: x['product'].amount, reverse=True)
            
            # 3. 消耗商品
            consumption_rate = self.config.inherent_market_consumption_rate
            
            for item in all_products:
                product = item['product']
                company_id = item['company_id']
                
                # 计算本次消耗数量（按比例）
                quantity_to_consume = product.amount * consumption_rate
                
                if quantity_to_consume > 0:
                    # 计算商品价值
                    value = quantity_to_consume * product.price
                    
                    # 更新库存（减少商品数量）
                    try:
                        await self.economic_center.consume_product_inventory.remote(
                            company_id, 
                            product.product_id, 
                            quantity_to_consume
                        )
                        
                        # ���业获得收入（政府作为固有市场的买家）
                        await self.economic_center.record_firm_monthly_income.remote(
                            company_id, 
                            self.current_month, 
                            value
                        )
                        
                        # 从政府账户支付并创建交易记录
                        try:
                            gov_balance = await self.economic_center.query_balance.remote("gov_main_simulation")
                            if gov_balance >= value:
                                # 创建固有市场专属交易记录
                                await self.economic_center.add_inherent_market_transaction.remote(
                                    month=self.current_month,
                                    sender_id="gov_main_simulation",
                                    receiver_id=company_id,
                                    amount=value,
                                    product_id=product.product_id,
                                    quantity=quantity_to_consume,
                                    product_name=product.name,
                                    product_price=product.price,
                                    product_classification=product.classification
                                )
                            else:
                                # 政府余额不足，由系统补充后再交易
                                # 先给政府账户补充资金
                                await self.economic_center.update_balance.remote(
                                    "gov_main_simulation", 
                                    value
                                )
                                # 然后创建交易记录
                                await self.economic_center.add_inherent_market_transaction.remote(
                                    month=self.current_month,
                                    sender_id="gov_main_simulation",
                                    receiver_id=company_id,
                                    amount=value,
                                    product_id=product.product_id,
                                    quantity=quantity_to_consume,
                                    product_name=product.name,
                                    product_price=product.price,
                                    product_classification=product.classification
                                )
                        except Exception as e:
                            logger.warning(f"固有市场交易记录创建失败: {e}")
                        
                        total_value_consumed += value
                        total_quantity_consumed += quantity_to_consume
                        
                    except Exception as e:
                        logger.warning(f"固有市场消耗商品失败 {product.product_id}: {e}")
            
            print(f"📊 固有市场消耗统计:")
            print(f"   消耗商品总价值: ${total_value_consumed:,.2f}")
            print(f"   消耗商品总数量: {total_quantity_consumed:,.2f} 单位")
            print(f"   参与企业数: {len(set(item['company_id'] for item in all_products))} 家")
            
            # 🔄 同步库存到商品市场
            try:
                sync_success = await self.economic_center.sync_product_inventory_to_market.remote(
                    self.product_market
                )
                if sync_success:
                    print(f"   ✅ 库存已同步到商品市场")
                else:
                    print(f"   ⚠️  库存同步失败")
            except Exception as e:
                logger.warning(f"库存同步到商品市场失败: {e}")
                print(f"   ⚠️  库存同步异常: {e}")
            
            print(f"✅ 第 {self.current_month} 月固有市场消耗完成")
            
        except Exception as e:
            print(f"❌ 固有市场处理失败: {e}")
            logger.error(f"固有市场处理失败: {e}")
        
        inherent_market_duration = time.time() - inherent_market_start
        return total_value_consumed, total_quantity_consumed, inherent_market_duration

    async def run_simulation(self):
        """运行经济仿真"""
        logger.info("===== 开始经济仿真 =====")
        
        # 初始化仿真指标记录
        simulation_metrics = []
        
        # 启动监控
        if self.config.enable_monitoring:
            await self.start_monitoring()
        
        try:
            print(f"\n🚀 经济仿真开始 - 北京时间: {self.get_beijing_time()}")
            print(f"📊 仿真配置: {self.config.num_iterations} 个月, {len(self.households)} 个家庭, {len(self.firms)} 家企业")
            print("="*80)
            
            # 运行多轮仿真
            for iteration in range(self.config.num_iterations):
                iteration_start_time = time.time()
                self.current_month = iteration + 1
                
                # ✨ 钩子1: 检查暂停状态
                if self._wrapper:
                    await self._wrapper._check_pause_state()
                
                # ✨ 钩子2: 执行待处理的干预
                if self._wrapper:
                    await self._wrapper._execute_pending_interventions(self.current_month)

                for household in self.households:
                    household.set_current_month(self.current_month)

                print(f"\n📅 第 {self.current_month}/{self.config.num_iterations} 月仿真开始 - {self.get_beijing_time()}")
                logger.info(f"===== 仿真迭代 {self.current_month}/{self.config.num_iterations} =====")
                
                # 📦 月初统计企业库存价值
                print(f"\n📦 ===== 第 {self.current_month} 月初企业库存统计 =====")
                firms_inventory_value = await self.get_firms_inventory_value()
                total_inventory_value = sum(firms_inventory_value.values())
                print(f"📊 企业库存总价值: ${total_inventory_value:,.2f}")
                
                # 显示库存价值最高的前5家企业
                top_firms = sorted(firms_inventory_value.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_firms:
                    print(f"🏆 库存价值TOP 5企业:")
                    for i, (firm_id, value) in enumerate(top_firms, 1):
                        print(f"   {i}. {firm_id}: ${value:,.2f}")
                print(f"{'='*50}")
                
                # 初始化本月各阶段时间统计
                stage_timings = {
                    'dismissal': 0.0,
                    'job_posting': 0.0,
                    'job_matching': 0.0,
                    'firm_decisions': 0.0,
                    'household_evaluation': 0.0,
                    'hiring_confirmation': 0.0,
                    'household_consumption': 0.0,
                    'price_update': 0.0,
                    'wage_processing': 0.0,
                    'skill_enhancement': 0.0,
                    'bank_interest': 0.0,
                    'tax_redistribution': 0.0,
                    'production_restocking': 0.0,
                    'inherent_market': 0.0,
                    'industry_competition': 0.0,  # 🆕 行业竞争分析
                    'innovation_export': 0.0,  # 🆕 创新数据导出
                    'monthly_summary': 0.0
                }
                
                # 1. 更新储蓄信息
                await self.update_deposit()
                
                # 2. 月初辞退 - ✅ 启用
                print(f"\n🔥 步骤2：月初辞退")
                unemployment_data, dismissal_duration = await self.handle_dismissal()
                stage_timings['dismissal'] = dismissal_duration
                
                # 3. 岗位发布 - ✅ 启用
                print(f"\n📢 步骤3：岗位发布")
                firms_to_post_jobs, total_job_postings, job_posting_duration = await self.post_jobs(unemployment_data)
                stage_timings['job_posting'] = job_posting_duration

                # 4. 家庭匹配工作 - ✅ 启用
                print(f"\n🔍 步骤4：家庭匹配工作")
                total_applications, job_matching_duration = await self.match_jobs()
                stage_timings['job_matching'] = job_matching_duration
                
                # 5. 企业招聘决策 - ✅ 启用
                print(f"\n🏢 步骤5：企业招聘决策")
                all_hiring_decisions, job_offers, firm_decisions_duration = await self.process_firm_hiring_decisions(firms_to_post_jobs)
                stage_timings['firm_decisions'] = firm_decisions_duration
                
                # 6. 家庭评估offers - ✅ 启用
                print(f"\n💼 步骤6：家庭评估job offers")
                accepted_offers, household_evaluation_duration = await self.process_household_offer_evaluation(job_offers)
                stage_timings['household_evaluation'] = household_evaluation_duration
                
                # 7. 最终雇佣确认 - ✅ 启用
                print(f"\n✅ 步骤7：最终雇佣确认")
                total_aligned_job, hiring_confirmation_duration = await self.process_hiring_confirmation(accepted_offers)
                stage_timings['hiring_confirmation'] = hiring_confirmation_duration
                
                # 8. 家庭消费
                successful_consumptions, household_consumption_duration = await self.process_household_consumption()
                stage_timings['household_consumption'] = household_consumption_duration
                
                # 9. 更新商品市场价格
                updated_products, price_update_duration = await self.update_product_prices()
                stage_timings['price_update'] = price_update_duration
                
                # 10. 工资发放 - ✅ 启用
                print(f"\n💵 步骤10：工资发放")
                matched_jobs_count, wage_processing_duration = await self.process_wage_payment_and_tracking()
                stage_timings['wage_processing'] = wage_processing_duration
                
                # 11. 技能提升 - ✅ 启用
                print(f"\n📚 步骤11：技能提升")
                total_skill_enhancements, skill_enhancement_duration = await self.process_skill_enhancement()
                stage_timings['skill_enhancement'] = skill_enhancement_duration
                
                # 12. 银行利息 - ✅ 启用
                print(f"\n🏦 步骤12：银行利息发放")
                total_interest, bank_interest_duration = await self.process_bank_interest()
                stage_timings['bank_interest'] = bank_interest_duration
                
                # 13. 税收再分配 - ✅ 启用
                print(f"\n🎯 步骤13：税收再分配")
                total_redistributed, tax_redistribution_duration = await self.process_tax_redistribution()
                stage_timings['tax_redistribution'] = tax_redistribution_duration
                
                # 14. 固有市场消耗（解决商品积压问题）
                # 💡 必须在生产补货之前执行，确保生产决策时能计入固定市场收入
                inherent_value, inherent_quantity, inherent_duration = await self.process_inherent_market()
                stage_timings['inherent_market'] = inherent_duration
                
                # 15. 月度生产补货周期（使用包含固定市场收入的完整利润数据）
                print(f"\n🏭 步骤15：月度生产补货（补充商品库存）")
                products_restocked, production_stats, production_duration = await self.process_production_restocking()
                stage_timings['production_restocking'] = production_duration
                
                # 16. 家庭属性系统月度更新 - ✅ 新增（v4.0社会比较功能）
                print(f"\n📊 步骤16：家庭属性系统月度更新（包含社会比较）")
                attribute_update_start = time.time()
                
                # 收集所有家庭的属性系统
                all_family_systems = []
                for household in self.households:
                    if hasattr(household, 'attribute_system') and household.attribute_system:
                        all_family_systems.append(household.attribute_system)
                
                # 统一执行月度更新（传入所有家庭以启用社会比较）
                for household in self.households:
                    if hasattr(household, 'attribute_system') and household.attribute_system:
                        try:
                            # 调用属性系统的月度更新（传入所有家庭）
                            household.attribute_system.monthly_update(
                                new_month=self.current_month,
                                all_families=all_family_systems
                            )
                            # 保存状态
                            household.attribute_system.save_to_file()
                        except Exception as e:
                            logger.error(f"❌ 家庭 {household.household_id} 属性系统月度更新失败: {e}")
                
                attribute_update_duration = time.time() - attribute_update_start
                stage_timings['attribute_update'] = attribute_update_duration
                print(f"   ✅ {len(all_family_systems)}个家庭属性系统更新完成，耗时{attribute_update_duration:.2f}秒")

                # 17. 🆕 行业竞争分析（市场份额统计）
                print(f"\n📊 步骤17：行业竞争分析（市场份额统计）")
                competition_start = time.time()

                try:
                    # 延迟初始化：确保输出目录指向实验目录
                    if self.competition_analyzer is None:
                        competition_output_dir = os.path.join(self.experiment_output_dir, "industry_competition")
                        self.competition_analyzer = IndustryCompetitionAnalyzer(
                            output_dir=competition_output_dir,
                            economic_center=self.economic_center,
                            use_timestamp=False  # 不使用时间戳，使用实验名称
                        )
                        # 注册行业-企业映射
                        self.competition_analyzer.register_industry_firms(self.firms)
                    
                    await self.competition_analyzer.analyze_monthly_competition(
                        self.economic_center,
                        self.current_month,
                        production_stats=production_stats  # 传递生产统计数据
                    )
                    competition_duration = time.time() - competition_start
                    stage_timings['industry_competition'] = competition_duration
                    print(f"   ✅ 行业竞争分析完成，耗时: {competition_duration:.2f}秒")
                except Exception as e:
                    logger.error(f"   ❌ 行业竞争分析失败: {e}")
                    stage_timings['industry_competition'] = time.time() - competition_start

                # 🆕 步骤18：导出创新数据报告
                print(f"\n📄 步骤18：导出创新数据报告")
                innovation_export_start = time.time()

                try:
                    # 延迟初始化：确保输出目录指向实验目录
                    if self.innovation_exporter is None:
                        innovation_output_dir = os.path.join(self.experiment_output_dir, "innovation_reports")
                        self.innovation_exporter = InnovationDataExporter(
                            output_dir=innovation_output_dir
                        )
                    
                    await self.innovation_exporter.export_monthly_innovation_report(
                        self.economic_center,
                        self.current_month,
                        self.config,
                        production_stats,  # 传递生产统计数据
                        self.firms  # 🆕 传递企业列表
                    )
                    innovation_export_duration = time.time() - innovation_export_start
                    stage_timings['innovation_export'] = innovation_export_duration
                    print(f"   ✅ 创新数据报告已导出，耗时: {innovation_export_duration:.2f}秒")
                except Exception as e:
                    logger.error(f"   ❌ 创新数据导出失败: {e}")
                    stage_timings['innovation_export'] = time.time() - innovation_export_start

                end_time = time.time()
                
                simulation_metrics.append({
                    "iteration": self.current_month,
                    "duration": end_time - iteration_start_time,
                    "jobs_aligned": total_aligned_job,
                    "households_processed": len(self.households),
                    "firms_used": len(firms_to_post_jobs)
                })
                
                logger.info(f"仿真迭代 {self.current_month} 完成: 对齐工作 {total_aligned_job} 个, 使用企业 {len(firms_to_post_jobs)} 家, 耗时: {end_time - iteration_start_time:.2f}秒")
                print(f"✅ 第 {self.current_month} 月仿真完成 - {self.get_beijing_time()} (耗时: {end_time - iteration_start_time:.2f}秒)")
                
                # 打印LLM缓存统计
                from agentsociety_ecosim.consumer_modeling import llm_utils
                cache_stats = llm_utils.get_llm_cache_stats()
                print(f"📊 LLM缓存统计: {cache_stats['hits']}次命中/{cache_stats['total_requests']}次请求 (命中率: {cache_stats['hit_rate']})")

                # 收集所有经济指标和月度统计数据（包括商品销售、库存、价格等）
                monthly_indicators = await self._collect_indicators_and_monthly(self.current_month, self.households, self.firms, total_job_postings)
                self.economic_metrics_history.append(monthly_indicators)
                
                # 收集家庭购买记录
                await self._collect_household_purchase_records(self.current_month)
                
                # 每个月打印月度统计信息（新版综合报告）
                monthly_summary_start = time.time()
                await self._print_monthly_summary(self.current_month)
                monthly_summary_duration = time.time() - monthly_summary_start
                stage_timings['monthly_summary'] = monthly_summary_duration
                
                # 显示本月各阶段时间统计汇总
                print(f"\n{'='*60}")
                print(f"⏱️  第 {self.current_month} 月各阶段运行时间统计")
                print(f"{'='*60}")
                
                total_iteration_time = time.time() - iteration_start_time
                stage_timings['total_iteration'] = total_iteration_time
                
                # 按时间排序显示各阶段耗时
                sorted_stages = sorted(stage_timings.items(), key=lambda x: x[1], reverse=True)
                
                print(f"📊 各阶段耗时排名:")
                for i, (stage_name, duration) in enumerate(sorted_stages, 1):
                    if stage_name != 'total_iteration':
                        percentage = (duration / total_iteration_time * 100) if total_iteration_time > 0 else 0
                        stage_name_zh = {
                            'dismissal': '辞退处理',
                            'job_posting': '岗位发布',
                            'job_matching': '工作匹配',
                            'firm_decisions': '企业招聘决策',
                            'household_evaluation': '家庭评估',
                            'hiring_confirmation': '雇佣确认',
                            'household_consumption': '家庭消费',
                            'price_update': '价格更新',
                            'wage_processing': '工资处理',
                            'skill_enhancement': '技能提升',
                            'bank_interest': '银行利息',
                            'tax_redistribution': '税收再分配',
                            'production_restocking': '生产补货',
                            'inherent_market': '固有市场',
                            'industry_competition': '行业竞争分析',  # 🆕
                            'monthly_summary': '月度统计'
                        }.get(stage_name, stage_name)
                        
                        print(f"  {i:2d}. {stage_name_zh:12} : {duration:8.3f}秒 ({percentage:5.1f}%)")
                
                print(f"\n⏱️  本月总耗时: {total_iteration_time:.3f}秒")
                print(f"📈 平均每阶段耗时: {total_iteration_time / (len(stage_timings) - 1):.3f}秒")
                print(f"{'='*60}")


            # 仿真完成，进行最终结算
            print(f"\n🏁 所有仿真迭代完成 - 北京时间: {self.get_beijing_time()}")
            print("="*80)
            logger.info("===== 仿真完成，进行最终结算 =====")

            # 🆕 生成行业竞争汇总报告
            print(f"\n📊 生成行业竞争汇总报告和趋势图...")
            try:
                self.competition_analyzer.generate_summary_report()
                print(f"   ✅ 行业竞争汇总报告生成完成")

                # 生成带创新标注的趋势图
                print(f"   📈 正在生成带创新标注的趋势图...")
                await self.competition_analyzer.generate_trend_charts_async(self.economic_center)
                print(f"   ✅ 趋势图生成完成")
            except Exception as e:
                logger.error(f"   ❌ 行业竞争汇总报告生成失败: {e}")

            await self._final_settlement()
            
            # 生成数据可视化图表
            print(f"\n📊 开始生成数据可视化图表...")
            print("="*80)
            await self._generate_all_visualization_charts()
            
        except Exception as e:
            logger.error(f"仿真运行过程中出错: {e}")
            raise
        finally:
            # 停止监控
            if self.config.enable_monitoring:
                await self.stop_monitoring()

    async def _collect_indicators_and_monthly(
        self,
        month: Optional[int] = None,
        households: Optional[List] = None,
        firms: Optional[List] = None,
        job_postings: int = 0
    ) -> Dict[str, Any]:
        """
        统一收集函数：一次性收集所有经济指标和月度统计数据。
        
        收集内容：
        1) 家庭经济指标（收入、支出、财富、就业）
        2) 企业月度指标（收入、支出、利润、生产、招聘）
        3) 商品详细数据（销售、库存、价格、供需）
        4) 失业和空缺岗位统计
        5) 企业营业率
        
        参数：
        - month: 月份（为空时使用 self.config.num_iterations）
        - households: 家庭列表（为空时使用 self.households）
        - firms: 企业列表（为空时使用 self.firms）
        - job_postings: 当月岗位发布数
        
        返回：
        - economic_indicators: 经济指标字典
        
        副作用：
        - 追加到 self.household_monthly_metrics[month]
        - 追加到 self.firm_monthly_metrics
        - 保存到 self.monthly_product_sales[month]
        - 保存到 self.monthly_product_inventory[month]
        - 保存到 self.monthly_product_prices[month]
        - 保存到 self.monthly_firm_operation_rate[month]
        - 保存到 self.monthly_supply_demand[month]
        """

        # ---------------------------
        # 通用小工具：兼容 awaitable / Ray ObjectRef
        # ---------------------------
        async def await_maybe(x):
            # 原生 awaitable
            if hasattr(x, "__await__"):
                return await x
            # 可能是 Ray ObjectRef
            try:
                import ray  # 局部导入，避免无 Ray 环境时报错
                return await asyncio.to_thread(ray.get, x)
            except Exception:
                # 既不是 awaitable、也不是可 ray.get 的对象
                return x

        try:
            # ---------------------------
            # 输入准备
            # ---------------------------
            if month is None:
                month = getattr(self.config, "num_iterations", 1)

            if households is None:
                households = getattr(self, "households", []) or []
            if firms is None:
                firms = getattr(self, "firms", []) or []

            total_households = len(households)

            # 先收集当月的再分配金额（从交易记录中获取实际分配金额）
            household_redistribution_amounts = {}
            try:
                all_transactions = await self.economic_center.query_all_tx.remote()
                for tx in all_transactions:
                    if tx.month == month and tx.type == 'redistribution':
                        household_redistribution_amounts[tx.receiver_id] = household_redistribution_amounts.get(tx.receiver_id, 0.0) + tx.amount
            except Exception as e:
                logger.warning(f"获取再分配交易记录失败: {e}")
                household_redistribution_amounts = {}

            # ---------------------------
            # 1) 并发收集"家庭快照"（总体统计所需）
            #    settlement(月度结算汇总)、monthly_stats(指定月汇总)、余额
            # ---------------------------
            print(f"📊 开始并行收集 {len(households)} 个家庭的经济指标与{month}月度数据...")

            async def collect_household_snapshot(hh):
                try:
                    monthly_task = self.economic_center.compute_household_monthly_stats.remote(hh.household_id, month)

                    # 并发等待（兼容 awaitable 与 Ray ObjectRef）
                    r = await monthly_task

                    monthly_income, monthly_expenditure, current_wealth = (r if not isinstance(r, Exception) else ({}, {}, 0))

                    monthly_income = monthly_income if isinstance(monthly_income, float) else 0
                    monthly_expenditure = monthly_expenditure if isinstance(monthly_expenditure, float) else 0

                    # 劳动力统计（与原综合口径保持一致）
                    labor_force = 0
                    employed_people = 0
                    if hasattr(hh, "labor_hours") and hh.labor_hours:
                        labor_force = len(hh.labor_hours)
                        for lh in hh.labor_hours:
                            if hasattr(lh, "is_valid") and hasattr(lh, "company_id"):
                                if not lh.is_valid and lh.company_id is not None:
                                    employed_people += 1
                    # 获取该家庭实际收到的再分配金额
                    redistribution_amount = household_redistribution_amounts.get(hh.household_id, 0.0)
                    return {
                        "household_id": hh.household_id,
                        "monthly_income": monthly_income,
                        "monthly_expenditure": monthly_expenditure,
                        "wealth": current_wealth,
                        "labor_force": labor_force,
                        "employed_people": employed_people,
                        "is_employed": employed_people > 0,
                        "redistribution_amount": redistribution_amount,
                        "hh": hh,
                    }
                except Exception as e:
                    logger.warning(f"收集家庭 {getattr(hh,'household_id', 'unknown')} 数据失败: {e}")
                    return {
                        "household_id": getattr(hh, "household_id", "unknown"),
                        "income": 0,
                        "spent": 0,
                        "monthly_income": 0,
                        "monthly_expenditure": 0,
                        "wealth": 0,
                        "labor_force": 0,
                        "employed_people": 0,
                        "is_employed": False,
                        "redistribution_amount": 0,
                        "hh": hh,
                    }

            snapshot_tasks = [collect_household_snapshot(h) for h in households]
            snapshots = await asyncio.gather(*snapshot_tasks, return_exceptions=True)

            ok_snapshots = [s for s in snapshots if s and not isinstance(s, Exception)]
            print(f"✅ 经济指标/快照收集完成: {len(ok_snapshots)}/{len(households)} 个家庭")

            # ---------------------------
            # 2) 汇总总体经济指标（沿用你原有口径）
            # ---------------------------
            # total_income = sum(s["income"] for s in ok_snapshots)
            # total_expenditure = sum(s["spent"] for s in ok_snapshots)
            total_monthly_income = sum(s["monthly_income"] + s["redistribution_amount"] for s in ok_snapshots)
            total_monthly_expenditure = sum(s["monthly_expenditure"] for s in ok_snapshots)
            total_labor_force_available = sum(s["labor_force"] for s in ok_snapshots)
            total_labor_force_employed = sum(s["employed_people"] for s in ok_snapshots)
            employed_households = sum(1 for s in ok_snapshots if s["is_employed"])

            # household_income_data = [s["income"] for s in ok_snapshots] + [0] * (total_households - len(ok_snapshots))
            # household_expenditure_data = [s["spent"] for s in ok_snapshots] + [0] * (total_households - len(ok_snapshots))
            household_monthly_income_data = [s["monthly_income"] + s["redistribution_amount"] for s in ok_snapshots] + [0] * (total_households - len(ok_snapshots))
            household_monthly_expenditure_data = [s["monthly_expenditure"] for s in ok_snapshots] + [0] * (total_households - len(ok_snapshots))
            wealth_distribution = [s["wealth"] for s in ok_snapshots] + [0] * (total_households - len(ok_snapshots))
            
            labor_utilization_rate = (
                total_labor_force_employed / total_labor_force_available if total_labor_force_available > 0 else 0
            )
            labor_unemployment_rate = 1 - labor_utilization_rate
            household_employment_rate = (employed_households / total_households) if total_households > 0 else 0
            household_unemployment_rate = 1 - household_employment_rate

            # avg_income = (total_income / total_households) if total_households > 0 else 0
            # avg_expenditure = (total_expenditure / total_households) if total_households > 0 else 0
            avg_monthly_income = (total_monthly_income / total_households) if total_households > 0 else 0
            avg_monthly_income_per_lh = (total_monthly_income / total_labor_force_available) if total_labor_force_available > 0 else 0
            avg_monthly_expenditure = (total_monthly_expenditure / total_households) if total_households > 0 else 0
            avg_wealth = sum(s['wealth'] for s in ok_snapshots) / len(ok_snapshots)
            avg_monthly_expenditure_income_ratio = (avg_monthly_expenditure / avg_monthly_income) if avg_monthly_income > 0 else 0

            wealth_sorted = sorted(wealth_distribution)
            median_wealth = wealth_sorted[len(wealth_sorted)//2] if wealth_sorted else 0
            gini_coefficient = self._calculate_gini_coefficient(wealth_sorted)

            # 收入分布
            income_ranges = {"0-1000": 0, "1000-5000": 0, "5000-10000": 0, "10000-50000": 0, "50000+": 0}
            for inc in household_monthly_income_data:
                if inc <= 1000:
                    income_ranges["0-1000"] += 1
                elif inc <= 5000:
                    income_ranges["1000-5000"] += 1
                elif inc <= 10000:
                    income_ranges["5000-10000"] += 1
                elif inc <= 50000:
                    income_ranges["10000-50000"] += 1
                else:
                    income_ranges["50000+"] += 1

            # 家庭财务健康
            healthy_households = sum(1 for a, b in zip(household_monthly_income_data, household_monthly_expenditure_data) if a > b)
            deficit_households = sum(1 for a, b in zip(household_monthly_income_data, household_monthly_expenditure_data) if a < b)
            balanced_households = total_households - healthy_households - deficit_households

            job_info = await self.labor_market.get_total_job_positions.remote()
            vacant_positions = job_info['total_positions']
            total_positions = vacant_positions + total_labor_force_employed
            economic_indicators = {
                "employment_statistics": {
                    "total_households": total_households,
                    "employed_households": employed_households,
                    "unemployed_households": total_households - employed_households,
                    "household_employment_rate": household_employment_rate,
                    "household_unemployment_rate": household_unemployment_rate,
                    "total_labor_force_available": total_labor_force_available,
                    "total_labor_force_employed": total_labor_force_employed,
                    "total_labor_force_unemployed": total_labor_force_available - total_labor_force_employed,
                    "labor_utilization_rate": labor_utilization_rate,
                    "labor_unemployment_rate": labor_unemployment_rate,
                    "avg_labor_force_per_household": (total_labor_force_available / total_households) if total_households > 0 else 0,
                    "avg_employed_people_per_household": (total_labor_force_employed / total_households) if total_households > 0 else 0,
                    "total_job_positions": total_positions,
                    "job_fill_rate": total_labor_force_employed / total_positions if total_positions > 0 else 0

                },
                "income_expenditure_analysis": {
                    "expenditure_income_ratio": (avg_monthly_expenditure / avg_monthly_income) if (avg_monthly_expenditure / avg_monthly_income) <= 1 else 0,
                    "current_month": month,
                    "total_monthly_income": total_monthly_income,
                    "monthly_redistribution_amount": sum(household_redistribution_amounts.values()),
                    "total_monthly_expenditure": total_monthly_expenditure,
                    "average_monthly_income": avg_monthly_income,
                    "average_monthly_income_per_lh": avg_monthly_income_per_lh,
                    "average_monthly_expenditure": avg_monthly_expenditure,
                    "monthly_expenditure_income_ratio": (avg_monthly_expenditure / avg_monthly_income) if avg_monthly_income > 0 else 0,
                    "monthly_savings_rate": ((avg_monthly_income - avg_monthly_expenditure) / avg_monthly_income) if avg_monthly_income > 0 else 0
                },
                "wealth_distribution": {
                    "average_wealth": avg_wealth,
                    "median_wealth": median_wealth,
                    "gini_coefficient": gini_coefficient,
                    "wealth_range": {
                        "min": min(wealth_sorted) if wealth_sorted else 0,
                        "max": max(wealth_sorted) if wealth_sorted else 0
                    }
                },
                "income_distribution": income_ranges,
                "household_financial_health": {
                    "healthy_households": healthy_households,
                    "deficit_households": deficit_households,
                    "balanced_households": balanced_households,
                    "healthy_rate": (healthy_households / total_households) if total_households > 0 else 0,
                    "deficit_rate": (deficit_households / total_households) if total_households > 0 else 0
                }
            }

            # ---------------------------
            # 3) 基于 snapshots 生成家庭"月度指标"对象并保存
            # ---------------------------
            # 计算 month-1 的收入（如果后面需要变化率）
            prev_income_map = {}
            if month > 1:
                # 并发取上月统计（可与企业统计并行，这里优先复用快照中的 hh 引用）
                async def fetch_prev_month(hh):
                    try:
                        prev_task = self.economic_center.compute_household_monthly_stats.remote(hh.household_id, month - 1)
                        prev_result = await await_maybe(prev_task)
                        if not isinstance(prev_result, Exception) and len(prev_result) >= 3:
                            prev_income, _prev_spent, _prev_wealth = prev_result
                            return hh.household_id, prev_income
                        else:
                            return hh.household_id, 0.0
                    except Exception:
                        return hh.household_id, 0.0

                prev_tasks = [fetch_prev_month(s["hh"]) for s in ok_snapshots]
                prev_results = await asyncio.gather(*prev_tasks, return_exceptions=True)
                for r in prev_results:
                    if isinstance(r, Exception): 
                        continue
                    hid, prev_income_value = r
                    prev_income_map[hid] = prev_income_value

            valid_metrics = []
            monthly_consumption_structure = {}

            for s in ok_snapshots:
                hh = s["hh"]
                monthly_income = s["monthly_income"]
                monthly_expenditure = s["monthly_expenditure"]
                savings_rate = (monthly_income - monthly_expenditure) / monthly_income if monthly_income > 0 else 0

                # 收入变化率
                income_change_rate = 0.0
                if month > 1:
                    # prev_income_map 现在直接存储上个月的收入值（float），不再是字典
                    prev_income = prev_income_map.get(s["household_id"], 0.0)
                    if prev_income > 0:
                        income_change_rate = (monthly_income - prev_income) / prev_income

                # 消费结构：优先真实预算
                try:
                    consume_budget_data = hh.get_consume_budget_data()
                    if isinstance(consume_budget_data, dict) and month in consume_budget_data:
                        consumption_structure = consume_budget_data[month]
                        for category, amount in consumption_structure.items():
                            if category not in monthly_consumption_structure:
                                monthly_consumption_structure[category] = 0
                            monthly_consumption_structure[category] += amount
                    else:
                        # 备选：按比例估算
                        m = monthly_expenditure
                        consumption_structure = {
                            "food": m * 0.25, "housing": m * 0.30, "transportation": m * 0.15,
                            "entertainment": m * 0.10, "clothing": m * 0.08,
                            "healthcare": m * 0.07, "education": m * 0.05
                        }
                except Exception as e:
                    logger.warning(f"获取家庭 {s['household_id']} 第{month}月消费预算失败: {e}")
                    m = monthly_expenditure
                    consumption_structure = {
                        "food": m * 0.25, "housing": m * 0.30, "transportation": m * 0.15,
                        "entertainment": m * 0.10, "clothing": m * 0.08,
                        "healthcare": m * 0.07, "education": m * 0.05
                    }

                # 与总体统计口径一致的就业人数
                household_labor_hours = s["labor_force"]
                household_employees = s["employed_people"]

                try:
                    metric = HouseholdMonthlyMetrics(
                        household_id=s["household_id"],
                        month=month,
                        monthly_income=monthly_income,
                        monthly_redistribution_amount=s["redistribution_amount"],
                        monthly_expenditure=monthly_expenditure,
                        savings_rate=savings_rate,
                        consumption_structure=consumption_structure,
                        income_change_rate=income_change_rate,
                        household_labor_hours=household_labor_hours,
                        household_employees=household_employees,
                        current_savings=s["wealth"]
                    )
                    valid_metrics.append(metric)
                except Exception as e:
                    logger.warning(f"构建家庭 {s['household_id']} 月度指标失败: {e}")

            # 确保容器存在
            if not hasattr(self, "household_monthly_metrics"):
                self.household_monthly_metrics = {}
            if month not in self.household_monthly_metrics:
                self.household_monthly_metrics[month] = []
            self.household_monthly_metrics[month].extend(valid_metrics)
            
            print(f"✅ 家庭月度数据收集完成: {len(valid_metrics)}/{len(households)} 个家庭数据收集成功")

            # ---------------------------
            # 4) 企业月度指标（占位逻辑沿用原实现）
            # ---------------------------
            for firm in firms:
                try:
                    # 获取企业真实月度财务数据
                    company_id = getattr(firm, "company_id", getattr(firm, "firm_id", "unknown"))
                    try:
                        monthly_financials = await self.economic_center.query_firm_monthly_financials.remote(company_id, month)
                        monthly_revenue = monthly_financials.get("monthly_income", 0.0)
                        monthly_expenses = monthly_financials.get("monthly_expenses", 0.0)
                        monthly_profit = monthly_financials.get("monthly_profit", 0.0)
                        
                        # 计算企业库存总价值
                        inventory_value = 0.0
                        try:
                            products = await self.economic_center.query_products.remote(company_id)
                            if products:
                                for product in products:
                                    inventory_value += product.amount * product.price
                        except Exception as e:
                            logger.debug(f"获取企业 {company_id} 库存价值失败: {e}")
                        
                        logger.info(f"企业 {company_id} 第{month}月财务: 收入${monthly_revenue:.2f}, 支出${monthly_expenses:.2f}, 利润${monthly_profit:.2f}, 库存价值${inventory_value:.2f}")
                    except Exception as e:
                        logger.warning(f"获取企业 {company_id} 第{month}月财务数据失败: {e}")
                        monthly_revenue = 0.0
                        monthly_expenses = 0.0
                        monthly_profit = 0.0
                    current_employees = 0
                    if hasattr(firm, "employees") and firm.employees:
                        current_employees = firm.employees

                    # 统计本月成功招聘数量
                    successful_hires = 0
                    
                    # 从雇佣确认结果中统计该企业的成功招聘数量
                    if hasattr(self, 'confirmed_hires_for_month') and month in self.confirmed_hires_for_month:
                        confirmed_hires = self.confirmed_hires_for_month[month]
                        for hire in confirmed_hires:
                            if hire.get("company_id") == company_id:
                                successful_hires += 1
                    
                    # 如果没有雇佣确认数据，尝试从劳动力市场获取
                    if successful_hires == 0 and hasattr(self, 'labor_market'):
                        try:
                            # 获取该企业的已匹配工作数量（只统计本月的）
                            matched_jobs = await self.labor_market.get_matched_jobs_for_firm.remote(company_id)
                            if matched_jobs:
                                # 只统计本月的匹配工作，避免累积数据
                                current_month_matches = [job for job in matched_jobs if hasattr(job, 'month') and job.month == month]
                                successful_hires = len(current_month_matches) if current_month_matches else 0
                        except Exception as e:
                            logger.debug(f"获取企业 {company_id} 匹配工作数量失败: {e}")
                    
                    # 统计本月实际发布的岗位数量
                    job_postings = 0
                    opening_jobs = firm.opening_jobs
                    for job in opening_jobs:
                        job_postings += job.positions_available
                    job_postings += firm.employees
                    
                    recruitment_success_rate = (successful_hires / job_postings) if job_postings > 0 else 0

                    metric_firm = FirmMonthlyMetrics(
                        company_id=company_id,
                        month=month,
                        monthly_revenue=monthly_revenue,
                        monthly_expenses=monthly_expenses,
                        monthly_profit=monthly_profit,
                        current_employees=current_employees,
                        job_postings=job_postings,  # 使用实际统计的岗位数量
                        successful_hires=successful_hires,
                        recruitment_success_rate=recruitment_success_rate
                    )
                    if not hasattr(self, "firm_monthly_metrics"):
                        self.firm_monthly_metrics = []
                    self.firm_monthly_metrics.append(metric_firm)
                except Exception as e:
                    logger.warning(f"收集企业 {getattr(firm,'company_id','unknown')} 月度数据失败: {e}")

            logger.info(
                f"综合采集完成：家庭失业率={household_unemployment_rate:.2%}, "
                f"劳动力利用率={labor_utilization_rate:.2%}, 收支比={avg_monthly_expenditure_income_ratio:.2f}; "
                f"{month}月家庭={len(households)}，企业={len(firms)}"
            )
            # 把monthly_consumption_stucture加到economic_indicators的"income_expendicture_analysis"里
            economic_indicators["income_expenditure_analysis"]["monthly_consumption_structure"] = monthly_consumption_structure

            # 5) 统计商品平均价格
            avg_price = await self.product_market.get_avg_price.remote()
            economic_indicators["income_expenditure_analysis"]["avg_price"] = avg_price
            
            # 6) 计算基尼系数（再分配前后）
            # 再分配前收入（不包含redistribution_amount）
            pre_redistribution_incomes = [s["monthly_income"] for s in ok_snapshots]
            # 再分配后收入（包含redistribution_amount）
            post_redistribution_incomes = [s["monthly_income"] + s["redistribution_amount"] for s in ok_snapshots]
            
            gini_pre_redistribution = self._calculate_gini_coefficient(pre_redistribution_incomes)
            gini_post_redistribution = self._calculate_gini_coefficient(post_redistribution_incomes)
            
            # 7) 计算平均工资（按工作人数）
            total_workers = total_labor_force_employed
            total_wage_payments = 0.0
            
            # 从经济中心获取当月工资支付总额
            for tx in await self.economic_center.query_all_tx.remote():
                if tx.month == month and tx.type == 'labor_payment':
                    total_wage_payments += tx.amount
            
            average_wage = total_wage_payments / total_workers if total_workers > 0 else 0.0
            
            # 添加到经济指标
            economic_indicators["inequality_analysis"] = {
                "gini_pre_redistribution": gini_pre_redistribution,
                "gini_post_redistribution": gini_post_redistribution,
                "gini_improvement": gini_pre_redistribution - gini_post_redistribution,
                "average_wage_per_worker": average_wage,
                "total_wage_payments": total_wage_payments,
                "total_workers": total_workers
            }
            
            # 记录历史数据
            self.gini_history.append({
                "month": month,
                "gini_pre": gini_pre_redistribution,
                "gini_post": gini_post_redistribution,
                "improvement": gini_pre_redistribution - gini_post_redistribution
            })
            
            self.wage_history.append({
                "month": month,
                "average_wage": average_wage,
                "total_wage_payments": total_wage_payments,
                "total_workers": total_workers
            })

            # ---------------------------
            # 5) 收集商品相关详细统计（合并自_collect_monthly_statistics）
            # ---------------------------
            print(f"📦 收集商品销售、库存、价格等详细数据...")
            
            # 5.1 失业和空缺岗位统计（使用已收集的数据）
            self.monthly_unemployment_stats[month] = {
                'total_unemployed': economic_indicators['employment_statistics']['total_labor_force_unemployed'],
                'unemployment_rate': economic_indicators['employment_statistics']['labor_unemployment_rate'],
                'unemployed_details': []
            }
            
            total_vacant = economic_indicators['employment_statistics']['total_job_positions'] - economic_indicators['employment_statistics']['total_labor_force_employed']
            self.monthly_vacant_jobs[month] = {
                'total_vacant_jobs': total_vacant,
                'vacant_jobs_details': []
            }
            
            # 5.2 企业收入统计（使用已收集的firm_monthly_metrics）
            firm_revenues = {}
            firm_metrics_this_month = [m for m in self.firm_monthly_metrics if m.month == month]
            for metric in firm_metrics_this_month:
                firm_revenues[metric.company_id] = {
                    'revenue': metric.monthly_revenue,
                    'expenses': metric.monthly_expenses,
                    'profit': metric.monthly_profit
                }
            
            # 5.3 收集商品信息（库存、价格）
            product_sales = {}
            product_inventory = {}
            product_prices = {}
            
            for firm in firms:
                try:
                    products = await self.economic_center.query_products.remote(firm.company_id)
                    if products:
                        for product in products:
                            product_id = product.product_id
                            # 商品库存
                            product_inventory[product_id] = {
                                'name': product.name,
                                'quantity': product.amount,
                                'company_id': firm.company_id
                            }
                            # 商品价格
                            product_prices[product_id] = {
                                'name': product.name,
                                'price': product.price,
                                'company_id': firm.company_id
                            }
                except Exception as e:
                    logger.warning(f"获取企业 {firm.company_id} 商品数据失败: {e}")
                    continue
            
            # 5.4 商品销售统计（从家庭购买记录中统计）
            total_records_checked = 0
            total_records_matched = 0
            for household in households:
                try:
                    for record in getattr(household, 'purchase_history', []) or []:
                        total_records_checked += 1
                        record_month = getattr(record, 'month', None)
                        if record_month == month:
                            total_records_matched += 1
                            product_id = getattr(record, 'product_id', None)
                            if product_id:
                                if product_id not in product_sales:
                                    product_sales[product_id] = {
                                        'name': getattr(record, 'product_name', 'Unknown'),
                                        'total_quantity': 0,
                                        'total_revenue': 0,
                                        'purchase_count': 0,
                                        'household_quantity': 0,
                                        'inherent_market_quantity': 0
                                    }
                                quantity = getattr(record, 'quantity', 0)
                                product_sales[product_id]['total_quantity'] += quantity
                                product_sales[product_id]['household_quantity'] += quantity
                                product_sales[product_id]['total_revenue'] += getattr(record, 'total_spent', 0)
                                product_sales[product_id]['purchase_count'] += 1
                except Exception as e:
                    logger.debug(f"处理家庭购买记录异常: {e}")
                    continue
            
            # 5.5 添加固有市场消耗统计
            try:
                all_transactions = await self.economic_center.query_all_tx.remote()
                inherent_market_count = 0
                for tx in all_transactions:
                    if tx.month == month and tx.type == 'inherent_market':
                        inherent_market_count += 1
                        for asset in tx.assets:
                            if hasattr(asset, 'product_id') and asset.product_id:
                                product_id = asset.product_id
                                quantity = getattr(asset, 'amount', 0)
                                
                                if product_id not in product_sales:
                                    product_sales[product_id] = {
                                        'name': getattr(asset, 'name', 'Unknown'),
                                        'total_quantity': 0,
                                        'total_revenue': 0,
                                        'purchase_count': 0,
                                        'household_quantity': 0,
                                        'inherent_market_quantity': 0
                                    }
                                
                                product_sales[product_id]['total_quantity'] += quantity
                                product_sales[product_id]['inherent_market_quantity'] += quantity
                                product_sales[product_id]['total_revenue'] += tx.amount
                
                logger.info(f"📊 月份 {month} 固有市场统计: {inherent_market_count} 笔交易")
            except Exception as e:
                logger.warning(f"固有市场销售统计失败: {e}")
            
            logger.info(f"📊 月份 {month} 销售统计: 检查了 {total_records_checked} 条家庭记录, 匹配了 {total_records_matched} 条, 得到 {len(product_sales)} 种商品销售数据")
            
            # 5.6 计算企业营业率
            firm_operation_rates = {}
            for firm in firms:
                try:
                    products = await self.economic_center.query_products.remote(firm.company_id)
                    if products:
                        total_products = len(products)
                        sold_products = sum(1 for p in products if p.product_id in product_sales)
                        operation_rate = sold_products / total_products if total_products > 0 else 0
                        firm_operation_rates[firm.company_id] = {
                            'total_products': total_products,
                            'sold_products': sold_products,
                            'operation_rate': operation_rate
                        }
                except Exception:
                    continue
            
            # 5.7 供需数据
            supply_demand = {}
            for product_id in set(list(product_inventory.keys()) + list(product_sales.keys())):
                supply = product_inventory.get(product_id, {}).get('quantity', 0)
                demand = product_sales.get(product_id, {}).get('total_quantity', 0)
                supply_demand[product_id] = {
                    'name': product_inventory.get(product_id, {}).get('name', 
                           product_sales.get(product_id, {}).get('name', 'Unknown')),
                    'supply': supply,
                    'demand': demand,
                    'supply_demand_ratio': supply / demand if demand > 0 else float('inf')
                }
            
            # 5.8 保存商品统计数据
            self.monthly_firm_revenue[month] = firm_revenues
            self.monthly_product_sales[month] = product_sales
            self.monthly_product_inventory[month] = product_inventory
            self.monthly_product_prices[month] = product_prices
            self.monthly_firm_operation_rate[month] = firm_operation_rates
            self.monthly_supply_demand[month] = supply_demand
            
            print(f"✅ 商品统计完成: {len(product_sales)}种商品, {len(firm_operation_rates)}家企业营业率")

            economic_indicators["iteration"] = self.current_month

            return economic_indicators

        except Exception as e:
            logger.error(f"综合采集失败: {e}")
            return {}

    async def _print_simulation_status(self, current_month: int):
        """打印仿真状态"""
        try:
            logger.info(f"===== 仿真状态报告 (月份 {current_month}) =====")
            
            # 统计家庭财富 - 并行处理
            print(f"💰 开始并行统计 {len(self.households)} 个家庭的财富...")
            
            # 并行查询所有家庭余额
            balance_tasks = [
                self.economic_center.query_balance.remote(household.household_id) 
                for household in self.households
            ]
            balances = await asyncio.gather(*balance_tasks, return_exceptions=True)
            
            # 统计有效的财富数据
            total_wealth = 0
            wealth_count = 0
            for i, balance in enumerate(balances):
                if not isinstance(balance, Exception) and balance is not None:
                    total_wealth += balance
                    wealth_count += 1
                else:
                    household = self.households[i] if i < len(self.households) else None
                    household_id = household.household_id if household else f"household_{i}"
                    logger.debug(f"获取家庭 {household_id} 财富失败: {balance}")
            
            avg_wealth = total_wealth / wealth_count if wealth_count > 0 else 0
            print(f"✅ 财富统计完成: {wealth_count}/{len(self.households)} 个家庭")
            logger.info(f"统计 {wealth_count} 个家庭: 总财富={total_wealth:.2f}, 平均财富={avg_wealth:.2f}")
            
            # 统计系统状态
            logger.info(f"系统状态: 活跃家庭={len(self.households)}, 活跃企业={len(self.firms)}")
            
        except Exception as e:
            logger.warning(f"状态报告生成失败: {e}")
    
    async def _print_monthly_summary(self, current_month: int) -> Dict[str, Any]:
        """打印月度统计摘要（直接使用已收集的指标数据）"""
        try:
            print(f"\n{'='*80}")
            print(f"📊 第 {current_month} 月度报告")
            print(f"{'='*80}")

            # 从已收集的经济指标中获取数据（避免重复计算）
            monthly_summary = self.economic_metrics_history[current_month - 1]

            # 检查是否收集到了有效的数据
            if not monthly_summary or 'employment_statistics' not in monthly_summary:
                logger.warning(f"第 {current_month} 月的经济指标数据不完整，无法生成月度报告")
                print(f"⚠️  第 {current_month} 月的经济指标数据不完整，无法生成月度报告")
                return {}

            # ==================== 1. 家庭部分 ====================
            print(f"\n{'─'*80}")
            print(f"🏠 家庭部分")
            print(f"{'─'*80}")

            # 从已收集的指标中获取数据
            employment_stats = monthly_summary['employment_statistics']
            income_stats = monthly_summary['income_expenditure_analysis']
            wealth_stats = monthly_summary['wealth_distribution']
            health_stats = monthly_summary['household_financial_health']
            
            # 打印家庭统计
            print(f"  家庭数量: {employment_stats['total_households']}")
            print(f"  总劳动力人数: {employment_stats['total_labor_force_available']}")
            print(f"  已就业劳动力: {employment_stats['total_labor_force_employed']} 人")
            print(f"  劳动力利用率: {employment_stats['labor_utilization_rate']:.1%}")
            print(f"  总岗位: {employment_stats['total_job_positions']} 个")
            print(f"  岗位占用率: {employment_stats['job_fill_rate']:.1%}")
            print(f"  就业家庭数: {employment_stats['employed_households']} ({employment_stats['household_employment_rate']*100:.1f}%)")
            print(f"  就业劳动力数: {employment_stats['total_labor_force_employed']} ({employment_stats['labor_utilization_rate']*100:.1f}%)")
            print(f"  平均薪资: ${income_stats['average_monthly_income_per_lh']:.2f} (总收入${income_stats['total_monthly_income']:.2f} / 劳动力{employment_stats['total_labor_force_available']})")
            print(f"  家庭平均收入: ${income_stats['average_monthly_income']:.2f} (总收入${income_stats['total_monthly_income']:.2f} / 家庭数{employment_stats['total_households']})")
            print(f"  其中再分配: ${income_stats['monthly_redistribution_amount']:.2f}")
            print(f"  当月平均支出: ${income_stats['average_monthly_expenditure']:.2f}")
            print(f"  当月储蓄率: {income_stats['monthly_savings_rate']:.2%}")
            print(f"  家庭平均财富: ${wealth_stats['average_wealth']:.2f}")
            print(f"  家庭财富中位数: ${wealth_stats['median_wealth']:.2f}")
            print(f"  财富基尼系数: {wealth_stats['gini_coefficient']:.4f}")
            print(f"  总消费: ${income_stats['total_monthly_expenditure']:.2f}")
            print(f"  财务健康: 盈余{health_stats['healthy_households']}家 | 赤字{health_stats['deficit_households']}家 | 平衡{health_stats['balanced_households']}家")
            
            # ==================== 2. 企业部分 ====================
            print(f"\n{'─'*80}")
            print(f"💼 企业部分 (共{len(self.firms)}家)")
            print(f"{'─'*80}")
            
            # 从已收集的企业月度指标中获取数据（避免重复查询）
            firm_metrics_this_month = [m for m in self.firm_monthly_metrics if m.month == current_month]
            
            # 汇总统计
            total_revenue = sum(m.monthly_revenue for m in firm_metrics_this_month)
            total_expenses = sum(m.monthly_expenses for m in firm_metrics_this_month)
            total_profit = sum(m.monthly_profit for m in firm_metrics_this_month)
            total_employees = sum(m.current_employees for m in firm_metrics_this_month)
            
            print(f"  企业总数: {len(self.firms)}")
            print(f"  总收入: ${total_revenue:,.2f}")
            print(f"  总支出: ${total_expenses:,.2f}")
            print(f"  总利润: ${total_profit:,.2f}")
            print(f"  总雇佣人数: {total_employees} 人")
            print(f"  平均每家企业收入: ${total_revenue/len(self.firms):.2f}")
            print(f"  平均每家企业利润: ${total_profit/len(self.firms):.2f}")
            
            # 打印所有企业的详细信息（使用已收集的指标数据 + 必要的补充查询）
            print(f"\n  📋 企业详情:")
            for i, metric in enumerate(firm_metrics_this_month, 1):
                try:
                    # 获取对应的firm对象
                    firm = next((f for f in self.firms if f.company_id == metric.company_id), None)
                    if not firm:
                        continue
                    
                    # 获取企业名称（行业）
                    firm_name = getattr(firm, 'main_business', 'Unknown')
                    
                    # 获取商品信息（库存、价格）
                    total_inventory = 0
                    total_inventory_value = 0.0
                    try:
                        products = await self.economic_center.query_products.remote(firm.company_id)
                        if products and isinstance(products, list):
                            for product in products:
                                if hasattr(product, 'amount'):
                                    total_inventory += product.amount
                                    if hasattr(product, 'price'):
                                        total_inventory_value += product.amount * product.price
                    except Exception:
                        pass
                    
                    # 获取生产数据
                    base_production = 0.0
                    labor_production = 0.0
                    try:
                        production_stats = await self.economic_center.query_firm_production_stats.remote(firm.company_id, current_month)
                        if isinstance(production_stats, dict):
                            base_production = production_stats.get('base_production', 0.0)
                            labor_production = production_stats.get('labor_production', 0.0)
                    except Exception:
                        pass
                    
                    # 计算营业率（有销售的商品数 / 总商品数，不考虑固有市场）
                    operation_rate = 0.0
                    if current_month in self.monthly_product_sales:
                        try:
                            products = await self.economic_center.query_products.remote(firm.company_id)
                            if products and isinstance(products, list):
                                product_sales = self.monthly_product_sales[current_month]
                                sold_products = 0
                                for product in products:
                                    if hasattr(product, 'product_id') and product.product_id in product_sales:
                                        # 只计算家庭购买量，不计算固有市场
                                        household_quantity = product_sales[product.product_id].get('household_quantity', 0)
                                        if household_quantity > 0:
                                            sold_products += 1
                                operation_rate = (sold_products / len(products) * 100) if len(products) > 0 else 0
                        except Exception:
                            pass
                    
                    # 打印企业信息
                    print(f"\n  [{i}] 企业编码: {metric.company_id[:12]}...")
                    print(f"      所属行业: {firm_name}")
                    print(f"      当月总收入: ${metric.monthly_revenue:.2f}")
                    print(f"      当月总支出: ${metric.monthly_expenses:.2f}")
                    print(f"      利润: ${metric.monthly_profit:.2f}")
                    print(f"      商品总库存: {total_inventory:.0f} 件")
                    print(f"      商品库存总价值: ${total_inventory_value:.2f}")
                    print(f"      当月生产数量: {base_production:.1f}")
                    print(f"      劳动力生产数量: {labor_production:.1f}")
                    print(f"      雇佣员工数: {metric.current_employees} 人")
                    print(f"      营业率（仅家庭购买）: {operation_rate:.1f}%")
                    print(f"      招聘: {metric.successful_hires}/{metric.job_postings} (成功率{metric.recruitment_success_rate:.1%})")
                    
                except Exception as e:
                    logger.warning(f"打印企业 {metric.company_id} 详情失败: {e}")
                    print(f"\n  [{i}] 企业编码: {metric.company_id[:12]}...")
                    print(f"      数据获取失败")
            
            print(f"\n{'='*80}")
            
        except Exception as e:
            logger.warning(f"月度统计生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def _final_settlement(self):
        """最终结算"""
        logger.info("开始最终结算...")
        
        try:
            # 并行计算所有家庭最终财富
            print(f"💰 开始并行处理 {len(self.households)} 个家庭的最终结算...")
            
            async def settle_household(household):
                try:
                    # 并行获取结算数据和最终财富
                    settlement_task = self.economic_center.compute_household_settlement.remote(household.household_id)
                    wealth_task = household.get_balance_ref()
                    
                    results = await asyncio.gather(settlement_task, wealth_task, return_exceptions=True)
                    
                    if not isinstance(results[0], Exception):
                        total_income, total_spent = results[0]
                    else:
                        total_income, total_spent = 0, 0
                        
                    final_wealth = results[1] if not isinstance(results[1], Exception) else 0
                    
                    return {
                        'household_id': household.household_id,
                        'total_income': total_income,
                        'total_spent': total_spent,
                        'final_wealth': final_wealth
                    }
                    
                except Exception as e:
                    logger.warning(f"家庭 {household.household_id} 结算失败: {e}")
                    return None
            
            # 并行处理所有家庭的结算
            settlement_tasks = [settle_household(h) for h in self.households]
            settlement_results = await asyncio.gather(*settlement_tasks, return_exceptions=True)
            
            # 输出结算结果
            successful_settlements = 0
            for result in settlement_results:
                if result and not isinstance(result, Exception):
                    logger.info(f"家庭 {result['household_id']}: 总收入={result['total_income']:.2f}, "
                               f"总支出={result['total_spent']:.2f}, 最终财富={result['final_wealth']:.2f}")
                    successful_settlements += 1
            
            print(f"✅ 最终结算完成: {successful_settlements}/{len(self.households)} 个家庭结算成功")
            logger.info("最终结算完成")
            
        except Exception as e:
            logger.error(f"最终结算失败: {e}")
    
    
    
    
    async def _generate_joint_debug_metrics(self) -> Dict[str, Any]:
        """生成联调指标报告"""
        logger.info("生成联调指标报告...")
        
        try:
            # 1. 家庭智能体数据收集指标
            household_metrics = {
                "monthly_tracking": {
                    "total_records": len(self.household_monthly_metrics[1]) * len(self.household_monthly_metrics),
                    "months_covered": len(self.household_monthly_metrics),
                    "households_tracked": len(self.household_monthly_metrics[1])
                },
                "final_summary": await self._generate_household_final_summary()
            }
            
            # 2. 企业智能体数据收集指标
            firm_metrics = {
                "monthly_tracking": {
                    "total_records": len(self.firm_monthly_metrics),
                    "months_covered": len(set(m.month for m in self.firm_monthly_metrics)),
                    "firms_tracked": len(set(m.company_id for m in self.firm_monthly_metrics))
                },
                "final_summary": self._generate_firm_final_summary()
            }
            
            # 3. 系统性能监控指标
            performance_metrics = self._generate_performance_summary()
            
            # 4. LLM调用性能指标
            # llm_metrics = self._generate_llm_summary()
            
            joint_debug_report = {
                "household_metrics": household_metrics,
                "firm_metrics": firm_metrics,
                "performance_metrics": performance_metrics,
                # "llm_metrics": llm_metrics,
                "data_quality": {
                    "household_data_completeness": len(self.household_monthly_metrics) / (len(self.households) * self.config.num_iterations) if self.households else 0,
                    "firm_data_completeness": len(self.firm_monthly_metrics) / (len(self.firms) * self.config.num_iterations) if self.firms else 0,
                    "performance_data_points": len(self.performance_metrics),
                    "llm_data_points": len(self.llm_metrics)
                }
            }
            
            return joint_debug_report
            
        except Exception as e:
            logger.error(f"生成联调指标报告失败: {e}")
            return {"error": f"生成联调指标报告失败: {e}"}
    
    async def _generate_household_final_summary(self) -> Dict[str, Any]:
        """生成家庭最终统计摘要"""
        try:
            # 计算所有家庭的总消费、总收入
            total_consumption = 0
            total_income = 0
            
            # # 按消费类别统计 - 动态收集所有消费类别
            # category_totals = {}
            
            # # 储蓄曲线数据
            # savings_curves = {}
            
            # # 并行收集所有家庭的最终统计数据
            # print(f"📊 开始并行收集 {len(self.households)} 个家庭的最终统计数据...")
            
            # async def collect_household_final_data(household):
            #     try:
            #         # 获取家庭总收入和支出 (支出已经通过/0.65还原为税前金额)
            #         income, spent = await self.economic_center.compute_household_settlement.remote(household.household_id)
                    
            #         # 收集该家庭的储蓄曲线
            #         household_savings = []
            #         household_categories = {}
                    
            #         for metric in self.household_monthly_metrics:
            #             if metric.household_id == household.household_id:
            #                 savings_amount = metric.monthly_income - metric.monthly_expenditure
            #                 household_savings.append({
            #                     "month": metric.month,
            #                     "savings": savings_amount,
            #                     "savings_rate": metric.savings_rate
            #                 })
                            
            #                 # 累计消费类别 - 动态添加所有类别
            #                 for category, amount in metric.consumption_structure.items():
            #                     if category not in household_categories:
            #                         household_categories[category] = 0
            #                     household_categories[category] += amount
                    
            #         return {
            #             'household_id': household.household_id,
            #             'income': income,
            #             'spent': spent,
            #             'savings_curve': household_savings if household_savings else None,
            #             'categories': household_categories
            #         }
                    
            #     except Exception as e:
            #         logger.warning(f"处理家庭 {household.household_id} 最终数据失败: {e}")
            #         return None
            
            # # 并行收集所有家庭数据
            # final_data_tasks = [collect_household_final_data(h) for h in self.households]
            # household_final_data = await asyncio.gather(*final_data_tasks, return_exceptions=True)
            
            # # 汇总年度统计数据
            # for data in household_final_data:
            #     if data and not isinstance(data, Exception):
            #         total_income += data['income']
            #         total_consumption += data['spent']
                    
            #         if data['savings_curve']:
            #             savings_curves[data['household_id']] = data['savings_curve']
                    
            #         # 累计消费类别 - 动态添加所有类别
            #         for category, amount in data['categories'].items():
            #             if category not in category_totals:
            #                 category_totals[category] = 0
            #             category_totals[category] += amount
            
            # print(f"✅ 最终统计数据收集完成: {len([d for d in household_final_data if d and not isinstance(d, Exception)])}/{len(self.households)} 个家庭")
            
            # 计算各类消费占比
            category_ratios = {}
            # if total_consumption > 0:
            #     for category, amount in category_totals.items():
            #         category_ratios[category] = amount / total_consumption
            category_expenditure = {}
            for i in self.economic_metrics_history:
                total_consumption += i["income_expenditure_analysis"]["total_monthly_expenditure"]
                total_income += i["income_expenditure_analysis"]["total_monthly_income"]
                category_expenditure = i["income_expenditure_analysis"]["monthly_consumption_structure"]
                for category, amount in category_expenditure.items():
                    if category not in category_expenditure:
                        category_expenditure[category] = 0
                    category_expenditure[category] += amount
            
            for category, total in category_expenditure.items():
                category_ratios[category] = total / total_consumption

            print(f"✅ 最终统计数据收集完成: {len(self.households)} 个家庭")

            return {
                "total_consumption": total_consumption,
                "total_income": total_income,
                "category_consumption_ratios": category_ratios,
                # "savings_curves_count": len(savings_curves),
                "average_savings_rate": (total_income - total_consumption) / total_income
            }
            
        except Exception as e:
            logger.error(f"生成家庭最终摘要失败: {e}")
            return {"error": f"生成家庭最终摘要失败: {e}"}
    
    def _generate_firm_final_summary(self) -> Dict[str, Any]:
        """生成企业最终统计摘要"""
        try:
            if not self.firm_monthly_metrics:
                return {"total_revenue": 0, "average_employees": 0, "overall_recruitment_rate": 0}
            
            # 按企业汇总数据
            firm_totals = {}
            for metric in self.firm_monthly_metrics:
                if metric.company_id not in firm_totals:
                    firm_totals[metric.company_id] = {
                        "total_revenue": 0,
                        "employee_months": [],
                        "total_job_postings": 0,
                        "total_successful_hires": 0
                    }
                
                firm_totals[metric.company_id]["total_revenue"] += metric.monthly_revenue
                firm_totals[metric.company_id]["employee_months"].append(metric.current_employees)
                firm_totals[metric.company_id]["total_job_postings"] += metric.job_postings
                firm_totals[metric.company_id]["total_successful_hires"] += metric.successful_hires
            
            # 计算汇总指标
            total_revenue = sum(data["total_revenue"] for data in firm_totals.values())
            average_employees = 0
            overall_recruitment_rate = 0
            
            if firm_totals:
                total_employee_months = sum(
                    sum(data["employee_months"]) for data in firm_totals.values()
                )
                total_months = sum(len(data["employee_months"]) for data in firm_totals.values())
                average_employees = total_employee_months / total_months if total_months > 0 else 0
                
                total_job_postings = sum(data["total_job_postings"] for data in firm_totals.values())
                total_successful_hires = sum(data["total_successful_hires"] for data in firm_totals.values())
                
                # 修复招聘成功率计算逻辑
                # 方法1：总成功率（总成功招聘数 / 总岗位发布数）
                if total_job_postings > 0:
                    overall_recruitment_rate = total_successful_hires / total_job_postings
                    # 确保成功率不超过100%
                    overall_recruitment_rate = min(overall_recruitment_rate, 1.0)
                else:
                    overall_recruitment_rate = 0
                
                # 方法2：平均成功率（各月成功率的平均值）
                monthly_success_rates = []
                for data in firm_totals.values():
                    if data["total_job_postings"] > 0:
                        monthly_rate = data["total_successful_hires"] / data["total_job_postings"]
                        monthly_success_rates.append(min(monthly_rate, 1.0))  # 确保单月成功率不超过100%
                
                if monthly_success_rates:
                    average_monthly_rate = sum(monthly_success_rates) / len(monthly_success_rates)
                    # 使用平均成功率作为备选指标
                    if overall_recruitment_rate > 1.0:
                        overall_recruitment_rate = average_monthly_rate
            
            return {
                "total_revenue": total_revenue,
                "average_employees": average_employees,
                "overall_recruitment_rate": overall_recruitment_rate,
                "firms_tracked": len(firm_totals),
                # 添加详细的招聘统计信息，帮助调试
                "recruitment_debug_info": {
                    "total_job_postings": total_job_postings,
                    "total_successful_hires": total_successful_hires,
                    "monthly_success_rates": monthly_success_rates if 'monthly_success_rates' in locals() else [],
                    "calculation_method": "total_rate" if total_job_postings > 0 and total_successful_hires <= total_job_postings else "average_monthly_rate"
                }
            }
            
        except Exception as e:
            logger.error(f"生成企业最终摘要失败: {e}")
            return {"error": f"生成企业最终摘要失败: {e}"}
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """生成性能监控摘要"""
        try:
            if not self.performance_metrics:
                return {"total_operations": 0, "average_duration": 0, "operations_by_type": {}}
            
            # 按操作类型分组
            operations_by_type = {}
            for metric in self.performance_metrics:
                if metric.operation_type not in operations_by_type:
                    operations_by_type[metric.operation_type] = []
                operations_by_type[metric.operation_type].append(metric.duration)
            
            # 计算每种操作的统计数据
            operation_stats = {}
            total_duration = 0
            total_operations = len(self.performance_metrics)
            
            for op_type, durations in operations_by_type.items():
                operation_stats[op_type] = {
                    "count": len(durations),
                    "total_duration": sum(durations),
                    "average_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations)
                }
                total_duration += sum(durations)
            
            return {
                "total_operations": total_operations,
                "total_duration": total_duration,
                "average_duration": total_duration / total_operations if total_operations > 0 else 0,
                "operations_by_type": operation_stats
            }
            
        except Exception as e:
            logger.error(f"生成性能监控摘要失败: {e}")
            return {"error": f"生成性能监控摘要失败: {e}"}
    
    def _generate_llm_summary(self) -> Dict[str, Any]:
        """生成LLM调用摘要"""
        try:
            if not self.llm_metrics:
                return {"total_calls": 0, "total_tokens": 0, "success_rate": 0, "average_duration": 0}
            
            total_calls = len(self.llm_metrics)
            successful_calls = sum(1 for m in self.llm_metrics if m.success)
            total_input_tokens = sum(m.input_tokens for m in self.llm_metrics)
            total_output_tokens = sum(m.output_tokens for m in self.llm_metrics)
            total_duration = sum(m.api_call_duration for m in self.llm_metrics)
            
            # 按智能体类型分组
            by_agent_type = {}
            for metric in self.llm_metrics:
                if metric.agent_type not in by_agent_type:
                    by_agent_type[metric.agent_type] = {
                        "calls": 0, "input_tokens": 0, "output_tokens": 0, "duration": 0
                    }
                by_agent_type[metric.agent_type]["calls"] += 1
                by_agent_type[metric.agent_type]["input_tokens"] += metric.input_tokens
                by_agent_type[metric.agent_type]["output_tokens"] += metric.output_tokens
                by_agent_type[metric.agent_type]["duration"] += metric.api_call_duration
            
            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "average_duration": total_duration / total_calls if total_calls > 0 else 0,
                "by_agent_type": by_agent_type
            }
            
        except Exception as e:
            logger.error(f"生成LLM调用摘要失败: {e}")
            return {"error": f"生成LLM调用摘要失败: {e}"}
    
    async def _generate_price_trend_chart(self):
        """生成商品平均价格趋势图"""
        try:
            if not self.economic_metrics_history:
                logger.warning("没有经济指标历史数据，跳过价格趋势图生成")
                return
            
            # 提取价格数据
            months = []
            avg_prices = []
            
            for metrics in self.economic_metrics_history:
                if not isinstance(metrics, dict):
                    continue
                
                month = metrics.get("iteration", 0)
                income_expenditure = metrics.get("income_expenditure_analysis", {})
                
                if month > 0 and income_expenditure:
                    avg_price = income_expenditure.get("avg_price", 0.0)
                    if avg_price > 0:  # 只处理有效的价格数据
                        months.append(month)
                        avg_prices.append(avg_price)
            
            if not months:
                logger.warning("没有有效的价格数据，跳过价格趋势图生成")
                return
            
            # 创建图表
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建单个图表
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle('Average Product Price Trend', fontsize=16, fontweight='bold')
            
            # 绘制平均价格趋势线
            ax.plot(months, avg_prices, 'b-o', linewidth=3, markersize=8, 
                   markerfacecolor='lightblue', markeredgecolor='darkblue', 
                   markeredgewidth=2, label='Average Price')
            
            # 设置图表属性
            ax.set_title('Average Product Price Trend', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Average Price ($)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=12)
            
            # 添加数值标签（每隔几个点显示一个，避免拥挤）
            label_interval = max(1, len(months) // 8)  # 最多显示8个标签
            for i, (month, price) in enumerate(zip(months, avg_prices)):
                if i % label_interval == 0 or i == len(months) - 1:  # 显示第一个、最后一个和中间的几个
                    ax.annotate(f'${price:.2f}', (month, price), 
                               textcoords="offset points", xytext=(0,15), ha='center',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 设置坐标轴
            ax.set_xlim(min(months) - 0.5, max(months) + 0.5)
            if len(avg_prices) > 1:
                price_range = max(avg_prices) - min(avg_prices)
                ax.set_ylim(min(avg_prices) - price_range * 0.1, max(avg_prices) + price_range * 0.1)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_dir = os.path.join(self.experiment_output_dir, "charts")
            os.makedirs(output_dir, exist_ok=True)
            
            chart_path = os.path.join(output_dir, "average_product_price_trend.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成价格统计摘要
            if len(avg_prices) > 1:
                price_change = ((avg_prices[-1] - avg_prices[0]) / avg_prices[0]) * 100
                max_price = max(avg_prices)
                min_price = min(avg_prices)
                
                print(f"\n📊 商品平均价格趋势分析:")
                print(f"   数据月份范围: 第{min(months)}月 - 第{max(months)}月")
                print(f"   初始平均价格: ${avg_prices[0]:.2f}")
                print(f"   最终平均价格: ${avg_prices[-1]:.2f}")
                print(f"   价格变化率: {price_change:+.2f}%")
                print(f"   期间最高价格: ${max_price:.2f}")
                print(f"   期间最低价格: ${min_price:.2f}")
                print(f"   价格波动幅度: ${max_price - min_price:.2f}")
                print(f"   图表已保存: {chart_path}")
            else:
                print(f"\n📊 商品平均价格: ${avg_prices[0]:.2f}")
                print(f"   图表已保存: {chart_path}")
            
            logger.info(f"价格趋势图生成完成: {chart_path}")
            
        except Exception as e:
            logger.error(f"生成价格趋势图失败: {e}")
            print(f"❌ 价格趋势图生成失败: {e}")
    
    async def _generate_gini_and_wage_charts(self):
        """生成基尼系数和平均工资折线图"""
        try:
            if not self.gini_history or not self.wage_history:
                logger.warning("没有基尼系数或工资历史数据，跳过图表生成")
                return
            
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 创建双子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('Gini Coefficient and Average Wage Trends', fontsize=16, fontweight='bold')
            
            # 提取数据
            months = [item["month"] for item in self.gini_history]
            gini_pre = [item["gini_pre"] for item in self.gini_history]
            gini_post = [item["gini_post"] for item in self.gini_history]
            improvements = [item["improvement"] for item in self.gini_history]
            
            wage_months = [item["month"] for item in self.wage_history]
            avg_wages = [item["average_wage"] for item in self.wage_history]
            
            # 绘制基尼系数图
            ax1.plot(months, gini_pre, 'r-o', linewidth=2, markersize=6, 
                    label='Pre-redistribution', markerfacecolor='lightcoral', markeredgecolor='darkred')
            ax1.plot(months, gini_post, 'b-s', linewidth=2, markersize=6, 
                    label='Post-redistribution', markerfacecolor='lightblue', markeredgecolor='darkblue')
            ax1.fill_between(months, gini_pre, gini_post, alpha=0.3, color='green', 
                           label='Inequality Reduction')
            
            ax1.set_title('Gini Coefficient Trends (Income Inequality)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Gini Coefficient', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(fontsize=10)
            ax1.set_ylim(0, 1)  # 基尼系数范围[0,1]
            
            # 添加改善程度标注
            for i, (month, improvement) in enumerate(zip(months, improvements)):
                if i % max(1, len(months) // 6) == 0 or i == len(months) - 1:  # 显示部分标签
                    ax1.annotate(f'{improvement:.3f}', (month, gini_pre[i]), 
                               textcoords="offset points", xytext=(0,10), ha='center',
                               fontsize=8, alpha=0.7)
            
            # 绘制平均工资图
            ax2.plot(wage_months, avg_wages, 'g-^', linewidth=3, markersize=8, 
                    markerfacecolor='lightgreen', markeredgecolor='darkgreen', 
                    label='Average Wage per Worker')
            
            ax2.set_title('Average Wage Trends', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Month', fontsize=12)
            ax2.set_ylabel('Average Wage ($)', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=10)
            
            # 添加工资数值标签
            label_interval = max(1, len(wage_months) // 8)
            for i, (month, wage) in enumerate(zip(wage_months, avg_wages)):
                if i % label_interval == 0 or i == len(wage_months) - 1:
                    ax2.annotate(f'${wage:.0f}', (month, wage), 
                               textcoords="offset points", xytext=(0,15), ha='center',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 设置坐标轴
            if months:
                ax1.set_xlim(min(months) - 0.5, max(months) + 0.5)
                ax2.set_xlim(min(wage_months) - 0.5, max(wage_months) + 0.5)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_dir = os.path.join(self.experiment_output_dir, "charts")
            os.makedirs(output_dir, exist_ok=True)
            
            chart_path = os.path.join(output_dir, "gini_coefficient_and_wage_trends.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印统计信息
            if self.gini_history:
                initial_gini_pre = self.gini_history[0]["gini_pre"]
                final_gini_pre = self.gini_history[-1]["gini_pre"]
                initial_gini_post = self.gini_history[0]["gini_post"]
                final_gini_post = self.gini_history[-1]["gini_post"]
                avg_improvement = sum(item["improvement"] for item in self.gini_history) / len(self.gini_history)
                
                print(f"\n📊 基尼系数趋势分析:")
                print(f"   初始基尼系数 (再分配前): {initial_gini_pre:.3f}")
                print(f"   最终基尼系数 (再分配前): {final_gini_pre:.3f}")
                print(f"   初始基尼系数 (再分配后): {initial_gini_post:.3f}")
                print(f"   最终基尼系数 (再分配后): {final_gini_post:.3f}")
                print(f"   平均改善幅度: {avg_improvement:.3f}")
            
            if self.wage_history:
                initial_wage = self.wage_history[0]["average_wage"]
                final_wage = self.wage_history[-1]["average_wage"]
                wage_growth = ((final_wage - initial_wage) / initial_wage * 100) if initial_wage > 0 else 0
                
                print(f"\n💰 平均工资趋势分析:")
                print(f"   初始平均工资: ${initial_wage:.2f}")
                print(f"   最终平均工资: ${final_wage:.2f}")
                print(f"   工资增长率: {wage_growth:+.1f}%")
                print(f"   图表已保存: {chart_path}")
            
            logger.info(f"基尼系数和工资趋势图生成完成: {chart_path}")
            
        except Exception as e:
            logger.error(f"生成基尼系数和工资趋势图失败: {e}")
            print(f"❌ 基尼系数和工资趋势图生成失败: {e}")
    
    async def generate_simulation_report(self) -> Dict[str, Any]:
        """生成仿真报告"""
        logger.info("生成详细仿真报告...")
        
        # 收集经济指标
        # economic_indicators = await self._collect_economic_indicators()
        economic_indicators = self.economic_metrics_history
        # 生成联调指标报告
        joint_debug_metrics = await self._generate_joint_debug_metrics()
        
        # 生成价格趋势图
        await self._generate_price_trend_chart()
        
        # 生成基尼系数和平均工资趋势图
        await self._generate_gini_and_wage_charts()
        
        # 生成新增的月度统计可视化
        await self._generate_monthly_statistics_charts()
        
        report = {
            "simulation_summary": {
                "total_iterations": self.config.num_iterations,
                "total_households": len(self.households),
                "total_firms": len(self.firms),
                "simulation_duration": "完成",
                "config": {
                    "max_concurrent_tasks": self.config.max_concurrent_tasks,
                    "monitor_interval": self.config.monitor_interval,
                    "monitoring_enabled": self.config.enable_monitoring
                }
            },
            "economic_indicators": economic_indicators,
            "economic_trends": self._analyze_economic_trends(),
            "system_metrics": {
                "total_metrics_collected": len(self.metrics_history),
                "avg_cpu_usage": sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0,
                "avg_memory_usage": sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0,
                "peak_memory_usage": max(m.memory_used_gb for m in self.metrics_history) if self.metrics_history else 0
            },
            "joint_debug_metrics": joint_debug_metrics,  # 新增联调指标
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
        }
        
        return report
    
    def _analyze_economic_trends(self) -> Dict[str, Any]:
        """分析经济趋势"""
        if len(self.economic_metrics_history) < 2:
            return {"trend_analysis": "数据不足，无法分析趋势"}
        
        try:
            trends = {
                "unemployment_trend": [],
                "income_trend": [],
                "expenditure_trend": [],
                "wealth_trend": [],
                "savings_rate_trend": [],
                "labor_utilization_trend": [],
                "monthly_income_trend": [],
                "monthly_expenditure_trend": [],
                "monthly_savings_rate_trend": []
            }
            
            for metrics in self.economic_metrics_history:
                if not isinstance(metrics, dict):
                    continue
                    
                iteration = metrics.get("iteration", 0)
                
                # 失业率趋势（基于家庭）
                household_unemployment_rate = metrics.get("employment_statistics", {}).get("household_unemployment_rate", 0)
                trends["unemployment_trend"].append({"iteration": iteration, "value": household_unemployment_rate})
                
                # 劳动力利用率趋势（新增）
                labor_utilization_rate = metrics.get("employment_statistics", {}).get("labor_utilization_rate", 0)
                trends["labor_utilization_trend"].append({"iteration": iteration, "value": labor_utilization_rate})
                
                # 收入趋势
                avg_income = metrics.get("income_expenditure_analysis", {}).get("average_income", 0)
                trends["income_trend"].append({"iteration": iteration, "value": avg_income})
                
                # 支出趋势
                avg_expenditure = metrics.get("income_expenditure_analysis", {}).get("average_expenditure", 0)
                trends["expenditure_trend"].append({"iteration": iteration, "value": avg_expenditure})
                
                # 财富趋势
                avg_wealth = metrics.get("wealth_distribution", {}).get("average_wealth", 0)
                trends["wealth_trend"].append({"iteration": iteration, "value": avg_wealth})
                
                # 储蓄率趋势
                savings_rate = metrics.get("income_expenditure_analysis", {}).get("savings_rate", 0)
                trends["savings_rate_trend"].append({"iteration": iteration, "value": savings_rate})
                
                # 月度收入趋势（新增）
                monthly_income = metrics.get("income_expenditure_analysis", {}).get("average_monthly_income", 0)
                trends["monthly_income_trend"].append({"iteration": iteration, "value": monthly_income})
                
                # 月度支出趋势（新增）
                monthly_expenditure = metrics.get("income_expenditure_analysis", {}).get("average_monthly_expenditure", 0)
                trends["monthly_expenditure_trend"].append({"iteration": iteration, "value": monthly_expenditure})
                
                # 月度储蓄率趋势（新增）
                monthly_savings_rate = metrics.get("income_expenditure_analysis", {}).get("monthly_savings_rate", 0)
                trends["monthly_savings_rate_trend"].append({"iteration": iteration, "value": monthly_savings_rate})
            
            # 计算趋势方向
            trend_summary = {}
            for trend_name, trend_data in trends.items():
                if isinstance(trend_data, list) and len(trend_data) >= 2:
                    first_data = trend_data[0]
                    last_data = trend_data[-1]
                    
                    if not isinstance(first_data, dict) or not isinstance(last_data, dict):
                        continue
                        
                    first_value = first_data.get("value", 0)
                    last_value = last_data.get("value", 0)
                    
                    if last_value > first_value:
                        direction = "上升"
                        change_rate = (last_value - first_value) / first_value if first_value != 0 else 0
                    elif last_value < first_value:
                        direction = "下降"
                        change_rate = (first_value - last_value) / first_value if first_value != 0 else 0
                    else:
                        direction = "稳定"
                        change_rate = 0
                    
                    trend_summary[trend_name] = {
                        "direction": direction,
                        "change_rate": change_rate,
                        "start_value": first_value,
                        "end_value": last_value
                    }
            
            return {
                "trends": trends,
                "trend_summary": trend_summary,
                "data_points": len(self.economic_metrics_history)
            }
            
        except Exception as e:
            logger.warning(f"分析经济趋势失败: {e}")
            return {"error": "趋势分析失败"}
    
    async def save_simulation_report(self, report: Dict[str, Any]):
        """保存仿真报告"""
        save_dir = f"{self.experiment_output_dir}/"

        try:
            beijing_time = datetime.now(pytz.timezone('Asia/Shanghai'))
            report_file = f"simulation_report_{beijing_time.strftime('%Y%m%d_%H%M%S')}.json"
            report_file = os.path.join(save_dir, report_file)
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"仿真报告已保存到: {report_file}")
            
            # 打印摘要
            self._print_simulation_summary(report)
            
        except Exception as e:
            logger.error(f"保存仿真报告失败: {e}")
    
    def _print_simulation_summary(self, report: Dict[str, Any]):
        """打印仿真摘要"""
        print("\n" + "="*80)
        print("🏛️  经济仿真完成报告")
        print("="*80)
        
        # 基本信息
        summary = report["simulation_summary"]
        print(f"📊 仿真基本信息:")
        print(f"   仿真轮数: {summary['total_iterations']}")
        print(f"   参与家庭: {summary['total_households']}")
        print(f"   参与企业: {summary['total_firms']}")
        print(f"   仿真状态: {summary['simulation_duration']}")
        
            # 经济指标 - 现在按月份保存
        if "economic_indicators" in report and report["economic_indicators"]:
            economic_indicators = report["economic_indicators"]
            
            # 检查是否是按月份保存的数据结构
            if isinstance(economic_indicators, list) and len(economic_indicators) > 0:
                print(f"\n📊 月度经济指标汇总:")
                print(f"   总月份数: {len(economic_indicators)}")
                
                # 显示最后一个月（最新）的详细指标
                latest_month_data = economic_indicators[-1]
                latest_month = latest_month_data.get("iteration", "未知")
                
                print(f"\n📅 最新月份 (第{latest_month}月) 详细指标:")
                
                # 就业统计
                if "employment_statistics" in latest_month_data:
                    emp_stats = latest_month_data["employment_statistics"]
                    print(f"\n💼 就业统计:")
                    print(f"   🏠 基于家庭的就业指标:")
                    print(f"      就业率: {emp_stats.get('household_employment_rate', 0):.1%}")
                    print(f"      失业率: {emp_stats.get('household_unemployment_rate', 0):.1%}")
                    print(f"      就业家庭: {emp_stats.get('employed_households', 0)}")
                    print(f"      失业家庭: {emp_stats.get('unemployed_households', 0)}")
                    
                    print(f"   👥 基于劳动力人数的就业指标:")
                    print(f"      劳动力利用率: {emp_stats.get('labor_utilization_rate', 0):.1%}")
                    print(f"      劳动力失业率: {emp_stats.get('labor_unemployment_rate', 0):.1%}")
                    print(f"      总可用劳动力人数: {emp_stats.get('total_labor_force_available', 0)}人")
                    print(f"      已就业劳动力人数: {emp_stats.get('total_labor_force_employed', 0)}人")
                    print(f"      未就业劳动力人数: {emp_stats.get('total_labor_force_unemployed', 0)}人")
                    print(f"      平均每家庭劳动力人数: {emp_stats.get('avg_labor_force_per_household', 0):.1f}人")
                    print(f"      平均每家庭就业人数: {emp_stats.get('avg_employed_people_per_household', 0):.1f}人")
                
                # 收入支出分析
                if "income_expenditure_analysis" in latest_month_data:
                    income_exp = latest_month_data["income_expenditure_analysis"]
                    print(f"\n💰 收入支出分析:")
                    
                    # 月度统计
                    if income_exp.get('average_monthly_income', 0) > 0:
                        print(f"   当月平均收入: ${income_exp.get('average_monthly_income', 0):.2f}")
                        print(f"   当月平均支出: ${income_exp.get('average_monthly_expenditure', 0):.2f}")
                        print(f"   当月支出收入比: {income_exp.get('monthly_expenditure_income_ratio', 0):.2f}")
                        print(f"   当月储蓄率: {income_exp.get('monthly_savings_rate', 0):.1%}")
                    
                    # 累积统计
                    # print(f"   累积平均收入: ${income_exp.get('average_income', 0):.2f}")
                    # print(f"   累积平均支出: ${income_exp.get('average_expenditure', 0):.2f}")
                    # print(f"   累积支出收入比: {income_exp.get('expenditure_income_ratio', 0):.2f}")
                    # print(f"   累积储蓄率: {income_exp.get('savings_rate', 0):.1%}")
                    
                    # 消费结构
                    if "monthly_consumption_structure" in income_exp:
                        consumption_structure = income_exp["monthly_consumption_structure"]
                        print(f"\n🛒 消费结构 (第{latest_month}月):")
                        total_consumption = sum(consumption_structure.values())
                        if total_consumption > 0:
                            for category, amount in consumption_structure.items():
                                percentage = (amount / total_consumption) * 100
                                print(f"   {category}: ${amount:.2f} ({percentage:.1f}%)")
                
                # 财富分布
                if "wealth_distribution" in latest_month_data:
                    wealth = latest_month_data["wealth_distribution"]
                    print(f"\n🏦 财富分布:")
                    print(f"   平均财富: ${wealth.get('average_wealth', 0):.2f}")
                    print(f"   财富中位数: ${wealth.get('median_wealth', 0):.2f}")
                    print(f"   基尼系数: {wealth.get('gini_coefficient', 0):.3f}")
                
                # 家庭财务健康
                if "household_financial_health" in latest_month_data:
                    health = latest_month_data["household_financial_health"]
                    print(f"\n🏥 家庭财务健康:")
                    print(f"   财务健康家庭: {health.get('healthy_households', 0)} ({health.get('healthy_rate', 0):.1%})")
                    print(f"   财务赤字家庭: {health.get('deficit_households', 0)} ({health.get('deficit_rate', 0):.1%})")
                    print(f"   收支平衡家庭: {health.get('balanced_households', 0)}")
                
                # 收入分布
                if "income_distribution" in latest_month_data:
                    income_dist = latest_month_data["income_distribution"]
                    print(f"\n📈 收入分布:")
                    for range_name, count in income_dist.items():
                        percentage = count / summary['total_households'] * 100 if summary['total_households'] > 0 else 0
                        print(f"   ${range_name}: {count} 家庭 ({percentage:.1f}%)")
                
                # 显示月度趋势摘要
                print(f"\n📈 月度趋势摘要:")
                if len(economic_indicators) > 1:
                    # 比较第一个月和最后一个月的关键指标
                    first_month = economic_indicators[0]
                    last_month = economic_indicators[-1]
                    
                    # 就业率变化
                    if "employment_statistics" in first_month and "employment_statistics" in last_month:
                        first_emp_rate = first_month["employment_statistics"].get("labor_utilization_rate", 0)
                        last_emp_rate = last_month["employment_statistics"].get("labor_utilization_rate", 0)
                        emp_change = last_emp_rate - first_emp_rate
                        emp_emoji = "📈" if emp_change > 0 else "📉" if emp_change < 0 else "➡️"
                        print(f"   {emp_emoji} 劳动力利用率: {first_emp_rate:.1%} → {last_emp_rate:.1%} ({emp_change:+.1%})")
                    
                    # 收入变化
                    if "income_expenditure_analysis" in first_month and "income_expenditure_analysis" in last_month:
                        first_income = first_month["income_expenditure_analysis"].get("average_monthly_income", 0)
                        last_income = last_month["income_expenditure_analysis"].get("average_monthly_income", 0)
                        if first_income > 0:
                            income_change = (last_income - first_income) / first_income
                            income_emoji = "📈" if income_change > 0 else "📉" if income_change < 0 else "➡️"
                            print(f"   {income_emoji} 月均收入: ${first_income:.2f} → ${last_income:.2f} ({income_change:+.1%})")
                    
                    # 储蓄率变化
                    if "income_expenditure_analysis" in first_month and "income_expenditure_analysis" in last_month:
                        first_savings = first_month["income_expenditure_analysis"].get("monthly_savings_rate", 0)
                        last_savings = last_month["income_expenditure_analysis"].get("monthly_savings_rate", 0)
                        savings_change = last_savings - first_savings
                        savings_emoji = "📈" if savings_change > 0 else "📉" if savings_change < 0 else "➡️"
                        print(f"   {savings_emoji} 月储蓄率: {first_savings:.1%} → {last_savings:.1%} ({savings_change:+.1%})")
                else:
                    print("   ℹ️  只有一个月的数据，无法显示趋势变化")
                
            else:
                # 兼容旧的数据结构
                print(f"\n⚠️  经济指标数据结构异常，无法解析")
                print(f"   数据类型: {type(economic_indicators)}")
                print(f"   数据内容: {economic_indicators}")
        
        # 系统性能指标
        print(f"\n⚙️  系统性能指标:")
        metrics = report["system_metrics"]
        print(f"   平均CPU使用率: {metrics['avg_cpu_usage']:.1f}%")
        print(f"   平均内存使用率: {metrics['avg_memory_usage']:.1f}%")
        print(f"   峰值内存使用: {metrics['peak_memory_usage']:.1f}GB")
        
        # 经济趋势分析
        if "economic_trends" in report and report["economic_trends"]:
            trends = report["economic_trends"]
            
            # 检查是否有错误或数据不足
            if "error" in trends:
                print(f"\n⚠️  经济趋势分析: {trends['error']}")
            elif "trends" in trends and isinstance(trends["trends"], dict):
                print(f"\n📊 月度经济指标变化:")
                
                # 获取所有月份
                all_months = set()
                for trend_name, trend_data in trends["trends"].items():
                    if isinstance(trend_data, list):
                        for data_point in trend_data:
                            if isinstance(data_point, dict) and "iteration" in data_point:
                                all_months.add(data_point["iteration"])
                
                # 按月份排序
                sorted_months = sorted(all_months)
                
                # 为每个趋势创建月度数据映射
                monthly_data = {}
                trend_names = {
                    "unemployment_trend": "失业率",
                    "monthly_income_trend": "当月收入",
                    "monthly_expenditure_trend": "当月支出", 
                    "monthly_savings_rate_trend": "当月储蓄率",
                    "wealth_trend": "平均财富",
                    "labor_utilization_trend": "劳动力利用率"
                }
                
                # 收集每个月份的数据
                for month in sorted_months:
                    monthly_data[month] = {}
                    for trend_name, trend_data in trends["trends"].items():
                        if isinstance(trend_data, list):
                            # 找到该月份的数据
                            month_data = next((dp for dp in trend_data if isinstance(dp, dict) and dp.get("iteration") == month), None)
                            if month_data and "value" in month_data:
                                monthly_data[month][trend_name] = month_data["value"]
                
                # 显示月度数据表格
                print(f"   {'月份':<6}", end="")
                for trend_name, display_name in trend_names.items():
                    if any(month in monthly_data and trend_name in monthly_data[month] for month in sorted_months):
                        print(f"{display_name:<12}", end="")
                print()
                
                print("   " + "-" * (6 + 12 * len(trend_names)))
                
                for month in sorted_months:
                    print(f"   {month:<6}", end="")
                    for trend_name, display_name in trend_names.items():
                        if trend_name in monthly_data[month]:
                            value = monthly_data[month][trend_name]
                            if "rate" in trend_name or "unemployment" in trend_name:
                                print(f"{value:<12.1%}", end="")
                            elif "wealth" in trend_name or "income" in trend_name or "expenditure" in trend_name:
                                print(f"${value:<11.0f}", end="")
                            else:
                                print(f"{value:<12.3f}", end="")
                        else:
                            print(f"{'N/A':<12}", end="")
                    print()
            
            # 显示总体趋势摘要
            if "trend_summary" in trends and isinstance(trends["trend_summary"], dict):
                print(f"\n📈 总体趋势摘要:")
                trend_summary = trends["trend_summary"]
                
                for trend_name, trend_info in trend_summary.items():
                    if isinstance(trend_info, dict):
                        trend_display_name = {
                        "unemployment_trend": "失业率",
                            "monthly_income_trend": "当月收入",
                            "monthly_expenditure_trend": "当月支出",
                            "monthly_savings_rate_trend": "当月储蓄率",
                            "wealth_trend": "平均财富",
                            "labor_utilization_trend": "劳动力利用率"
                    }.get(trend_name, trend_name)
                    
                    direction = trend_info.get("direction", "未知")
                    change_rate = trend_info.get("change_rate", 0)
                    
                    # 根据趋势方向选择emoji
                    emoji = "📈" if direction == "上升" else "📉" if direction == "下降" else "➡️"
                    
                    print(f"   {emoji} {trend_display_name}: {direction} ({change_rate:.1%})")
        
        # 联调指标报告
        if "joint_debug_metrics" in report and report["joint_debug_metrics"]:
            joint_metrics = report["joint_debug_metrics"]
            
            print(f"\n🔧 联调指标报告:")
            
            # 家庭数据收集指标
            if "household_metrics" in joint_metrics:
                hm = joint_metrics["household_metrics"]
                print(f"   🏠 家庭数据收集:")
                print(f"      月度记录数: {hm['monthly_tracking']['total_records']}")
                print(f"      覆盖月份数: {hm['monthly_tracking']['months_covered']}")
                print(f"      跟踪家庭数: {hm['monthly_tracking']['households_tracked']}")
                
                if "final_summary" in hm and "total_consumption" in hm["final_summary"]:
                    fs = hm["final_summary"]
                    print(f"      总消费金额: ${fs['total_consumption']:.2f}")
                    print(f"      总收入金额: ${fs['total_income']:.2f}")
                    print(f"      平均储蓄率: {fs['average_savings_rate']:.1%}")
            
            # 企业数据收集指标
            if "firm_metrics" in joint_metrics:
                fm = joint_metrics["firm_metrics"]
                print(f"   🏢 企业数据收集:")
                print(f"      月度记录数: {fm['monthly_tracking']['total_records']}")
                print(f"      覆盖月份数: {fm['monthly_tracking']['months_covered']}")
                print(f"      跟踪企业数: {fm['monthly_tracking']['firms_tracked']}")
                
                if "final_summary" in fm:
                    fs = fm["final_summary"]
                    print(f"      总销售收入: ${fs['total_revenue']:.2f}")
                    print(f"      平均员工数: {fs['average_employees']:.1f}")
                    print(f"      整体招聘成功率: {fs['overall_recruitment_rate']:.1%}")
                    
                    # 显示招聘成功率调试信息
                    if "recruitment_debug_info" in fs:
                        debug_info = fs["recruitment_debug_info"]
                        print(f"      📊 招聘统计详情:")
                        print(f"        总岗位发布数: {debug_info['total_job_postings']}")
                        print(f"        总成功招聘数: {debug_info['total_successful_hires']}")
                        print(f"        计算方式: {debug_info['calculation_method']}")
                        if debug_info['monthly_success_rates']:
                            print(f"        月度成功率: {[f'{rate:.1%}' for rate in debug_info['monthly_success_rates']]}")
            
            # 性能监控指标
            if "performance_metrics" in joint_metrics:
                pm = joint_metrics["performance_metrics"]
                print(f"   ⚡ 性能监控:")
                print(f"      总操作次数: {pm['total_operations']}")
                print(f"      总耗时: {pm['total_duration']:.2f}秒")
                print(f"      平均操作耗时: {pm['average_duration']:.3f}秒")
                
                if "operations_by_type" in pm:
                    for op_type, stats in pm["operations_by_type"].items():
                        print(f"      {op_type}: {stats['count']}次, 平均{stats['average_duration']:.3f}秒")
            
            # LLM调用指标
            if "llm_metrics" in joint_metrics:
                lm = joint_metrics["llm_metrics"]
                print(f"   🤖 LLM调用:")
                print(f"      总调用次数: {lm['total_calls']}")
                print(f"      成功率: {lm['success_rate']:.1%}")
                print(f"      总Token数: {lm['total_tokens']}")
                print(f"      平均响应时间: {lm['average_duration']:.3f}秒")
            
            # 数据质量指标
            if "data_quality" in joint_metrics:
                dq = joint_metrics["data_quality"]
                print(f"   📊 数据质量:")
                print(f"      家庭数据完整性: {dq['household_data_completeness']:.1%}")
                print(f"      企业数据完整性: {dq['firm_data_completeness']:.1%}")
                print(f"      性能数据点数: {dq['performance_data_points']}")
                print(f"      LLM数据点数: {dq['llm_data_points']}")
        
        print("="*80)
        print(f"📅 报告生成时间: {report['timestamp']}")
        print("="*80)

    async def cleanup_resources(self):
        """清理仿真资源"""
        logger.info("开始清理仿真资源...")
        
        try:
            # 停止监控
            if self.is_monitoring:
                await self.stop_monitoring()
            
            # 清理Qdrant连接
            try:
                if 'client' in globals() and client:
                    # 注意：QdrantClient通常不需要显式关闭
                    pass
            except Exception as e:
                logger.warning(f"清理Qdrant连接失败: {e}")
            
            # 清理Ray资源
            try:
                if ray.is_initialized():
                    logger.info("关闭Ray...")
                    ray.shutdown()
            except Exception as e:
                logger.warning(f"清理Ray资源失败: {e}")
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

    async def _generate_all_visualization_charts(self):
        """生成所有数据可视化图表"""
        try:
            # 创建图表输出目录
            charts_dir = os.path.join(self.experiment_output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            print(f"📊 Charts will be saved to: {charts_dir}")
            
            # 1. Product market: quarterly consumption by category bar chart
            await self._generate_monthly_consumption_chart(charts_dir)
            
            # 2. Labor market: quarterly employment rate, unemployment rate, average salary line chart
            await self._generate_labor_market_metrics_chart(charts_dir)
            
            # 3. Households: monthly income, expenditure, savings line chart
            await self._generate_household_financial_chart(charts_dir)
            
            # 4. Overall: monthly employment, total expenditure, total income bar chart
            await self._generate_overall_monthly_chart(charts_dir)
            
            # 5. Household wealth gap: before vs after savings comparison
            await self._generate_wealth_gap_chart(charts_dir)
            
            # 6. 保存仿真数据到本地文件
            await self._save_simulation_data_to_files()
            
            print(f"✅ All data visualization charts generated successfully!")
            
        except Exception as e:
            logger.error(f"Failed to generate data visualization charts: {e}")
            print(f"❌ Chart generation failed: {e}")
    
    async def _generate_monthly_consumption_chart(self, charts_dir: str):
        """生成每月各类消费品总消费柱状图"""
        try:
            print("📊 Generating monthly consumption chart...")
            
            # 收集每月消费数据和所有消费类别
            monthly_data = {}
            all_categories = set()
            
            # 先遍历一遍收集所有的消费类别
            for metric in self.household_monthly_metrics[1]:
                for category in metric.consumption_structure.keys():
                    all_categories.add(category)
            
            if not all_categories:
                print("⚠️  No consumption category data, skipping quarterly consumption chart")
                return
            
            # 按类别名称排序
            consumption_categories = sorted(list(all_categories))
            
            # 收集阅读数据
            for i in range(1,self.config.num_iterations+1):
                for metric in self.household_monthly_metrics[i]:
                    if metric.month not in monthly_data:
                        monthly_data[metric.month] = {cat: 0 for cat in consumption_categories}
                
                for category, amount in metric.consumption_structure.items():
                    if category in monthly_data[metric.month]:
                        monthly_data[metric.month][category] += amount
            
            
            # 创建柱状图
            months = sorted(monthly_data.keys())
            categories = consumption_categories
            
            fig, ax = plt.subplots(figsize=(14, 8))  # 稍微增加宽度以适应更多类别
            
            # 设置柱状图位置
            x = np.arange(len(months))
            width = 0.8 / len(categories)  # 动态调整宽度
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            for i, category in enumerate(categories):
                values = [monthly_data[q].get(category, 0) for q in months]
                ax.bar(x + i * width - width * (len(categories) - 1) / 2, values, width, label=category, color=colors[i])
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Consumption Amount ($)')
            ax.set_title('Quarterly Consumption by Category')
            ax.set_xticks(x)
            ax.set_xticklabels(months)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(charts_dir, "monthly_consumption_by_category.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Monthly consumption chart saved: {chart_path}")
            print(f"   📋 Consumption categories: {', '.join(categories)}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate quarterly consumption chart: {e}")
    
    async def _generate_labor_market_metrics_chart(self, charts_dir: str):
        """生成劳动力市场指标折线图（月度）"""
        try:
            print("📊 Generating labor market metrics chart...")
            
            # 收集月度劳动力数据
            monthly_labor_data = {}
            
            for metric_data in self.economic_metrics_history:
                if "iteration" in metric_data:
                    month = metric_data["iteration"]
                    
                    if month not in monthly_labor_data:
                        monthly_labor_data[month] = {
                            'employment_rates': [],
                            'unemployment_rates': [],       
                            'avg_wages': []
                        }
                    
                    # 就业率和失业率
                    employment_rate = metric_data['employment_statistics']['labor_utilization_rate']
                    unemployment_rate = 1 - employment_rate
                    
                    monthly_labor_data[month]['employment_rates'].append(employment_rate)
                    monthly_labor_data[month]['unemployment_rates'].append(unemployment_rate)
                    
                    # 平均薪资（使用工人平均工资，与Gini图保持一致）
                    if 'inequality_analysis' in metric_data:
                        avg_wage_per_worker = metric_data['inequality_analysis']['average_wage_per_worker']
                        monthly_labor_data[month]['avg_wages'].append(avg_wage_per_worker)
            
            if not monthly_labor_data:
                print("⚠️  No labor market data, skipping labor market chart")
                return
            
            # 计算月度平均值
            months = sorted(monthly_labor_data.keys())
            employment_rates = []
            unemployment_rates = []
            avg_monthly_wages = []
            
            for month in months:
                data = monthly_labor_data[month]
                employment_rates.append(np.mean(data['employment_rates']) * 100)  # 转换为百分比
                unemployment_rates.append(np.mean(data['unemployment_rates']) * 100)
                avg_monthly_wages.append(np.mean(data['avg_wages']))
            
            # 创建双y轴折线图
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # 左y轴：就业率和失业率
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Rate (%)', color='tab:blue')
            line1 = ax1.plot(months, employment_rates, 'b-o', label='Employment Rate', linewidth=2, markersize=6)
            line2 = ax1.plot(months, unemployment_rates, 'r-s', label='Unemployment Rate', linewidth=2, markersize=6)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True, alpha=0.3)
            
            # 右y轴：平均工资
            ax2 = ax1.twinx()
            ax2.set_ylabel('Average Wage per Worker ($)', color='tab:green')
            line3 = ax2.plot(months, avg_monthly_wages, 'g-^', label='Average Wage per Worker', linewidth=2, markersize=6)
            ax2.tick_params(axis='y', labelcolor='tab:green')
            
            # 合并图例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            plt.title('Monthly Labor Market Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "monthly_labor_market_metrics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Labor market chart saved: {chart_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate labor market chart: {e}")
    
    async def _generate_household_financial_chart(self, charts_dir: str):
        """生成家庭按月收入、支出、储蓄变化折线图"""
        try:
            print("📊 Generating household financial chart...")
            
            # 收集月度家庭财务数据
            monthly_data = {}
            
            # for metric in self.household_monthly_metrics:
            #     month = metric.month
            #     if month not in monthly_data:
            #         monthly_data[month] = {
            #             'total_income': 0,
            #             'total_expenditure': 0,
            #             'total_savings': 0,
            #             'household_count': 0
            #         }
                
            #     monthly_data[month]['total_income'] += metric.monthly_income
            #     monthly_data[month]['total_expenditure'] += metric.monthly_expenditure
            #     monthly_data[month]['total_savings'] += (metric.monthly_income - metric.monthly_expenditure)
            #     monthly_data[month]['household_count'] += 1

            for month in range(1, self.config.num_iterations + 1):
                if month not in monthly_data:
                    monthly_data[month] = {
                        'total_income': 0,
                        'total_expenditure': 0,
                        'total_savings': 0,
                        'household_count': 0
                    }
                monthly_data[month]['total_income'] = sum(s.monthly_income for s in self.household_monthly_metrics[month])
                monthly_data[month]['total_expenditure'] = sum(s.monthly_expenditure for s in self.household_monthly_metrics[month])
                monthly_data[month]['total_savings'] = sum(s.monthly_income - s.monthly_expenditure for s in self.household_monthly_metrics[month])
                monthly_data[month]['household_count'] = len(self.household_monthly_metrics[month])

            if not monthly_data:
                print("⚠️  No household financial data, skipping household financial chart")
                return
            
            # 计算平均值
            months = sorted(monthly_data.keys())
            avg_incomes = []
            avg_expenditures = []
            avg_savings = []
            
            for month in months:
                data = monthly_data[month]
                count = data['household_count'] if data['household_count'] > 0 else 1
                avg_incomes.append(data['total_income'] / count)
                avg_expenditures.append(data['total_expenditure'] / count)
                avg_savings.append(data['total_savings'] / count)
            
            # 创建折线图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.plot(months, avg_incomes, 'g-o', label='Average Income', linewidth=2, markersize=5)
            ax.plot(months, avg_expenditures, 'r-s', label='Average Expenditure', linewidth=2, markersize=5)
            ax.plot(months, avg_savings, 'b-^', label='Average Savings', linewidth=2, markersize=5)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount ($)')
            ax.set_title('Monthly Household Financial Changes', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "monthly_household_financial_changes.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            
            # 同时保存 SVG 格式（真矢量图）
            chart_path_svg = os.path.join(charts_dir, "monthly_household_financial_changes.svg")
            plt.savefig(chart_path_svg, format='svg', bbox_inches='tight')
            
            plt.close()
            
            print(f"   ✅ Household financial chart saved: {chart_path}")
            print(f"   ✅ SVG version saved: {chart_path_svg}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate household financial chart: {e}")
    
    async def _generate_overall_monthly_chart(self, charts_dir: str):
        """生成整体每月就业人数、总支出、总收入柱状图"""
        try:
            print("📊 Generating overall monthly indicators chart...")
            
            # 收集月度整体数据
            monthly_overall_data = {}
            
            # 从经济指标历史收集数据
            for metric_data in self.economic_metrics_history:
                if "iteration" in metric_data:
                    month = metric_data["iteration"]
                    monthly_overall_data[month] = {
                        'employed_people': metric_data['employment_statistics']['total_labor_force_employed'],
                        'total_monthly_income': metric_data['income_expenditure_analysis']['total_monthly_income'],
                        'total_monthly_expenditure': metric_data['income_expenditure_analysis']['total_monthly_expenditure']
                    }
            
            # # 如果经济指标历史数据不足，从家庭月度指标补充
            # for metric in self.household_monthly_metrics:
            #     month = metric.month
            #     if month not in monthly_overall_data:
            #         monthly_overall_data[month] = {
            #             'employed_people': 0,
            #             'total_income': 0,
            #             'total_expenditure': 0
            #         }
                
            #     monthly_overall_data[month]['total_income'] += metric.monthly_income
            #     monthly_overall_data[month]['total_expenditure'] += metric.monthly_expenditure
            
            if not monthly_overall_data:
                print("⚠️  No overall data, skipping overall monthly chart")
                return
            
            months = sorted(monthly_overall_data.keys())
            employed_people = [monthly_overall_data[m]['employed_people'] for m in months]
            total_incomes = [monthly_overall_data[m]['total_monthly_income'] for m in months]
            total_expenditures = [monthly_overall_data[m]['total_monthly_expenditure'] for m in months]
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 上图：就业人数
            ax1.bar(months, employed_people, color='skyblue', alpha=0.7, label='Employed People')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Number of Employed')
            ax1.set_title('Monthly Employment', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            ax1.legend()
            
            # 下图：总收入和总支出
            x = np.arange(len(months))
            width = 0.35
            
            ax2.bar(x - width/2, total_incomes, width, label='Total Income', color='green', alpha=0.7)
            ax2.bar(x + width/2, total_expenditures, width, label='Total Expenditure', color='red', alpha=0.7)
            
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Amount ($)')
            ax2.set_title('Monthly Total Income vs Expenditure', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(months)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            plt.suptitle('Overall Economic Monthly Indicators', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "overall_monthly_indicators.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Overall monthly chart saved: {chart_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate overall monthly chart: {e}")
    
    async def _generate_wealth_gap_chart(self, charts_dir: str):
        """生成家庭财富差距对比图表（仿真前后储蓄对比）"""
        try:
            print("📊 Generating household wealth gap chart...")
            
            # 收集家庭初始和最终储蓄数据
            household_wealth_data = []
            
            # 收集每个家庭的最终储蓄
            for household in self.households:
                try:
                    # 获取家庭ID和最终储蓄
                    household_id = household.household_id
                    final_savings = await household.get_balance_ref()
                    
                    # 获取初始储蓄（从记录的字典中）
                    initial_savings = self.initial_household_savings.get(household_id, 0.0)
                    
                    household_wealth_data.append({
                        'household_id': household_id,
                        'initial_savings': initial_savings,
                        'final_savings': final_savings,
                        'savings_change': final_savings - initial_savings
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get wealth data for household: {e}")
                    continue
            
            if not household_wealth_data:
                print("⚠️  No household wealth data, skipping wealth gap chart")
                return
            
            # 按初始储蓄排序
            household_wealth_data.sort(key=lambda x: x['initial_savings'])
            
            # 提取数据
            household_ids = [data['household_id'] for data in household_wealth_data]
            initial_savings = [data['initial_savings'] for data in household_wealth_data]
            final_savings = [data['final_savings'] for data in household_wealth_data]
            savings_changes = [data['savings_change'] for data in household_wealth_data]
            
            # 创建单一图表，包含两个柱状图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 左图: 仿真前后储蓄对比柱状图
            x = np.arange(len(household_ids))
            width = 0.35
            
            ax1.bar(x - width/2, initial_savings, width, label='Initial Savings', color='lightblue', alpha=0.7)
            ax1.bar(x + width/2, final_savings, width, label='Final Savings', color='darkblue', alpha=0.7)
            
            ax1.set_xlabel('Household ID')
            ax1.set_ylabel('Savings Amount ($)')
            ax1.set_title('Household Savings: Before vs After Simulation', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(household_ids, rotation=45)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # 右图: 储蓄分布直方图（使用动态区间）
            # 合并所有储蓄数据来确定合适的区间
            all_savings = initial_savings + final_savings
            min_savings = min(all_savings)
            max_savings = max(all_savings)
            
            # 如果数据范围太小，使用默认区间数
            if max_savings - min_savings < 1000:
                bins = min(5, len(household_ids))  # 至少5个区间或家庭数量
            else:
                # 根据数据范围动态确定区间数
                data_range = max_savings - min_savings
                if data_range < 10000:
                    bins = 5
                elif data_range < 50000:
                    bins = 8
                else:
                    bins = 10
            
            # 创建动态区间边界
            bin_edges = np.linspace(min_savings, max_savings, bins + 1)
            
            # 绘制直方图
            ax2.hist([initial_savings, final_savings], bins=bin_edges, alpha=0.7, 
                    label=['Initial Savings', 'Final Savings'], color=['lightblue', 'darkblue'])
            ax2.set_xlabel('Savings Amount ($)')
            ax2.set_ylabel('Number of Households')
            ax2.set_title('Savings Distribution Comparison', fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # 计算基尼系数
            initial_gini = self._calculate_gini_coefficient(initial_savings)
            final_gini = self._calculate_gini_coefficient(final_savings)
            
            # 计算统计信息
            initial_mean = np.mean(initial_savings)
            final_mean = np.mean(final_savings)
            initial_std = np.std(initial_savings)
            final_std = np.std(final_savings)
            
            # 在图表下方添加统计信息
            stats_text = f"""Wealth Gap Analysis Summary:
Initial: Avg=${initial_mean:,.0f}, Std=${initial_std:,.0f}, Gini={initial_gini:.3f}
Final: Avg=${final_mean:,.0f}, Std=${final_std:,.0f}, Gini={final_gini:.3f}
Change: Avg=${final_mean - initial_mean:+,.0f}, Gini={final_gini - initial_gini:+.3f}
Growth: {sum(1 for c in savings_changes if c > 0)}/{len(savings_changes)} households positive"""
            
            plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.suptitle('Household Wealth Gap Analysis: Before vs After Simulation', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # 为底部统计信息留出空间
            
            chart_path = os.path.join(charts_dir, "household_wealth_gap_analysis.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Wealth gap chart saved: {chart_path}")
            print(f"   📊 Initial Gini: {initial_gini:.3f}, Final Gini: {final_gini:.3f}")
            print(f"   📊 Dynamic bins used: {bins}, Range: ${min_savings:,.0f} - ${max_savings:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate wealth gap chart: {e}")

    async def _collect_unemployment_data(self, current_month: int) -> Dict[str, Any]:
        """
        收集失业统计数据，使用现有的家庭统计逻辑
        
        Args:
            current_month: 当前月份
            
        Returns:
            包含失业统计信息的字典
        """
        try:
            # 快速收集关键指标
            total_households = len(self.households)
            employed_households = 0
            total_labor_force = 0
            total_employed_people = 0
            
            # 并行收集所有家庭的劳动力统计数据
            async def collect_labor_data(household):
                try:
                    household_labor_force = 0
                    household_employed = 0
                    
                    if hasattr(household, 'labor_hours') and household.labor_hours:
                        household_labor_force = len(household.labor_hours)
                        
                        # 统计已就业人数
                        for labor_hour in household.labor_hours:
                            if hasattr(labor_hour, 'is_valid') and hasattr(labor_hour, 'company_id'):
                                if not labor_hour.is_valid and labor_hour.company_id is not None:
                                    household_employed += 1
                    
                    return {
                        'labor_force': household_labor_force,
                        'employed': household_employed
                    }
                        
                except Exception as e:
                    logger.debug(f"获取家庭 {household.household_id} 劳动力数据失败: {e}")
                    return {'labor_force': 0, 'employed': 0}
            
            # 并行收集所有家庭数据
            labor_tasks = [collect_labor_data(h) for h in self.households]
            labor_results = await asyncio.gather(*labor_tasks, return_exceptions=True)
            
            # 汇总统计数据
            for result in labor_results:
                if not isinstance(result, Exception):
                    total_labor_force += result['labor_force']
                    total_employed_people += result['employed']
                    
                    if result['employed'] > 0:
                        employed_households += 1
            
            # 计算就业率
            employment_rate = employed_households / total_households if total_households > 0 else 0
            labor_utilization_rate = total_employed_people / total_labor_force if total_labor_force > 0 else 0
            
            # 构建失业统计数据
            unemployment_data = {
                'total_labor_force_unemployed': total_labor_force - total_employed_people,
                'household_unemployment_rate': 1 - employment_rate,
                'total_labor_force_available': total_labor_force,
                'total_labor_force_employed': total_employed_people,
                'total_open_positions': 0  # 这个值暂时设为0，因为还没有统计开放岗位
            }
            
            logger.info(f"失业数据收集完成: 总劳动力={total_labor_force}, 已就业={total_employed_people}, 失业={unemployment_data['total_labor_force_unemployed']}")
            
            return unemployment_data
            
        except Exception as e:
            logger.error(f"收集失业数据失败: {e}")
            return None

    async def _collect_all_household_data_once(self, current_month: int) -> Dict[str, Any]:
        """
        一次性收集所有家庭数据，避免重复调用
        
        Args:
            current_month: 当前月份
            
        Returns:
            包含所有家庭数据的字典
        """
        try:
            print(f"📈 开始一次性收集 {len(self.households)} 个家庭的所有数据...")
            month = current_month
            async def collect_household_all_data(household):
                try:
                    # 并行获取所有需要的数据
                    monthly_stats_task = self.economic_center.compute_household_monthly_stats.remote(
                        household.household_id, current_month
                    )
                    settlement_task = self.economic_center.compute_household_settlement.remote(
                        household.household_id
                    )
                    balance_task = household.get_balance_ref()
                    
                    # 并行执行所有任务
                    results = await asyncio.gather(
                        monthly_stats_task, 
                        settlement_task, 
                        balance_task, 
                        return_exceptions=True
                    )
                    
                    # 解析结果
                    monthly_income_dict, monthly_expense_dict = results[0] if not isinstance(results[0], Exception) else ({}, {})
                    current_balance = results[1] if not isinstance(results[1], Exception) else 0
                    
                    # 从月度统计字典中提取指定月份的数据
                    monthly_income = monthly_income_dict.get(month, 0) if isinstance(monthly_income_dict, dict) else 0
                    monthly_expenditure = monthly_expense_dict.get(month, 0) if isinstance(monthly_expense_dict, dict) else 0
                    
                    # 计算储蓄率
                    savings_rate = (monthly_income - monthly_expenditure) / monthly_income if monthly_income > 0 else 0
                    
                    # 计算收入变化率
                    income_change_rate = 0.0
                    if month > 1 and len(results) > 2 and not isinstance(results[2], Exception):
                        prev_monthly_income_dict, prev_monthly_expense_dict = results[2]
                        prev_income = prev_monthly_income_dict.get(month - 1, 0) if isinstance(prev_monthly_income_dict, dict) else 0
                        if prev_income > 0:
                            income_change_rate = (monthly_income - prev_income) / prev_income
                    
                    # 使用实际的消费预算数据
                    consumption_structure = {}
                    try:
                        # 获取household的实际消费预算数据
                        consume_budget_data = household.get_consume_budget_data()
                        if month in consume_budget_data:
                            consumption_structure = consume_budget_data[month]
                        else:
                            # 如果没有实际数据，使用简化的消费结构作为备选
                            consumption_structure = {
                                "food": monthly_expenditure * 0.25,
                                "housing": monthly_expenditure * 0.30,
                                "transportation": monthly_expenditure * 0.15,
                                "entertainment": monthly_expenditure * 0.10,
                                "clothing": monthly_expenditure * 0.08,
                                "healthcare": monthly_expenditure * 0.07,
                                "education": monthly_expenditure * 0.05
                            }
                    except Exception as e:
                        logger.warning(f"获取家庭 {household.household_id} 第{month}月消费预算失败: {e}")
                        # 使用简化的消费结构作为备选
                        consumption_structure = {
                            "food": monthly_expenditure * 0.25,
                            "housing": monthly_expenditure * 0.30,
                            "transportation": monthly_expenditure * 0.15,
                            "entertainment": monthly_expenditure * 0.10,
                            "clothing": monthly_expenditure * 0.08,
                            "healthcare": monthly_expenditure * 0.07,
                            "education": monthly_expenditure * 0.05
                        }
                    
                    # 创建家庭月度指标
                    return HouseholdMonthlyMetrics(
                        household_id=household.household_id,
                        month=month,
                        monthly_income=monthly_income,
                        monthly_expenditure=monthly_expenditure,
                        savings_rate=savings_rate,
                        consumption_structure=consumption_structure,
                        income_change_rate=income_change_rate
                    )
            
                except Exception as e:
                    logger.warning(f"收集家庭 {household.household_id} 月度数据失败: {e}")
                    return None
            
            # 并行收集所有家庭数据
            all_data_tasks = [collect_household_all_data(h) for h in self.households]
            all_household_data = await asyncio.gather(*all_data_tasks, return_exceptions=True)
            
            # 汇总统计数据
            total_monthly_income = 0
            total_monthly_expenditure = 0
            total_cumulative_income = 0
            total_cumulative_expenditure = 0
            total_current_balance = 0
            employed_households = 0
            total_labor_force = 0
            total_employed_people = 0
            valid_data_count = 0
            
            for data in all_household_data:
                if data and not isinstance(data, Exception):
                    total_monthly_income += data['monthly_income']
                    total_monthly_expenditure += data['monthly_expenditure']
                    total_cumulative_income += data['cumulative_income']
                    total_cumulative_expenditure += data['cumulative_spent']
                    total_current_balance += data['current_balance']
                    total_labor_force += data['labor_force']
                    total_employed_people += data['employed']
                    
                    if data['employed'] > 0:
                        employed_households += 1
                    
                    valid_data_count += 1
            
            print(f"✅ 家庭数据收集完成: {valid_data_count}/{len(self.households)} 个家庭")
            
            # 计算平均值
            sample_size = len(self.households)
            avg_monthly_income = total_monthly_income / sample_size if sample_size > 0 else 0
            avg_monthly_expenditure = total_monthly_expenditure / sample_size if sample_size > 0 else 0
            avg_cumulative_income = total_cumulative_income / sample_size if sample_size > 0 else 0
            avg_cumulative_expenditure = total_cumulative_expenditure / sample_size if sample_size > 0 else 0
            avg_current_balance = total_current_balance / sample_size if sample_size > 0 else 0
            
            # 计算就业率
            employment_rate = employed_households / sample_size if sample_size > 0 else 0
            labor_utilization_rate = total_employed_people / total_labor_force if total_labor_force > 0 else 0
            
            # 计算储蓄率
            monthly_savings_rate = (total_monthly_income - total_monthly_expenditure) / total_monthly_income if total_monthly_income > 0 else 0
            cumulative_savings_rate = (total_cumulative_income - total_cumulative_expenditure) / total_cumulative_income if total_cumulative_income > 0 else 0
            
            # 构建失业统计数据
            unemployment_data = {
                'total_labor_force_unemployed': total_labor_force - total_employed_people,
                'household_unemployment_rate': 1 - employment_rate,
                'total_labor_force_available': total_labor_force,
                'total_labor_force_employed': total_employed_people,
                'total_open_positions': 0
            }
            
            # 返回所有汇总数据
            return {
                'summary_stats': {
                    'total_households': sample_size,
                    'employed_households': employed_households,
                    'employment_rate': employment_rate,
                    'total_labor_force': total_labor_force,
                    'total_employed_people': total_employed_people,
                    'labor_utilization_rate': labor_utilization_rate
                },
                'income_expenditure': {
                    'total_monthly_income': total_monthly_income,
                    'total_monthly_expenditure': total_monthly_expenditure,
                    'avg_monthly_income': avg_monthly_income,
                    'avg_monthly_expenditure': avg_monthly_expenditure,
                    'monthly_savings_rate': monthly_savings_rate,
                    'total_cumulative_income': total_cumulative_income,
                    'total_cumulative_expenditure': total_cumulative_expenditure,
                    'avg_cumulative_income': avg_cumulative_income,
                    'avg_cumulative_expenditure': avg_cumulative_expenditure,
                    'cumulative_savings_rate': cumulative_savings_rate,
                    'total_current_balance': total_current_balance,
                    'avg_current_balance': avg_current_balance
                },
                'unemployment_data': unemployment_data,
                'individual_data': all_household_data
            }
            
        except Exception as e:
            logger.error(f"收集家庭数据失败: {e}")
            return None

    def _get_consistent_firm_id(self, firm) -> str:
        """统一获取企业ID的方法"""
        if hasattr(firm, 'company_id') and firm.company_id:
            return firm.company_id
        elif hasattr(firm, 'firm_id') and firm.firm_id:
            return firm.firm_id
        else:
            # 作为最后的备选方案，使用对象的字符串表示
            return str(id(firm))

    def _find_household_by_id(self, household_id: str, households_list=None):
        """安全地通过ID查找家庭对象"""
        if households_list is None:
            households_list = self.households

        for household in households_list:
            if household.household_id == household_id:
                return household
        return None

    def _find_firm_by_id(self, firm_id: str, firms_list=None):
        """安全地通过ID查找企业对象"""
        if firms_list is None:
            firms_list = self.firms

        for firm in firms_list:
            if self._get_consistent_firm_id(firm) == firm_id:
                return firm
        return None

    async def _collect_household_purchase_records(self, month: int):
        """
        收集家庭每个月的成功购买商品记录
        
        Args:
            month: 月份
        """
        try:
            # 从经济中心获取所有交易记录
            all_transactions = await self.economic_center.query_all_tx.remote()
            
            # 筛选出指定月份的购买交易（type='purchase'）
            # 家庭购买交易的特征：type='purchase', sender_id是家庭ID, receiver_id是企业ID或不是政府
            purchase_records = []
            household_ids = {h.household_id for h in self.households} if self.households else set()
            
            for tx in all_transactions:
                # 检查是否是购买交易且是目标月份
                if (hasattr(tx, 'type') and tx.type == 'purchase' and 
                    hasattr(tx, 'month') and tx.month == month):
                    
                    sender_id = getattr(tx, 'sender_id', '')
                    receiver_id = getattr(tx, 'receiver_id', '')
                    
                    # 判断是否是家庭购买：sender_id在家庭列表中，且receiver_id不是政府
                    is_household_purchase = (
                        sender_id in household_ids or 
                        (receiver_id != 'gov_main_simulation' and receiver_id != 'bank_main_simulation')
                    )
                    
                    if is_household_purchase:
                        # 提取商品信息
                        if hasattr(tx, 'assets') and tx.assets:
                            for product in tx.assets:
                                try:
                                    # 序列化商品信息
                                    if hasattr(product, 'model_dump'):
                                        product_dict = product.model_dump()
                                    elif hasattr(product, 'dict'):
                                        product_dict = product.dict()
                                    elif isinstance(product, dict):
                                        product_dict = product
                                    else:
                                        # 手动提取属性
                                        product_dict = {
                                            'product_id': getattr(product, 'product_id', None),
                                            'name': getattr(product, 'name', None),
                                            'price': getattr(product, 'price', None),
                                            'amount': getattr(product, 'amount', None),
                                            'classification': getattr(product, 'classification', None),
                                            'brand': getattr(product, 'brand', None),
                                            'description': getattr(product, 'description', None),
                                            'manufacturer': getattr(product, 'manufacturer', None),
                                            'attributes': getattr(product, 'attributes', None),
                                            'is_food': getattr(product, 'is_food', None),
                                            'nutrition_supply': getattr(product, 'nutrition_supply', None),
                                            'satisfaction_attributes': getattr(product, 'satisfaction_attributes', None),
                                            'duration_months': getattr(product, 'duration_months', None),
                                            'expiration_date': str(getattr(product, 'expiration_date', None)) if hasattr(product, 'expiration_date') and getattr(product, 'expiration_date') else None
                                        }
                                    
                                    # 构建购买记录
                                    purchase_record = {
                                        'transaction_id': getattr(tx, 'id', None),
                                        'household_id': getattr(tx, 'sender_id', None),
                                        'seller_id': getattr(tx, 'receiver_id', None),
                                        'month': month,
                                        'total_amount': getattr(tx, 'amount', None),
                                        'product': product_dict,
                                        'quantity': product_dict.get('amount', 1.0),
                                        'unit_price': product_dict.get('price', 0.0),
                                        'total_price': product_dict.get('price', 0.0) * product_dict.get('amount', 1.0)
                                    }
                                    purchase_records.append(purchase_record)
                                except Exception as e:
                                    logger.warning(f"处理购买记录中的商品信息失败: {e}")
                                    continue
            
            # 存储购买记录
            if month not in self.household_purchase_records:
                self.household_purchase_records[month] = []
            self.household_purchase_records[month].extend(purchase_records)
            
            logger.info(f"✅ 第 {month} 月购买记录收集完成: {len(purchase_records)} 条购买记录")
            print(f"✅ 第 {month} 月购买记录收集完成: {len(purchase_records)} 条购买记录")
            
        except Exception as e:
            logger.error(f"收集第 {month} 月购买记录失败: {e}")
            import traceback
            traceback.print_exc()

    def _serialize_metric_for_json(self, metric):
        """将指标对象序列化为JSON可序列化的格式"""
        if isinstance(metric, dict):
            # 如果已经是字典，直接返回副本
            result = metric.copy()
        elif hasattr(metric, '__dict__'):
            # 如果是对象，转换为字典
            result = metric.__dict__.copy()
        else:
            # 其他类型，尝试转换为字符串
            result = {"value": str(metric), "type": type(metric).__name__}

        # 处理所有可能的日期时间字段
        datetime_fields = ['timestamp', 'created_at', 'updated_at', 'start_time', 'end_time', 'date']
        for field in datetime_fields:
            if field in result:
                value = result[field]
                if isinstance(value, (datetime, date)):
                    result[f"{field}_iso"] = value.isoformat()
                    result[field] = value.isoformat()  # 同时保留原始字段
                elif isinstance(value, (int, float)) and field == 'timestamp':
                    # 处理timestamp数值
                    try:
                        dt = datetime.fromtimestamp(value)
                        result[f"{field}_readable"] = dt.isoformat()
                    except (ValueError, OSError):
                        pass  # 无效的timestamp，跳过

        return result

    def _backup_existing_file(self, file_path: str):
        """如果文件存在，则创建备份"""
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{int(time.time())}"
            try:
                shutil.copy2(file_path, backup_path)
                print(f"   📋 已创建备份: {backup_path}")
                return backup_path
            except Exception as e:
                logger.warning(f"创建文件备份失败 {file_path}: {e}")
        return None

    def _validate_data_integrity(self, data: Any, data_type: str) -> bool:
        """验证数据完整性"""
        try:
            if data is None:
                logger.warning(f"数据验证失败: {data_type} 为 None")
                return False

            if data_type == "economic_metrics_history":
                if not isinstance(data, list):
                    logger.warning(f"经济指标历史数据类型错误: 期望 list，实际 {type(data)}")
                    return False
                if len(data) == 0:
                    logger.warning("经济指标历史数据为空")
                    return False

            elif data_type == "household_monthly_metrics":
                if not isinstance(data, dict):
                    logger.warning(f"家庭月度指标数据类型错误: 期望 dict，实际 {type(data)}")
                    return False
                if len(data) == 0:
                    logger.warning("家庭月度指标数据为空")
                    return False
                # 验证月份键的格式
                for month in data.keys():
                    if not isinstance(month, (str, int)):
                        logger.warning(f"月份键格式错误: {month} ({type(month)})")
                        return False

            elif data_type == "firm_monthly_metrics":
                if not isinstance(data, list):
                    logger.warning(f"企业月度指标数据类型错误: 期望 list，实际 {type(data)}")
                    return False
                if len(data) == 0:
                    logger.warning("企业月度指标数据为空")
                    return False

            elif data_type == "performance_metrics":
                if not isinstance(data, list):
                    logger.warning(f"性能指标数据类型错误: 期望 list，实际 {type(data)}")
                    return False

            elif data_type == "llm_metrics":
                if not isinstance(data, list):
                    logger.warning(f"LLM指标数据类型错误: 期望 list，实际 {type(data)}")
                    return False

            logger.info(f"数据完整性验证通过: {data_type}")
            return True

        except Exception as e:
            logger.error(f"数据完整性验证过程中出错 ({data_type}): {e}")
            return False

    async def _save_simulation_data_to_files(self):
        """保存仿真数据到本地文件"""
        try:
            # 创建数据输出目录
            data_dir = os.path.join(self.experiment_output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            print(f"💾 Simulation data will be saved to: {data_dir}")
            
            # 1. 保存经济指标历史数据
            if self.economic_metrics_history:
                # 验证数据完整性
                if not self._validate_data_integrity(self.economic_metrics_history, "economic_metrics_history"):
                    print("   ⚠️  跳过保存经济指标历史数据: 数据验证失败")
                else:
                    economic_data_path = os.path.join(data_dir, "economic_metrics_history.json")

                    # 创建备份
                    self._backup_existing_file(economic_data_path)

                    # 转换数据为可序列化的格式
                    serializable_economic_data = []
                    for metric in self.economic_metrics_history:
                        try:
                            serializable_data = self._serialize_metric_for_json(metric)
                            serializable_economic_data.append(serializable_data)
                        except Exception as e:
                            logger.warning(f"无法序列化经济指标数据: {e}")
                            serializable_economic_data.append({"error": str(e), "original_type": type(metric).__name__})

                    with open(economic_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_economic_data, f, ensure_ascii=False, indent=2, default=str)

                    print(f"   ✅ Economic metrics history saved: {economic_data_path}")
                    print(f"   📊 Total economic metrics records: {len(serializable_economic_data)}")
            else:
                print("   ⚠️  No economic metrics history data to save")
            
            # 2. 保存家庭月度指标数据
            if self.household_monthly_metrics:
                # 验证数据完整性
                if not self._validate_data_integrity(self.household_monthly_metrics, "household_monthly_metrics"):
                    print("   ⚠️  跳过保存家庭月度指标数据: 数据验证失败")
                else:
                    household_data_path = os.path.join(data_dir, "household_monthly_metrics.json")

                    # 创建备份
                    self._backup_existing_file(household_data_path)

                    # 转换数据为可序列化的格式
                    serializable_household_data = {}
                    for month, metrics_list in self.household_monthly_metrics.items():
                        serializable_household_data[month] = []
                        for metric in metrics_list:
                            try:
                                serializable_data = self._serialize_metric_for_json(metric)
                                serializable_household_data[month].append(serializable_data)
                            except Exception as e:
                                logger.warning(f"无法序列化家庭指标数据 (月份 {month}): {e}")
                                serializable_household_data[month].append({"error": str(e), "original_type": type(metric).__name__})

                    with open(household_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_household_data, f, ensure_ascii=False, indent=2, default=str)

                    print(f"   ✅ Household monthly metrics saved: {household_data_path}")
                    total_household_records = sum(len(metrics) for metrics in serializable_household_data.values())
                    print(f"   📊 Total household metrics records: {total_household_records} across {len(serializable_household_data)} months")
            else:
                print("   ⚠️  No household monthly metrics data to save")
            
            # 3. 保存企业月度指标数据
            if self.firm_monthly_metrics:
                # 验证数据完整性
                if not self._validate_data_integrity(self.firm_monthly_metrics, "firm_monthly_metrics"):
                    print("   ⚠️  跳过保存企业月度指标数据: 数据验证失败")
                else:
                    firm_data_path = os.path.join(data_dir, "firm_monthly_metrics.json")

                    # 创建备份
                    self._backup_existing_file(firm_data_path)

                    # 转换数据为可序列化的格式
                    serializable_firm_data = []
                    for metric in self.firm_monthly_metrics:
                        try:
                            serializable_data = self._serialize_metric_for_json(metric)
                            serializable_firm_data.append(serializable_data)
                        except Exception as e:
                            logger.warning(f"无法序列化企业指标数据: {e}")
                            serializable_firm_data.append({"error": str(e), "original_type": type(metric).__name__})

                    with open(firm_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_firm_data, f, ensure_ascii=False, indent=2, default=str)

                    print(f"   ✅ Firm monthly metrics saved: {firm_data_path}")
                    print(f"   📊 Total firm metrics records: {len(serializable_firm_data)}")
            else:
                print("   ⚠️  No firm monthly metrics data to save")
            
            # 4. 保存家庭购买记录数据
            if self.household_purchase_records:
                purchase_records_path = os.path.join(data_dir, "household_purchase_records.json")
                
                # 创建备份
                self._backup_existing_file(purchase_records_path)
                
                # 转换数据为可序列化的格式
                serializable_purchase_data = {}
                for month, records_list in self.household_purchase_records.items():
                    serializable_purchase_data[month] = []
                    for record in records_list:
                        try:
                            # 确保所有数据都是可序列化的
                            serializable_record = {}
                            for key, value in record.items():
                                if key == 'product' and isinstance(value, dict):
                                    # 处理商品字典中的日期等特殊类型
                                    serializable_record[key] = {}
                                    for k, v in value.items():
                                        if hasattr(v, 'isoformat'):  # datetime/date对象
                                            serializable_record[key][k] = v.isoformat()
                                        else:
                                            serializable_record[key][k] = v
                                elif hasattr(value, 'isoformat'):  # datetime/date对象
                                    serializable_record[key] = value.isoformat()
                                else:
                                    serializable_record[key] = value
                            serializable_purchase_data[month].append(serializable_record)
                        except Exception as e:
                            logger.warning(f"无法序列化购买记录 (月份 {month}): {e}")
                            serializable_purchase_data[month].append({"error": str(e), "original_record": str(record)})
                
                with open(purchase_records_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_purchase_data, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"   ✅ Household purchase records saved: {purchase_records_path}")
                total_purchase_records = sum(len(records) for records in serializable_purchase_data.values())
                print(f"   📊 Total purchase records: {total_purchase_records} across {len(serializable_purchase_data)} months")
            else:
                print("   ⚠️  No household purchase records data to save")
            
            # 5. 保存性能监控数据
            if self.performance_metrics:
                # 验证数据完整性（允许为空列表）
                if not self._validate_data_integrity(self.performance_metrics, "performance_metrics"):
                    print("   ⚠️  跳过保存性能监控数据: 数据验证失败")
                else:
                    performance_data_path = os.path.join(data_dir, "performance_metrics.json")

                    # 创建备份
                    self._backup_existing_file(performance_data_path)

                    serializable_performance_data = []
                    for metric in self.performance_metrics:
                        try:
                            serializable_data = self._serialize_metric_for_json(metric)
                            serializable_performance_data.append(serializable_data)
                        except Exception as e:
                            logger.warning(f"无法序列化性能指标数据: {e}")
                            serializable_performance_data.append({"error": str(e), "original_type": type(metric).__name__})

                    with open(performance_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_performance_data, f, ensure_ascii=False, indent=2, default=str)

                    print(f"   ✅ Performance metrics saved: {performance_data_path}")
                    print(f"   📊 Total performance records: {len(serializable_performance_data)}")
            else:
                print("   ⚠️  No performance metrics data to save")
            
            # 5. 保存LLM调用指标数据
            if self.llm_metrics:
                # 验证数据完整性（允许为空列表）
                if not self._validate_data_integrity(self.llm_metrics, "llm_metrics"):
                    print("   ⚠️  跳过保存LLM调用指标数据: 数据验证失败")
                else:
                    llm_data_path = os.path.join(data_dir, "llm_metrics.json")

                    # 创建备份
                    self._backup_existing_file(llm_data_path)

                    serializable_llm_data = []
                    for metric in self.llm_metrics:
                        try:
                            serializable_data = self._serialize_metric_for_json(metric)
                            serializable_llm_data.append(serializable_data)
                        except Exception as e:
                            logger.warning(f"无法序列化LLM指标数据: {e}")
                            serializable_llm_data.append({"error": str(e), "original_type": type(metric).__name__})

                    with open(llm_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_llm_data, f, ensure_ascii=False, indent=2, default=str)

                    print(f"   ✅ LLM metrics saved: {llm_data_path}")
                    print(f"   📊 Total LLM records: {len(serializable_llm_data)}")
            else:
                print("   ⚠️  No LLM metrics data to save")
            
            # 6. 保存备用候选人统计数据
            if hasattr(self, 'monthly_backup_stats') and self.monthly_backup_stats:
                backup_stats_path = os.path.join(data_dir, "backup_candidates_stats.json")
                try:
                    # 创建备份
                    self._backup_existing_file(backup_stats_path)
                    with open(backup_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(self.monthly_backup_stats, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Backup candidates stats saved: {backup_stats_path}")
                    print(f"   📊 Total backup stats records: {len(self.monthly_backup_stats)}")
                except Exception as e:
                    print(f"   ❌ Failed to save backup candidates stats: {e}")
            else:
                print("   ⚠️  No backup candidates stats data to save")

            # 7. 保存辞退统计数据
            if hasattr(self, 'monthly_dismissal_stats') and self.monthly_dismissal_stats:
                dismissal_stats_path = os.path.join(data_dir, "dismissal_stats.json")
                try:
                    # 创建备份
                    self._backup_existing_file(dismissal_stats_path)
                    with open(dismissal_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(self.monthly_dismissal_stats, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Dismissal stats saved: {dismissal_stats_path}")
                    print(f"   📊 Total dismissal stats records: {len(self.monthly_dismissal_stats)}")
                except Exception as e:
                    print(f"   ❌ Failed to save dismissal stats: {e}")
            else:
                print("   ⚠️  No dismissal stats data to save")
            
            # 8. 保存销售统计数据（行业竞争分析器使用）
            try:
                sales_stats_data = {}
                for month in range(1, self.config.num_iterations + 1):
                    try:
                        sales_stats = await self.economic_center.collect_sales_statistics.remote(month)
                        # 将销售统计数据转换为可序列化格式
                        serializable_sales = {}
                        for key, value in sales_stats.items():
                            if isinstance(key, tuple):
                                # key 是 (product_id, seller_id) 元组
                                serializable_key = f"{key[0]}_{key[1]}"
                            else:
                                serializable_key = str(key)
                            serializable_sales[serializable_key] = self._serialize_metric_for_json(value)
                        sales_stats_data[month] = serializable_sales
                    except Exception as e:
                        logger.warning(f"获取第 {month} 月销售统计数据失败: {e}")
                
                if sales_stats_data:
                    sales_stats_path = os.path.join(data_dir, "sales_statistics.json")
                    self._backup_existing_file(sales_stats_path)
                    with open(sales_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(sales_stats_data, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Sales statistics saved: {sales_stats_path}")
                    print(f"   📊 Total months: {len(sales_stats_data)}")
                else:
                    print("   ⚠️  No sales statistics data to save")
            except Exception as e:
                logger.warning(f"保存销售统计数据失败: {e}")
            
            # 10. 保存创新事件数据（创新导出器和行业竞争分析器使用）
            try:
                innovation_events = await self.economic_center.query_all_firm_innovation_events.remote()
                if innovation_events:
                    serializable_events = []
                    for event in innovation_events:
                        try:
                            if hasattr(event, 'model_dump'):
                                event_dict = event.model_dump()
                            elif hasattr(event, 'dict'):
                                event_dict = event.dict()
                            elif hasattr(event, '__dict__'):
                                event_dict = event.__dict__.copy()
                            else:
                                event_dict = {"company_id": getattr(event, 'company_id', None),
                                            "innovation_type": getattr(event, 'innovation_type', None),
                                            "month": getattr(event, 'month', None),
                                            "old_value": getattr(event, 'old_value', None),
                                            "new_value": getattr(event, 'new_value', None),
                                            "price_change": getattr(event, 'price_change', None),
                                            "attribute_change": getattr(event, 'attribute_change', None)}
                            serializable_events.append(event_dict)
                        except Exception as e:
                            logger.warning(f"序列化创新事件失败: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                    
                    innovation_events_path = os.path.join(data_dir, "innovation_events.json")
                    self._backup_existing_file(innovation_events_path)
                    with open(innovation_events_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_events, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Innovation events saved: {innovation_events_path}")
                    print(f"   📊 Total innovation events: {len(serializable_events)}")
                else:
                    print("   ⚠️  No innovation events data to save")
            except Exception as e:
                logger.error(f"保存创新事件数据失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 11. 保存创新配置数据（行业竞争分析器使用）
            try:
                innovation_configs_data = {}
                if self.firms:
                    for firm in self.firms:
                        try:
                            config = await self.economic_center.query_firm_innovation_config.remote(firm.company_id)
                            if config:
                                if hasattr(config, 'model_dump'):
                                    config_dict = config.model_dump()
                                elif hasattr(config, 'dict'):
                                    config_dict = config.dict()
                                elif hasattr(config, '__dict__'):
                                    config_dict = config.__dict__.copy()
                                else:
                                    config_dict = {
                                        "firm_id": getattr(config, 'firm_id', firm.company_id),
                                        "innovation_strategy": getattr(config, 'innovation_strategy', None),
                                        "labor_productivity_factor": getattr(config, 'labor_productivity_factor', None),
                                        "profit_margin": getattr(config, 'profit_margin', None),
                                        "fund_share": getattr(config, 'fund_share', None)
                                    }
                                innovation_configs_data[firm.company_id] = config_dict
                        except Exception as e:
                            logger.warning(f"获取企业 {firm.company_id} 创新配置失败: {e}")
                
                if innovation_configs_data:
                    innovation_configs_path = os.path.join(data_dir, "innovation_configs.json")
                    self._backup_existing_file(innovation_configs_path)
                    with open(innovation_configs_path, 'w', encoding='utf-8') as f:
                        json.dump(innovation_configs_data, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Innovation configs saved: {innovation_configs_path}")
                    print(f"   📊 Total firms: {len(innovation_configs_data)}")
                else:
                    print("   ⚠️  No innovation configs data to save")
            except Exception as e:
                logger.warning(f"保存创新配置数据失败: {e}")
            
            # 12. 保存生产统计数据（行业竞争分析器使用）
            if self.monthly_production_stats:
                production_stats_path = os.path.join(data_dir, "production_statistics.json")
                self._backup_existing_file(production_stats_path)
                try:
                    # 转换生产统计数据为可序列化格式
                    serializable_production_stats = {}
                    for month, stats in self.monthly_production_stats.items():
                        serializable_production_stats[month] = self._serialize_metric_for_json(stats)
                    
                    with open(production_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_production_stats, f, ensure_ascii=False, indent=2, default=str)
                    print(f"   ✅ Production statistics saved: {production_stats_path}")
                    print(f"   📊 Total months: {len(serializable_production_stats)}")
                except Exception as e:
                    logger.warning(f"保存生产统计数据失败: {e}")
            else:
                print("   ⚠️  No production statistics data to save")
            
            # 13. 生成数据摘要报告
            summary_path = os.path.join(data_dir, "data_summary.txt")
            self._backup_existing_file(summary_path)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"仿真数据摘要报告\n")
                f.write(f"================\n\n")
                f.write(f"实验名称: {self.experiment_name}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"数据统计:\n")
                f.write(f"- 经济指标历史: {len(self.economic_metrics_history) if self.economic_metrics_history else 0} 条记录\n")
                f.write(f"- 家庭月度指标: {sum(len(metrics) for metrics in self.household_monthly_metrics.values()) if self.household_monthly_metrics else 0} 条记录\n")
                f.write(f"- 企业月度指标: {len(self.firm_monthly_metrics) if self.firm_monthly_metrics else 0} 条记录\n")
                f.write(f"- 家庭购买记录: {sum(len(records) for records in self.household_purchase_records.values()) if self.household_purchase_records else 0} 条记录\n")
                f.write(f"- 性能监控指标: {len(self.performance_metrics) if self.performance_metrics else 0} 条记录\n")
                f.write(f"- LLM调用指标: {len(self.llm_metrics) if self.llm_metrics else 0} 条记录\n")
                f.write(f"- 备用候选人统计: {len(self.monthly_backup_stats) if hasattr(self, 'monthly_backup_stats') and self.monthly_backup_stats else 0} 条记录\n")
                f.write(f"- 辞退统计: {len(self.monthly_dismissal_stats) if hasattr(self, 'monthly_dismissal_stats') and self.monthly_dismissal_stats else 0} 条记录\n\n")
                
                if self.household_monthly_metrics:
                    f.write(f"月份覆盖范围: {min(self.household_monthly_metrics.keys())} - {max(self.household_monthly_metrics.keys())}\n")
                
                f.write(f"\n文件列表:\n")
                f.write(f"- economic_metrics_history.json: 经济指标历史数据\n")
                f.write(f"- household_monthly_metrics.json: 家庭月度指标数据\n")
                f.write(f"- firm_monthly_metrics.json: 企业月度指标数据\n")
                f.write(f"- household_purchase_records.json: 家庭购买记录数据\n")
                f.write(f"- sales_statistics.json: 销售统计数据（行业竞争分析器使用）\n")
                f.write(f"- production_statistics.json: 生产统计数据（行业竞争分析器使用）\n")
                f.write(f"- innovation_statistics.json: 创新统计数据（创新导出器使用）\n")
                f.write(f"- innovation_events.json: 创新事件数据\n")
                f.write(f"- innovation_configs.json: 创新配置数据\n")
                f.write(f"- performance_metrics.json: 性能监控数据\n")
                f.write(f"- llm_metrics.json: LLM调用指标数据\n")
                f.write(f"- backup_candidates_stats.json: 备用候选人统计数据\n")
                f.write(f"- dismissal_stats.json: 辞退统计数据\n")
                f.write(f"- data_summary.txt: 本摘要报告\n")
                f.write(f"\n其他输出目录:\n")
                f.write(f"- industry_competition/: 行业竞争分析报告和图表\n")
                f.write(f"- innovation_reports/: 创新数据报告\n")
            
            print(f"   ✅ Data summary report saved: {summary_path}")
            print(f"✅ All simulation data saved successfully to {data_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save simulation data to files: {e}")
            print(f"❌ Data saving failed: {e}")
    
    async def _generate_monthly_statistics_charts(self):
        """生成所有月度统计可视化图表和数据文件"""
        try:
            print("\n📊 开始生成月度统计可视化图表...")
            
            visualizer = MonthlyVisualization(self.experiment_name)
            
            # 1. 失业率趋势
            if self.monthly_unemployment_stats:
                visualizer.plot_unemployment_trend(self.monthly_unemployment_stats)
            
            # 2. 企业收入分布和商品购买率
            if self.monthly_firm_revenue:
                visualizer.plot_firm_revenue_distribution(self.monthly_firm_revenue)
                # 新增：企业全年利润分布图
                visualizer.plot_annual_firm_profit_distribution(self.monthly_firm_revenue)
            
            if self.monthly_product_sales and self.monthly_product_inventory:
                visualizer.plot_product_purchase_rate(self.monthly_product_sales, 
                                                      self.monthly_product_inventory)
            
            # 3. 商品库存变化
            if self.monthly_product_inventory:
                visualizer.plot_product_inventory_trend(self.monthly_product_inventory)
            
            # 4. 商品价格变化
            if self.monthly_product_prices:
                visualizer.plot_product_price_trend(self.monthly_product_prices)
            
            # 5. 购买量分布（销量 vs 补货）
            if self.monthly_product_sales:
                visualizer.plot_purchase_quantity_distribution(self.monthly_product_sales, self.monthly_product_inventory)
            
            # 6. 供需曲线
            if self.monthly_supply_demand:
                visualizer.plot_supply_demand_curve(self.monthly_supply_demand)
            
            # 7. 企业营业率
            if self.monthly_firm_operation_rate:
                visualizer.plot_firm_operation_rate(self.monthly_firm_operation_rate)
            
            # 8. 商品销量排名（长尾分布）
            if self.monthly_product_sales:
                visualizer.plot_product_sales_ranking(self.monthly_product_sales)
            
            # 8b. 商品销量排名（多月对比：1、4、7、10月）
            if self.monthly_product_sales:
                visualizer.plot_product_sales_ranking_multi_months(self.monthly_product_sales, [1, 4, 7, 10])
            
            print(f"✅ 所有月度统计图表已生成并保存到: {visualizer.charts_dir}")
            
            # 保存月度统计数据到文件
            await self._save_monthly_statistics_data()
            
        except Exception as e:
            logger.error(f"月度统计图表生成失败: {e}")
            print(f"❌ 月度统计图表生成失败: {e}")
    
    async def _save_monthly_statistics_data(self):
        """保存月度统计数据到JSON和TXT文件"""
        try:
            data_dir = os.path.join(self.experiment_output_dir, "monthly_statistics")
            os.makedirs(data_dir, exist_ok=True)
            
            print("\n📝 开始保存月度统计数据...")
            
            # 1. 保存失业统计数据
            if self.monthly_unemployment_stats:
                # JSON格式（详细数据）
                unemployment_json = f"{data_dir}unemployment_stats.json"
                with open(unemployment_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_unemployment_stats, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Unemployment stats saved: {unemployment_json}")
                
                # TXT格式（可读摘要）
                unemployment_txt = f"{data_dir}unemployment_summary.txt"
                with open(unemployment_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY UNEMPLOYMENT STATISTICS SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_unemployment_stats.keys()):
                        stats = self.monthly_unemployment_stats[month]
                        f.write(f"Month {month}:\n")
                        f.write(f"  - Total Unemployed: {stats['total_unemployed']}\n")
                        f.write(f"  - Unemployment Rate: {stats['unemployment_rate']*100:.2f}%\n")
                        f.write(f"  - Number of Unemployed Details: {len(stats.get('unemployed_details', []))}\n")
                        f.write("\n")
                print(f"   ✅ Unemployment summary saved: {unemployment_txt}")
            
            # 2. 保存空缺岗位数据
            if self.monthly_vacant_jobs:
                # JSON格式（详细数据）
                vacant_jobs_json = f"{data_dir}vacant_jobs.json"
                with open(vacant_jobs_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_vacant_jobs, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Vacant jobs data saved: {vacant_jobs_json}")
                
                # TXT格式（可读摘要）
                vacant_jobs_txt = f"{data_dir}vacant_jobs_summary.txt"
                with open(vacant_jobs_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY VACANT JOBS STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_vacant_jobs.keys()):
                        stats = self.monthly_vacant_jobs[month]
                        f.write(f"Month {month}:\n")
                        f.write(f"  - Total Vacant Jobs: {stats['total_vacant_jobs']}\n")
                        f.write(f"  - Top Job Titles:\n")
                        
                        # 统计职位出现频率
                        from collections import Counter
                        job_counter = Counter(stats.get('vacant_jobs_details', []))
                        for job_title, count in job_counter.most_common(10):
                            f.write(f"    • {job_title}: {count}\n")
                        f.write("\n")
                print(f"   ✅ Vacant jobs summary saved: {vacant_jobs_txt}")
            
            # 3. 保存企业收入数据
            if self.monthly_firm_revenue:
                # JSON格式
                revenue_json = f"{data_dir}firm_revenue.json"
                with open(revenue_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_firm_revenue, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Firm revenue data saved: {revenue_json}")
                
                # TXT格式（统计摘要）
                revenue_txt = f"{data_dir}firm_revenue_summary.txt"
                with open(revenue_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY FIRM REVENUE STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_firm_revenue.keys()):
                        firms = self.monthly_firm_revenue[month]
                        total_revenue = sum(f['revenue'] for f in firms.values())
                        total_profit = sum(f['profit'] for f in firms.values())
                        profitable_firms = sum(1 for f in firms.values() if f['profit'] > 0)
                        
                        f.write(f"Month {month}:\n")
                        f.write(f"  - Number of Firms: {len(firms)}\n")
                        f.write(f"  - Total Revenue: ${total_revenue:,.2f}\n")
                        f.write(f"  - Total Profit: ${total_profit:,.2f}\n")
                        f.write(f"  - Profitable Firms: {profitable_firms} ({profitable_firms/len(firms)*100:.1f}%)\n")
                        f.write(f"  - Average Revenue per Firm: ${total_revenue/len(firms):,.2f}\n")
                        f.write(f"  - Average Profit per Firm: ${total_profit/len(firms):,.2f}\n")
                        f.write("\n")
                print(f"   ✅ Firm revenue summary saved: {revenue_txt}")
            
            # 4. 保存商品销售数据
            if self.monthly_product_sales:
                # JSON格式
                sales_json = f"{data_dir}product_sales.json"
                with open(sales_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_product_sales, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Product sales data saved: {sales_json}")
                
                # TXT格式（统计摘要）
                sales_txt = f"{data_dir}product_sales_summary.txt"
                with open(sales_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY PRODUCT SALES STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_product_sales.keys()):
                        products = self.monthly_product_sales[month]
                        total_quantity = sum(p['total_quantity'] for p in products.values())
                        total_revenue = sum(p['total_revenue'] for p in products.values())
                        total_purchases = sum(p['purchase_count'] for p in products.values())
                        
                        f.write(f"Month {month}:\n")
                        f.write(f"  - Products Sold: {len(products)}\n")
                        f.write(f"  - Total Quantity Sold: {total_quantity:,.0f}\n")
                        f.write(f"  - Total Sales Revenue: ${total_revenue:,.2f}\n")
                        f.write(f"  - Total Purchase Transactions: {total_purchases}\n")
                        if len(products) > 0:
                            f.write(f"  - Average Quantity per Product: {total_quantity/len(products):.1f}\n")
                            f.write(f"  - Average Revenue per Product: ${total_revenue/len(products):,.2f}\n")
                        f.write("\n")
                print(f"   ✅ Product sales summary saved: {sales_txt}")
            
            # 5. 保存商品库存数据
            if self.monthly_product_inventory:
                inventory_json = f"{data_dir}product_inventory.json"
                with open(inventory_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_product_inventory, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Product inventory data saved: {inventory_json}")
            
            # 6. 保存商品价格数据
            if self.monthly_product_prices:
                prices_json = f"{data_dir}product_prices.json"
                with open(prices_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_product_prices, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Product prices data saved: {prices_json}")
                
                # TXT格式（价格趋势摘要）
                prices_txt = f"{data_dir}product_prices_summary.txt"
                with open(prices_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY PRODUCT PRICE STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_product_prices.keys()):
                        prices = self.monthly_product_prices[month]
                        price_values = [p['price'] for p in prices.values() if p['price'] > 0]
                        
                        if price_values:
                            f.write(f"Month {month}:\n")
                            f.write(f"  - Number of Products: {len(prices)}\n")
                            f.write(f"  - Average Price: ${np.mean(price_values):,.2f}\n")
                            f.write(f"  - Median Price: ${np.median(price_values):,.2f}\n")
                            f.write(f"  - Min Price: ${min(price_values):,.2f}\n")
                            f.write(f"  - Max Price: ${max(price_values):,.2f}\n")
                            f.write(f"  - Price Std Dev: ${np.std(price_values):,.2f}\n")
                            f.write("\n")
                print(f"   ✅ Product prices summary saved: {prices_txt}")
            
            # 7. 保存企业营业率数据
            if self.monthly_firm_operation_rate:
                operation_json = f"{data_dir}firm_operation_rate.json"
                with open(operation_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_firm_operation_rate, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Firm operation rate data saved: {operation_json}")
                
                # TXT格式（营业率摘要）
                operation_txt = f"{data_dir}firm_operation_rate_summary.txt"
                with open(operation_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY FIRM OPERATION RATE STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_firm_operation_rate.keys()):
                        firms = self.monthly_firm_operation_rate[month]
                        rates = [f['operation_rate'] for f in firms.values()]
                        
                        if rates:
                            f.write(f"Month {month}:\n")
                            f.write(f"  - Number of Firms: {len(firms)}\n")
                            f.write(f"  - Average Operation Rate: {np.mean(rates)*100:.2f}%\n")
                            f.write(f"  - Median Operation Rate: {np.median(rates)*100:.2f}%\n")
                            f.write(f"  - Firms with 100% Rate: {sum(1 for r in rates if r >= 1.0)}\n")
                            f.write(f"  - Firms with 0% Rate: {sum(1 for r in rates if r == 0.0)}\n")
                            f.write("\n")
                print(f"   ✅ Firm operation rate summary saved: {operation_txt}")
            
            # 8. 保存供需数据
            if self.monthly_supply_demand:
                supply_demand_json = f"{data_dir}supply_demand.json"
                with open(supply_demand_json, 'w', encoding='utf-8') as f:
                    json.dump(self.monthly_supply_demand, f, ensure_ascii=False, indent=2, default=str)
                print(f"   ✅ Supply-demand data saved: {supply_demand_json}")
                
                # TXT格式（供需摘要）
                supply_demand_txt = f"{data_dir}supply_demand_summary.txt"
                with open(supply_demand_txt, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTHLY SUPPLY-DEMAND STATISTICS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for month in sorted(self.monthly_supply_demand.keys()):
                        products = self.monthly_supply_demand[month]
                        total_supply = sum(p['supply'] for p in products.values())
                        total_demand = sum(p['demand'] for p in products.values())
                        
                        balanced = sum(1 for p in products.values() 
                                     if 0.8 <= p['supply_demand_ratio'] <= 1.2 
                                     and p['supply_demand_ratio'] != float('inf'))
                        oversupply = sum(1 for p in products.values() 
                                       if p['supply_demand_ratio'] > 1.2 
                                       and p['supply_demand_ratio'] != float('inf'))
                        undersupply = sum(1 for p in products.values() 
                                        if p['supply_demand_ratio'] < 0.8)
                        
                        f.write(f"Month {month}:\n")
                        f.write(f"  - Number of Products: {len(products)}\n")
                        f.write(f"  - Total Supply: {total_supply:,.0f}\n")
                        f.write(f"  - Total Demand: {total_demand:,.0f}\n")
                        f.write(f"  - Overall Supply/Demand Ratio: {total_supply/total_demand if total_demand > 0 else float('inf'):.2f}\n")
                        f.write(f"  - Balanced Products (0.8-1.2): {balanced} ({balanced/len(products)*100:.1f}%)\n")
                        f.write(f"  - Oversupply Products (>1.2): {oversupply} ({oversupply/len(products)*100:.1f}%)\n")
                        f.write(f"  - Undersupply Products (<0.8): {undersupply} ({undersupply/len(products)*100:.1f}%)\n")
                        f.write("\n")
                print(f"   ✅ Supply-demand summary saved: {supply_demand_txt}")
            
            print(f"\n✅ 所有月度统计数据已保存到: {data_dir}")
            
        except Exception as e:
            logger.error(f"保存月度统计数据失败: {e}")
            print(f"❌ 保存月度统计数据失败: {e}")
    
    async def _generate_monthly_consumption_chart(self, charts_dir: str):
        """生成每月各类消费品总消费柱状图"""
        try:
            print("📊 Generating monthly consumption chart...")
            
            # 收集每月消费数据和所有消费类别
            monthly_data = {}
            all_categories = set()
            
            # 先遍历一遍收集所有的消费类别
            for metric in self.household_monthly_metrics[1]:
                for category in metric.consumption_structure.keys():
                    all_categories.add(category)
            
            if not all_categories:
                print("⚠️  No consumption category data, skipping quarterly consumption chart")
                return
            
            # 按类别名称排序
            consumption_categories = sorted(list(all_categories))
            
            # 收集阅读数据
            for i in range(1,self.config.num_iterations+1):
                for metric in self.household_monthly_metrics[i]:
                    if metric.month not in monthly_data:
                        monthly_data[metric.month] = {cat: 0 for cat in consumption_categories}
                
                for category, amount in metric.consumption_structure.items():
                    if category in monthly_data[metric.month]:
                        monthly_data[metric.month][category] += amount
            
            
            # 创建柱状图
            months = sorted(monthly_data.keys())
            categories = consumption_categories
            
            fig, ax = plt.subplots(figsize=(14, 8))  # 稍微增加宽度以适应更多类别
            
            # 设置柱状图位置
            x = np.arange(len(months))
            width = 0.8 / len(categories)  # 动态调整宽度
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            for i, category in enumerate(categories):
                values = [monthly_data[q].get(category, 0) for q in months]
                ax.bar(x + i * width - width * (len(categories) - 1) / 2, values, width, label=category, color=colors[i])
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Consumption Amount ($)')
            ax.set_title('Quarterly Consumption by Category')
            ax.set_xticks(x)
            ax.set_xticklabels(months)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(charts_dir, "monthly_consumption_by_category.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Monthly consumption chart saved: {chart_path}")
            print(f"   📋 Consumption categories: {', '.join(categories)}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate quarterly consumption chart: {e}")

async def main():
    """主函数"""
    # 创建仿真配置
    config = SimulationConfig()
    # print(os.getenv("DEEPSEEK_API_KEY", "1232"))
    # 创建仿真器
    simulation = EconomicSimulation(config)
    
    try:
        # 设置仿真环境
        if not await simulation.setup_simulation_environment():
            logger.error("仿真环境设置失败")
            return
        
        # 运行仿真
        await simulation.run_simulation()
        
        # 生成并保存报告
        report = await simulation.generate_simulation_report()
        await simulation.save_simulation_report(report)
        
        logger.info("经济仿真完成")
        
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在停止仿真...")
    except Exception as e:
        logger.error(f"经济仿真执行失败: {e}")
        raise
    finally:
        # 清理资源
        await simulation.cleanup_resources()

if __name__ == "__main__":
    # 运行经济仿真
    asyncio.run(main())
