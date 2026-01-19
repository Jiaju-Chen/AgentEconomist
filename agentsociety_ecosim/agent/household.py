import asyncio
import time
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from agentsociety_ecosim.llm.llm import LLM,  ChatCompletionMessageParam
from typing import  List, Any, Optional, Dict, Tuple
from agentsociety_ecosim.center.model import LaborHour, Product, PurchaseRecord, JobApplication
from agentsociety_ecosim.center.ecocenter import EconomicCenter
from agentsociety_ecosim.center.assetmarket import ProductMarket
from agentsociety_ecosim.center.jobmarket import Job, LaborMarket
from agentsociety_ecosim.agent.firm import Firm
import uuid
import json
import os
from agentsociety_ecosim.logger import get_logger, set_logger_level
# from agentsociety_ecosim.consumer_modeling import llm_utils
from agentsociety_ecosim.utils.log_utils import setup_global_logger
from agentsociety_ecosim.utils.product_attribute_loader import inject_product_attributes
import tiktoken
# 导入高级消费模块
from agentsociety_ecosim.utils.data_loader import match_pro_firm
from agentsociety_ecosim.consumer_modeling.consumer_decision import BudgetAllocator
from agentsociety_ecosim.consumer_modeling.family_attribute_manager import FamilyAttributeSystem
# from ..consumer_modeling.family_data import get_family_consumption_and_profile_by_id
ADVANCED_CONSUMPTION_AVAILABLE = True
from agentsociety_ecosim.consumer_modeling.attribute_benchmark import AttributeBenchmarkManager
from openai import AsyncOpenAI
# 使用环境变量获取API key - 改为异步客户端实现真正并发
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url=os.getenv("BASE_URL", ""),
    timeout=60.0  # 设置60秒超时
)
# 导入PSID数据加载功能
import random
logger = setup_global_logger(__name__)

def calculate_tokens_household(text: str) -> int:
    """计算文本的token数量 - household版本"""
    try:
        encoding = tiktoken.encoding_for_model('gpt-4')
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token计算失败: {e}")
        return int(len(text.split()) * 1.3)  # 粗略估算

# @ray.remote
class Household:  #TODO: 加入初始化资金、产品等
    def __init__(
        self,
        household_id: Optional[str] = None,
        labor_hour: Optional[List[LaborHour]] = None, # Type hint for list of LaborHour objects
        income_this_period: float = 0.0,
        hours_worked_this_period: float = 0.0,
        llm: Optional[LLM] = None, # Can be provided or will be created by default
        economic_center: Optional[EconomicCenter] = None, # Expects explicit provision, can be None initially
        # ===== 新增：高级消费模式参数 =====
        consumption_mode: str = "advanced",  # "simple" 或 "advanced"
        family_profile: Optional[Dict] = None,  # 家庭画像信息
        # ===== 新增：PSID数据初始化参数 =====
        use_psid_data: bool = True,  # 默认使用PSID数据进行初始化
        psid_family_id: Optional[str] = None,  # 指定使用的PSID家庭ID
        initial_wealth: Optional[float] = None,  # 手动指定初始财富，覆盖PSID数据
        labormarket: Optional[LaborMarket] = None,
        product_market:ProductMarket = None,  # Job market reference
        # ===== 新增：税率参数 =====
        income_tax_rate: float = 0.225,  # 个人所得税率，默认22.5%
        vat_rate: float = 0.08  # 消费税率（增值税），默认8%
    ):
        # ===== 原有属性初始化 =====
        self.household_id: str = household_id if household_id is not None else str(uuid.uuid4())
        self.labor_hours: List[LaborHour] = labor_hour if labor_hour is not None else []
        self.income_this_period: float = income_this_period
        self.hours_worked_this_period: float = hours_worked_this_period
        self.labormarket = labormarket
        self.product_market = product_market
        # ===== 新增：PSID数据初始化处理 =====
        self.use_psid_data: bool = use_psid_data
        self.psid_family_id: Optional[str] = household_id
        
        # 确定要使用的PSID家庭ID
        target_psid_id = psid_family_id if psid_family_id else self.household_id
        
        self.llm: LLM = llm 
        self.economic_center: Optional[EconomicCenter] = economic_center
        self.purchase_history: List[PurchaseRecord] = []  # Track purchase history
        
        # ===== 新增：税率配置 =====
        self.income_tax_rate: float = income_tax_rate  # 个人所得税率
        self.vat_rate: float = vat_rate  # 消费税率
        
        # ===== 新增：高级消费模式相关属性 =====
        self.consumption_mode: str = consumption_mode

        # 新增数据处理工作匹配 - 使用JobApplication
        self.head_job_applications: List[JobApplication] = []
        self.spouse_job_applications: List[JobApplication] = []
        
        self.head_job:Job = None
        self.spouse_job:Job = None
        
        # 月度工作追踪变量：记录每个劳动力每个月的工作情况
        # 结构: {month: {'head': job_info, 'spouse': job_info}}
        # job_info: {'company_id': str, 'job_title': str, 'job_SOC': str, 'wage': float, 'employed': bool}
        self.monthly_job_tracking: Dict[int, Dict[str, Dict[str, Any]]] = {}
        
        # 用于保存budget和消费信息
        self.consume_budget:Dict[int, Dict] = {}
        
        # ===== 家庭属性系统 (新版) =====
        self.attribute_system: Optional[FamilyAttributeSystem] = None  # 属性系统实例
        self.attribute_initialized: bool = False  # 属性系统是否已初始化

        
        
        # 处理家庭画像：优先使用PSID数据，其次使用传入的family_profile
        if use_psid_data:
            # 优先使用PSID数据生成家庭画像
            psid_profile = self.get_family_profile_from_psid(self.psid_family_id)
            
            # 如果提供了family_profile，将其与PSID数据合并（PSID数据优先）
            if family_profile is not None:
                # 合并画像数据，PSID数据优先，但允许family_profile补充缺失字段
                merged_profile = family_profile.copy()
                merged_profile.update(psid_profile)  # PSID数据覆盖同名字段
                self.family_profile = merged_profile
                # logger.info(f"Loaded family profile from PSID data with manual overrides: {self.family_profile}")
            else:
                self.family_profile = psid_profile
                # logger.info(f"Loaded family profile from PSID data: {self.family_profile}")
        elif family_profile is not None:
            # 如果不使用PSID数据，使用传入的family_profile
            self.family_profile: Optional[Dict] = family_profile
        else:
            # 都没有提供时为None
            self.family_profile = None
        
        # 处理初始财富：优先使用手动指定值，否则从PSID数据获取2021年消费支出
        if initial_wealth is not None:
            self.initial_wealth: float = initial_wealth
        elif use_psid_data:
            self.initial_wealth = self.get_initial_wealth_from_psid_2021_expenditure(target_psid_id)
            # logger.info(f"Loaded initial wealth from PSID 2021 expenditure data: ${self.initial_wealth:.2f}")
        else:
            self.initial_wealth = 50000.0  # 默认值
        
        # 其他高级消费相关属性
        self.budget_allocator: Optional[BudgetAllocator] = None  # 延迟初始化
        self.annual_plan: Optional[Dict] = None  # 年度消费计划
        self.current_month: int = 1  # 当前月份
        # ========================================
        # 🔧 新增：保存上月预算供下月LLM决策使用
        # ========================================
        self.last_month_budget: Optional[float] = None  # 上月预算
    # Represents an individual economic household in the simulation.
    # This class manages its internal state (finances, labor potential, jobs)
    # and interacts with various external economic entities (markets, firms,)
    # to perform core economic activities like consumption and work.
    # These dependencies should be ActorHandles or Ray remote objects
    # product_market: ProductMarket # Not needed as a direct attribute if passed to consume
    # labor_market: LaborMarket # Not needed as a direct attribute if passed to work
    # firms: List[Firm] # Not needed as a direct attribute if passed to work
    # it's usually better to pass their handles directly.
    # If initial deposit is needed, it should be done externally, e.g., in the simulation setup
    # def __init__(self, **data: Any):
    #    super().__init__(**data)
    #    # Removed direct init to avoid Pydantic conflict with Ray actors.
    #    # Initializing ledger should be done by the simulation manager via deposit_funds.
    async def initialize(self):
        """
        Initializes the household agent, setting up its initial state.
        This method can be used to set up initial balances, labor potential, etc.
        """
        try:
            if self.economic_center: 
                # 使用初始财富进行初始化（来自PSID数据或手动指定）
                initial_balance = self.initial_wealth if self.initial_wealth > 0 else 0.0
                
                await asyncio.gather(
                    self.economic_center.init_agent_ledger.remote(self.household_id, initial_balance),
                    self.economic_center.init_agent_product.remote(self.household_id),
                    self.economic_center.init_agent_labor.remote(self.household_id, self.labor_hours),
                    self.economic_center.register_id.remote(self.household_id, 'household')
                )
                
                if self.use_psid_data and initial_balance > 0:
                    # logger.info(f"Household {self.household_id} initialized with PSID-based wealth: ${initial_balance:.2f}")
                    if self.family_profile and self.family_profile.get('psid_family_id'):
                        pass
                        # logger.info(f"Using PSID family ID: {self.family_profile['psid_family_id']}")
                else:
                    pass
                    # logger.info(f"Household {self.household_id} registered in EconomicCenter with balance: ${initial_balance:.2f}")
        except Exception as e:
            logger.warning(f"Household {self.household_id} failed to register: {e}")

    def get_balance_ref(self): 
        """Returns household's current balance."""
        current_balance_ref = self.economic_center.query_balance.remote(self.household_id)
        return current_balance_ref
    
    def _enrich_product_kwargs(self, product_kwargs: Dict[str, Any], source_product: Optional[Product] = None) -> Dict[str, Any]:
        """
        Attach attribute information to product kwargs from an existing Product instance
        or the global attribute mapping.
        """
        enriched = dict(product_kwargs)
        product_id = enriched.get("product_id")

        if source_product:
            enriched.setdefault("attributes", getattr(source_product, "attributes", None))
            enriched.setdefault("is_food", getattr(source_product, "is_food", None))
            enriched.setdefault("nutrition_supply", getattr(source_product, "nutrition_supply", None))
            enriched.setdefault("satisfaction_attributes", getattr(source_product, "satisfaction_attributes", None))
            enriched.setdefault("duration_months", getattr(source_product, "duration_months", None))

        return inject_product_attributes(enriched, product_id)
    

    def set_current_month(self, month: int):
        """
        Sets the current month for the household.
        This can be used to manage monthly budgets and consumption plans.
        """
        if 1 <= month <= 12:
            self.current_month = month
            # logger.info(f"Household {self.household_id} set current month to {self.current_month}.")
        else:
            logger.warning(f"Invalid month {month} for Household {self.household_id}. Must be between 1 and 12.")

    def get_owned_products_from_ec_ref(self) -> Any: # Returns ObjectRef[List[Product]]
        """Returns a Ray ObjectRef to the list of products owned by the household."""
        return self.economic_center.query_products.remote(self.household_id)


    async def find_jobs(self):
        """
        为家庭寻找工作，每月清空之前的申请记录
        """
        # 清空上个月的申请记录，开始新一轮的工作搜索
        self.head_job_applications = []
        self.spouse_job_applications = []
        
        if self.labor_hours:
            for labor_hour in self.labor_hours:
                if labor_hour.is_valid:
                    lh_type = labor_hour.lh_type
                    matched_job_list = await self.labormarket.match_jobs.remote(labor_hour)
                    if matched_job_list:
                        # 为匹配的工作生成期望薪资并创建JobApplication
                        job_applications = await self.create_job_applications(matched_job_list, labor_hour)
                        # 保存工作申请
                        self.save_job_applications(job_applications, lh_type)
                        # 提交工作申请到劳动力市场
                        await self.submit_job_applications_to_market(job_applications, self.current_month)
        return self.head_job_applications, self.spouse_job_applications
    
    def save_job_applications(self, job_applications: List[JobApplication], lh_type: str):
        """
        保存工作申请列表
        """
        if lh_type == 'head':
            self.head_job_applications.extend(job_applications)
        elif lh_type == 'spouse':
            self.spouse_job_applications.extend(job_applications)
        else:
            logger.warning(f"Invalid lh_type: {lh_type}")
    
    async def create_job_applications(self, matched_job_list: List[Job], labor_hour: LaborHour) -> List[JobApplication]:
        """
        为匹配的工作创建JobApplication对象
        优化：并发处理所有匹配工作的LLM调用
        """
        async def process_single_job(job):
            """处理单个工作的LLM期望薪资生成"""
            try:
                # 生成期望薪资
                expectation_result = await self.llm_generate_wage_expectation(
                    job=job,
                    worker_skills=labor_hour.skill_profile,
                    worker_abilities=labor_hour.ability_profile,
                    family_context=self.get_family_context_for_wage_expectation(),
                    labor_hour_type=labor_hour.lh_type
                )
                
                # 创建JobApplication
                application = JobApplication.create(
                    job_id=job.job_id,
                    household_id=self.household_id,
                    lh_type=labor_hour.lh_type,
                    expected_wage=expectation_result.get("expected_wage", job.wage_per_hour),
                    worker_skills=labor_hour.skill_profile,
                    worker_abilities=labor_hour.ability_profile,
                    month=self.current_month
                )
                
                return application
                
            except Exception as e:
                logger.warning(f"Failed to create job application for job {job.title}: {e}")
                # 创建默认JobApplication
                application = JobApplication.create(
                    job_id=job.job_id,
                    household_id=self.household_id,
                    lh_type=labor_hour.lh_type,
                    expected_wage=job.wage_per_hour * 1.1,  # 期望比职位薪资高10%
                    worker_skills=labor_hour.skill_profile,
                    worker_abilities=labor_hour.ability_profile,
                    month=self.current_month
                )
                return application
        
        # 并发执行所有LLM调用
        if matched_job_list:
            job_tasks = [process_single_job(job) for job in matched_job_list]
            job_applications = await asyncio.gather(*job_tasks, return_exceptions=True)
            
            # 过滤掉异常结果，只保留成功的申请
            valid_applications = []
            for app in job_applications:
                if not isinstance(app, Exception) and app is not None:
                    valid_applications.append(app)
                else:
                    logger.warning(f"Job application creation failed: {app}")
            
            return valid_applications
        else:
            return []
    
    async def generate_job_expectations(self, matched_job_list: List[Job], labor_hour: LaborHour) -> List[Dict]:
        """
        使用LLM为匹配的工作生成期望薪资
        
        Args:
            matched_job_list: 匹配的工作列表
            labor_hour: 劳动力小时对象，包含技能和能力信息
        
        Returns:
            List[Dict]: 包含工作和期望薪资的字典列表
            格式: [{"job": Job, "expected_wage": float, "reasoning": str, "confidence": float}]
        """
        jobs_with_expectations = []
        
        # 获取家庭基本信息用于期望薪资计算
        family_info = self.get_family_context_for_wage_expectation()
        
        for job in matched_job_list:
            try:
                # 为每个工作生成期望薪资
                expectation_result = await self.llm_generate_wage_expectation(
                    job=job,
                    worker_skills=labor_hour.skill_profile,
                    worker_abilities=labor_hour.ability_profile,
                    family_context=family_info,
                    labor_hour_type=labor_hour.lh_type
                )
                
                jobs_with_expectations.append({
                    "job": job,
                    "expected_wage": expectation_result.get("expected_wage", job.wage_per_hour),
                    "reasoning": expectation_result.get("reasoning", "Default wage expectation"),
                    "confidence": expectation_result.get("confidence", 0.5),
                    "key_factors": expectation_result.get("key_factors", [])
                })
                
            except Exception as e:
                logger.warning(f"Failed to generate wage expectation for job {job.title}: {e}")
                # 如果LLM调用失败，使用默认期望薪资（略高于职位薪资）
                jobs_with_expectations.append({
                    "job": job,
                    "expected_wage": job.wage_per_hour * 1.1,  # 期望比职位薪资高10%
                    "reasoning": f"LLM调用失败，使用默认期望薪资: {str(e)}",
                    "confidence": 0.3,
                    "key_factors": ["default_calculation"]
                })
        
        return jobs_with_expectations
    
    def get_family_context_for_wage_expectation(self) -> Dict:
        """
        获取家庭上下文信息，用于期望薪资计算
        """
        context = {
            "household_id": self.household_id,
            "family_size": self.family_profile.get("family_size", 3) if self.family_profile else 3,
            "current_balance": 0.0,  # 将在调用时异步获取
            "monthly_expenses": 0.0,  # 基于历史消费记录估算
            "current_income": self.income_this_period,
            "has_spouse": any(lh.lh_type == 'spouse' and lh.is_valid for lh in self.labor_hours),  # 是否有有效的配偶
            "num_children": self.family_profile.get("num_children", 0) if self.family_profile else 0,
            "head_age": self.family_profile.get("head_age", 40) if self.family_profile else 40,
            "location_state": self.family_profile.get("state_code", 0) if self.family_profile else 0
        }
        
        # 计算月平均支出
        if self.purchase_history:
            total_spent = sum(record.total_spent for record in self.purchase_history)
            months_with_spending = len(set(record.month for record in self.purchase_history))
            context["monthly_expenses"] = total_spent / max(months_with_spending, 1)
        
        return context
    
    async def llm_generate_wage_expectation(
        self, 
        job: Job, 
        worker_skills: Dict, 
        worker_abilities: Dict, 
        family_context: Dict,
        labor_hour_type: str
    ) -> Dict:
        """
        使用LLM生成对特定工作的期望薪资
        
        Args:
            job: 工作对象
            worker_skills: 工人技能档案
            worker_abilities: 工人能力档案  
            family_context: 家庭上下文信息
            labor_hour_type: 劳动力类型 ('head' 或 'spouse')
        
        Returns:
            Dict: 包含期望薪资、推理过程等信息
        """
        # 获取当前余额
        try:
            current_balance = await self.economic_center.query_balance.remote(self.household_id)
            family_context["current_balance"] = current_balance
        except:
            family_context["current_balance"] = 0.0
        
        # print(f"    🧠 家庭 {self.household_id} ({labor_hour_type}) 正在为职位 '{job.title}' 生成期望薪资...")
        # print(f"        职位薪资: ${job.wage_per_hour:.2f}/小时")
        # print(f"        家庭余额: ${family_context['current_balance']:.2f}")
        
        # 构建LLM提示
        prompt = self.build_wage_expectation_prompt(
            job, worker_skills, worker_abilities, family_context, labor_hour_type
        )
        
        # 计算并打印token数量
        # prompt_tokens = calculate_tokens_household(prompt)
        # print(f"👨‍👩‍👧‍👦 [薪资期望] Prompt Token数量: {prompt_tokens} (家庭: {self.household_id}, 角色: {labor_hour_type})")
        
        try:
            # response = await self.llm.atext_request(messages)
            response = await client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.8
            )
            # 解析LLM响应
            response_content = response.choices[0].message.content.strip()
            
            # 尝试清理响应内容，提取JSON部分
            if response_content.startswith("```json"):
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_content = response_content[start_idx:end_idx]
            elif response_content.startswith("```"):
                lines = response_content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') or in_json:
                        in_json = True
                        json_lines.append(line)
                        if line.strip().endswith('}') and json_lines:
                            break
                response_content = '\n'.join(json_lines)
            
            scores = json.loads(response_content)
            
            # 使用新的评分计算期望薪资
            wage_result = self.calculate_expected_wage_from_scores(job, scores, labor_hour_type)
            
            # 确保期望薪资在合理范围内（职位薪资的0.8-2.0倍）
            min_wage = job.wage_per_hour * 0.8
            max_wage = job.wage_per_hour * 2.0
            original_expected = wage_result["expected_wage"]
            expected_wage = max(min_wage, min(original_expected, max_wage))
            
            # # 显示LLM决策结果
            # print(f"        📊 LLM评分:")
            # print(f"           技能匹配: {scores.get('skill_match', 0.5):.2f}")
            # print(f"           预算压力: {scores.get('budget_pressure', 0.3):.2f}")
            # print(f"           角色优先级: {scores.get('role_priority', 0.5):.2f}")
            # print(f"        🎯 期望薪资: ${job.wage_per_hour:.2f}/小时 → ${original_expected:.2f}/小时 (LLM调整后)")
            # if original_expected != expected_wage:
            #     print(f"        🎯 期望薪资: ${original_expected:.2f}/小时 → ${expected_wage:.2f}/小时 (范围调整后)")
            # print(f"        📊 薪资倍数: {wage_result['calc_details']['multiplier']:.3f}")
            # print(f"        📊 信心度: {wage_result['confidence']:.1%}")
            # print(f"        💭 推理: {scores.get('brief_rationale', '基于评分计算')[:60]}...")
            
            # if original_expected != expected_wage:
            #     print(f"        ⚠️  期望薪资已调整到合理范围 (${min_wage:.2f} - ${max_wage:.2f})")
            
            # 更新最终结果
            wage_result["expected_wage"] = expected_wage
            wage_result["expected_total_period_pay"] = round(expected_wage * job.hours_per_period, 2)
            
            return wage_result
            
        except json.JSONDecodeError as e:
            print(f"        ❌ LLM响应JSON解析失败: {e}")
            logger.warning(f"Failed to parse LLM response for wage expectation: {e}")
            # 使用默认评分计算期望薪资
            import random
            default_scores = {
                "skill_match": random.uniform(0.4, 0.7),
                "budget_pressure": random.uniform(0.3, 0.6),
                "role_priority": 1.0 if labor_hour_type == "head" else 0.5,
                "brief_rationale": f"JSON解析失败，使用默认评分: {str(e)}"
            }
            wage_result = self.calculate_expected_wage_from_scores(job, default_scores, labor_hour_type)
            print(f"        🔄 使用默认评分计算期望薪资: ${wage_result['expected_wage']:.2f}/小时")
            return wage_result
        except Exception as e:
            print(f"        ❌ LLM调用失败: {e}")
            logger.warning(f"LLM call failed for wage expectation: {e}")
            # 使用默认评分计算期望薪资
            import random
            default_scores = {
                "skill_match": random.uniform(0.4, 0.7),
                "budget_pressure": random.uniform(0.3, 0.6),
                "role_priority": 1.0 if labor_hour_type == "head" else 0.5,
                "brief_rationale": f"LLM调用失败，使用默认评分: {str(e)}"
            }
            wage_result = self.calculate_expected_wage_from_scores(job, default_scores, labor_hour_type)
            print(f"        🔄 使用默认评分计算期望薪资: ${wage_result['expected_wage']:.2f}/小时")
            return wage_result
    
    def build_wage_expectation_prompt_old(
        self, 
        job: Job, 
        worker_skills: Dict, 
        worker_abilities: Dict, 
        family_context: Dict,
        labor_hour_type: str
    ) -> str:
        """
        构建用于生成期望薪资的LLM提示
        """
        role_type = "household head" if labor_hour_type == "head" else "spouse"
        
        prompt = f"""You are an expert career counselor helping a {role_type} determine their wage expectations for a job opportunity.

**Job Information:**
- Title: {job.title}
- Description: {job.description}
- Posted Wage: ${job.wage_per_hour:.2f}/hour
- Hours per Period: {job.hours_per_period}
- Company ID: {job.company_id}
- Required Skills: {json.dumps(job.required_skills, indent=2)}
- Required Abilities: {json.dumps(job.required_abilities, indent=2)}

**Worker Profile:**
- Role: {role_type}
- Skills: {json.dumps(worker_skills, indent=2)}
- Abilities: {json.dumps(worker_abilities, indent=2)}

**Family Context:**
- Family Size: {family_context['family_size']} people
- Number of Children: {family_context['num_children']}
- Has Spouse: {'Yes' if family_context['has_spouse'] else 'No'}
- Current Balance: ${family_context['current_balance']:.2f}
- Monthly Expenses: ${family_context['monthly_expenses']:.2f}
- Current Income: ${family_context['current_income']:.2f}
- Head Age: {family_context['head_age']} years

**Task:**
Based on the worker's skills/abilities match with job requirements, family financial needs, and market considerations, determine a reasonable wage expectation.

**Consider these factors:**
1. **Skill Match**: How well do the worker's skills align with job requirements?
2. **Family Needs**: Monthly expenses, number of dependents, current financial situation
3. **Market Position**: How does the posted wage compare to typical market rates?
4. **Negotiation Power**: Worker's leverage based on skill match and family circumstances
5. **Role Priority**: Is this the primary earner (head) or secondary earner (spouse)?

**Response Format (JSON only):**
{{
    "expected_wage": ajusted_wage,
    "reasoning": "Detailed explanation of wage expectation rationale",
    "confidence": 0.8,
    "key_factors": ["skill_match", "family_needs", "market_rate"],
    "negotiation_flexibility": "high|medium|low"
}}

**Guidelines:**
- Expected wage should be reasonable (0.8x to 2.0x the posted wage)
- Higher expectations if skills exceed requirements and family has high expenses
- Lower expectations if desperate for income or skills don't fully match
- Consider the role's importance to family income (head vs spouse)
"""
        
        return prompt
    def build_wage_expectation_prompt(
        self, 
        job: Job, 
        worker_skills: Dict, 
        worker_abilities: Dict, 
        family_context: Dict,
        labor_hour_type: str
    ) -> str:
        """
        构建用于生成期望薪资评分的LLM提示
        """
        role_type = "head" if labor_hour_type == "head" else "spouse"
        
        # 压缩技能要求展示
        def compress_skills(skills_dict, max_items=5):
            if not skills_dict:
                return "N/A"
            items = list(skills_dict.items())[:max_items]
            return ", ".join([f"{k}:{v.get('mean', v) if isinstance(v, dict) else v}" for k, v in items])
        
        # 压缩工人技能展示 - 保留所有技能但用紧凑格式
        def compress_worker_skills(skills_dict):
            if not skills_dict:
                return "N/A"
            return ", ".join([f"{k}:{v}" for k, v in skills_dict.items()])
        
        # 简化工作描述（保留前80个字符）
        job_desc = job.description[:80] + "..." if len(job.description) > 80 else job.description
        
        prompt = f"""=== Wage Expectation Analysis ===
Position: {job.title} | Posted: ${job.wage_per_hour:.2f}/h | Hours: {job.hours_per_period or 40}h/period
Required Skills: {compress_skills(job.required_skills)}
Required Abilities: {compress_skills(job.required_abilities)}

=== Worker Profile ({role_type.title()}) ===
Skills: {compress_worker_skills(worker_skills)}
Abilities: {compress_worker_skills(worker_abilities)}

=== Family Context ===
Size: {family_context.get('family_size', 'N/A')} | Children: {family_context.get('num_children', 0)} | Age: {family_context.get('head_age', 'N/A')}
Balance: ${family_context.get('current_balance', 0):.0f} | Monthly Expenses: ${family_context.get('monthly_expenses', 0):.0f}
Current Income: ${family_context.get('current_income', 0):.0f}

=== Task ===
Analyze worker-job-family fit. Return JSON with scores [0-1] and brief rationale.

=== Response Format ===
{{
    "skill_match": 0.0-1.0,
    "budget_pressure": 0.0-1.0,
    "role_priority": 0.0-1.0,
    "brief_rationale": "max 40 words"
}}"""
        
        return prompt
    
    def calculate_expected_wage_from_scores(
        self, 
        job: Job, 
        scores: Dict, 
        labor_hour_type: str
    ) -> Dict:
        """
        根据LLM评分计算期望薪资
        
        Args:
            job: 职位信息
            scores: LLM返回的评分字典，包含skill_match, budget_pressure, role_priority
            labor_hour_type: 劳动力类型 ("head" 或 "spouse")
            
        Returns:
            包含期望薪资和相关信息的字典
        """
        # 获取评分
        skill_match = float(scores.get("skill_match", 0.5))
        budget_pressure = float(scores.get("budget_pressure", 0.3))
        role_priority = float(scores.get("role_priority", 0.5))
        brief_rationale = scores.get("brief_rationale", "基于技能匹配、预算压力和角色优先级计算")
        
        # 确保评分在合理范围内
        skill_match = max(0.0, min(1.0, skill_match))
        budget_pressure = max(0.0, min(1.0, budget_pressure))
        role_priority = max(0.0, min(1.0, role_priority))
        
        # 计算薪资倍数
        # 基础倍数 + 技能匹配影响 + 预算压力影响 + 角色优先级影响
        # 调整系数让期望薪资更合理：高匹配度约1.5倍，其他情况±10%左右
        multiplier = 1.0 + 0.25 * skill_match + 0.1 * budget_pressure + 0.05 * role_priority
        
        # 限制倍数在合理范围内
        multiplier = max(0.80, min(2.00, multiplier))
        
        # 计算期望薪资
        expected_wage = round(job.wage_per_hour * multiplier, 2)
        
        # 计算信心度（基于评分的一致性）
        confidence = (skill_match + budget_pressure + role_priority) / 3.0
        
        # 确定谈判灵活性
        if skill_match > 0.7 and budget_pressure < 0.4:
            negotiation_flexibility = "high"
        elif skill_match > 0.5 or budget_pressure > 0.6:
            negotiation_flexibility = "medium"
        else:
            negotiation_flexibility = "low"
        
        # 确定关键因素
        key_factors = []
        if skill_match > 0.6:
            key_factors.append("skill_match")
        if budget_pressure > 0.5:
            key_factors.append("family_needs")
        if role_priority > 0.7:
            key_factors.append("role_priority")
        if not key_factors:
            key_factors.append("market_anchor")
        
        return {
            "expected_wage": expected_wage,
            "expected_total_period_pay": round(expected_wage * job.hours_per_period, 2),
            "confidence": confidence,
            "negotiation_flexibility": negotiation_flexibility,
            "key_factors": key_factors,
            "brief_rationale": brief_rationale,
            "calc_details": {
                "posted_wage": job.wage_per_hour,
                "multiplier": multiplier,
                "skill_match": skill_match,
                "budget_pressure": budget_pressure,
                "role_priority": role_priority,
                "near_posted_reason": "" if abs(expected_wage - job.wage_per_hour) / job.wage_per_hour > 0.03 else "接近发布薪资"
            }
        }
    
    async def evaluate_job_offers(self, job_offers: List[Dict], std_jobs=None) -> List[Dict]:
        """
        评估收到的job offers并决定接受哪些
        
        Args:
            job_offers: 收到的job offers列表
            std_jobs: 标准工作数据，用于市场薪资比较
            
        Returns:
            List[Dict]: 决定接受的offers列表
        """
        if not job_offers:
            return []
        
        print(f"🤔 家庭 {self.household_id} 开始评估 {len(job_offers)} 个工作offer...")
        
        # 检查劳动力是否已经有工作，过滤掉已就业成员的offers
        valid_offers = []
        for offer in job_offers:
            lh_type = offer.get("lh_type")
            # 检查对应的劳动力是否已经有工作
            is_employed = False
            for labor_hour in self.labor_hours:
                if (labor_hour.lh_type == lh_type and 
                    not labor_hour.is_valid and 
                    labor_hour.company_id is not None):
                    is_employed = True
                    break
            
            if is_employed:
                print(f"  ⚠️  跳过offer: 家庭 {self.household_id} ({lh_type}) 已经有工作了")
            else:
                valid_offers.append(offer)
        
        if not valid_offers:
            print(f"  ℹ️  家庭 {self.household_id} 所有成员都已就业，无需评估offers")
            return []
        
        # 按家庭成员分组有效offers
        head_offers = [offer for offer in valid_offers if offer.get("lh_type") == "head"]
        spouse_offers = [offer for offer in valid_offers if offer.get("lh_type") == "spouse"]
        
        accepted_offers = []
        
        # 为户主选择最佳offer
        if head_offers:
            print(f"  👨 户主收到 {len(head_offers)} 个offers")
            head_choice = await self.llm_evaluate_offers(head_offers, "head", std_jobs)
            if head_choice:
                accepted_offers.append(head_choice)
                print(f"    ✅ 家庭 {self.household_id}户主接受: {head_choice['job_title']} @ ${head_choice['offered_wage']:.2f}/小时")
            else:
                print(f"    ❌ 家庭 {self.household_id}户主拒绝所有offers")
        
        # 为配偶选择最佳offer
        if spouse_offers:
            print(f"  👩 家庭 {self.household_id}配偶收到 {len(spouse_offers)} 个offers")
            spouse_choice = await self.llm_evaluate_offers(spouse_offers, "spouse", std_jobs)
            if spouse_choice:
                accepted_offers.append(spouse_choice)
                print(f"    ✅ 家庭 {self.household_id}配偶接受: {spouse_choice['job_title']} @ ${spouse_choice['offered_wage']:.2f}/小时")
            else:
                print(f"    ❌ 家庭 {self.household_id}配偶拒绝所有offers")
        
        return accepted_offers
    
    async def llm_evaluate_offers(self, offers: List[Dict], role: str, std_jobs=None) -> Optional[Dict]:
        """
        使用LLM评估多个job offers并选择最佳的一个
        
        Args:
            offers: job offers列表
            role: 角色类型 ("head" 或 "spouse")
            std_jobs: 标准工作数据，用于市场薪资比较
            
        Returns:
            Optional[Dict]: 选择的offer，如果都拒绝则返回None
        """
        if not offers:
            return None
        
        if len(offers) == 1:
            # 只有一个offer时，评估是否接受
            offer = offers[0]
            should_accept = await self.llm_should_accept_offer(offer, role)
            if should_accept:
                print(f"        ✅ LLM决定接受单个offer")
                return offer
            else:
                print(f"        ❌ LLM决定拒绝单个offer")
                return None
        
        # 多个offers时，选择最佳的一个
        return await self.llm_choose_best_offer(offers, role, std_jobs)
    
    async def llm_should_accept_offer(self, offer: Dict, role: str) -> bool:
        """
        使用LLM决定是否接受单个job offer
        """
        # 获取当前家庭状况
        try:
            current_balance = await self.economic_center.query_balance.remote(self.household_id)
        except:
            current_balance = 0.0
        
        family_context = self.get_family_context_for_wage_expectation()
        family_context["current_balance"] = current_balance
        
        # 数据清理：确保薪资是数字类型
        offered_wage = self._clean_wage_data(offer.get('offered_wage', 0))
        
        prompt = f"""You are helping a {role} decide whether to accept a job offer.

**Job Offer Details:**
- Position: {offer.get('job_title', 'Unknown')}
- Company: {offer.get('company_id', 'Unknown')}
- Offered Wage: ${offered_wage:.2f}/hour
- Hours per Period: {offer.get('hours_per_period', 40)}
- Monthly Income: ${offered_wage * offer.get('hours_per_period', 40) * 4:.2f}

**Family Context:**
- Family Size: {family_context['family_size']} people
- Current Balance: ${family_context['current_balance']:.2f}
- Monthly Expenses: ${family_context['monthly_expenses']:.2f}
- Current Income: ${family_context['current_income']:.2f}
- Has Spouse: {'Yes' if family_context['has_spouse'] else 'No'}
- Number of Children: {family_context['num_children']}

**Decision Criteria:**
1. **Financial Need**: Does the family need this income urgently?
2. **Wage Adequacy**: Is the offered wage sufficient for family needs?
3. **Job Quality**: Is this a reasonable job for the person's skills?
4. **Family Situation**: How does this job fit the family's overall situation?

**Task:**
Decide whether to ACCEPT or REJECT this job offer.

**Response Format (JSON only):**
{{
    "decision": "accept" or "reject",
    "reasoning": "Detailed explanation of the decision",
    "confidence": 0.8,
    "key_factors": ["factor1", "factor2", "factor3"]
}}

**Guidelines:**
- Accept if the job provides needed income and reasonable working conditions
- Reject if the wage is too low relative to family needs or if family doesn't need the income urgently
- Consider the role importance (head vs spouse) in family income
"""
        
        try:
            response = await client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            response_content = response.choices[0].message.content.strip()
            
            # 清理响应内容，提取JSON部分
            if response_content.startswith("```json"):
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_content = response_content[start_idx:end_idx]
            elif response_content.startswith("```"):
                lines = response_content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') or in_json:
                        in_json = True
                        json_lines.append(line)
                        if line.strip().endswith('}') and json_lines:
                            break
                response_content = '\n'.join(json_lines)
            
            result = json.loads(response_content)
            
            # 验证JSON解析结果
            if not isinstance(result, dict):
                raise ValueError(f"LLM返回的不是有效的JSON对象: {result}")
            
            decision = result.get("decision", "reject").lower()
            reasoning = result.get("reasoning", "No reasoning provided")
            confidence = result.get("confidence", 0.5)
            
            print(f"        💭 LLM决策: {decision.upper()}")
            print(f"        📝 理由: {reasoning[:100]}...")
            print(f"        📊 信心度: {confidence:.1%}")
            
            # 添加决策统计信息
            if decision == "accept":
                print(f"        ✅ LLM决定接受单个 offer")
            else:
                print(f"        ❌ LLM决定拒绝单个 offer")
            
            return decision == "accept"
            
        except Exception as e:
            print(f"        ❌ LLM评估失败: {e}")
            # 默认策略：如果薪资合理就接受
            offered_wage = self._clean_wage_data(offer.get('offered_wage', 0))
            monthly_income = offered_wage * offer.get('hours_per_period', 40) * 4
            should_accept_default = monthly_income > family_context['monthly_expenses'] * 0.5
            print(f"        🔄 使用默认策略: {'ACCEPT' if should_accept_default else 'REJECT'} (月收入: ${monthly_income:.2f}, 月支出: ${family_context['monthly_expenses']:.2f})")
            return should_accept_default
    
    def get_market_average_wage(self, job_title: str, std_jobs=None) -> float:
        """
        获取特定职位的市场平均薪资
        
        Args:
            job_title: 职位名称
            std_jobs: 标准工作数据
            
        Returns:
            float: 市场平均时薪，如果找不到则返回默认值
        """
        try:
            if std_jobs is not None and not std_jobs.empty:
                # 尝试精确匹配职位名称
                matching_jobs = std_jobs[std_jobs['Title'].str.contains(job_title, case=False, na=False)]
                
                if not matching_jobs.empty:
                    # 如果有多个匹配，取平均值
                    wages = []
                    for _, job in matching_jobs.head(5).iterrows():  # 最多取5个相似职位
                        wage = job.get('wage_per_hour', 0)
                        if isinstance(wage, (int, float)) and wage > 0:
                            wages.append(wage)
                    
                    if wages:
                        market_wage = sum(wages) / len(wages)
                        return round(market_wage, 2)
                
                # 如果精确匹配失败，尝试关键词匹配
                keywords = job_title.lower().split()
                for keyword in keywords:
                    if len(keyword) > 3:  # 忽略太短的词
                        matching_jobs = std_jobs[std_jobs['Title'].str.contains(keyword, case=False, na=False)]
                        if not matching_jobs.empty:
                            wage = matching_jobs.iloc[0].get('wage_per_hour', 0)
                            if isinstance(wage, (int, float)) and wage > 0:
                                return round(wage, 2)
            
            # 如果都找不到，根据职位类型返回默认市场薪资
            return self._get_default_market_wage(job_title)
            
        except Exception as e:
            logger.warning(f"获取市场薪资失败 {job_title}: {e}")
            return self._get_default_market_wage(job_title)
    
    def _get_default_market_wage(self, job_title: str) -> float:
        """根据职位类型返回默认市场薪资"""
        job_title_lower = job_title.lower()
        
        # 基于职位关键词的默认薪资映射
        wage_mapping = {
            'manager': 35.0, 'director': 45.0, 'executive': 55.0,
            'engineer': 40.0, 'developer': 38.0, 'analyst': 32.0,
            'specialist': 28.0, 'coordinator': 25.0, 'assistant': 20.0,
            'clerk': 18.0, 'representative': 22.0, 'technician': 26.0,
            'supervisor': 30.0, 'lead': 33.0, 'senior': 35.0,
            'sales': 25.0, 'marketing': 28.0, 'finance': 32.0,
            'hr': 30.0, 'operations': 28.0, 'customer': 22.0
        }
        
        for keyword, wage in wage_mapping.items():
            if keyword in job_title_lower:
                return wage
        
        # 默认市场薪资
        return 25.0

    async def llm_choose_best_offer(self, offers: List[Dict], role: str, std_jobs=None) -> Optional[Dict]:
        """
        使用LLM从多个offers中选择最佳的一个
        
        Args:
            offers: job offers列表
            role: 角色类型 ("head" 或 "spouse")
            std_jobs: 标准工作数据，用于市场薪资比较
        """
        # 获取当前家庭状况
        try:
            current_balance = await self.economic_center.query_balance.remote(self.household_id)
        except:
            current_balance = 0.0
        
        family_context = self.get_family_context_for_wage_expectation()
        family_context["current_balance"] = current_balance
        
        offers_info = ""
        for i, offer in enumerate(offers, 1):
            offered_wage = self._clean_wage_data(offer.get('offered_wage', 0))
            monthly_income = offered_wage * offer.get('hours_per_period', 40) * 4
            job_title = offer.get('job_title', 'Unknown')
            
            # 添加技能匹配信息（如果可用）
            skill_match_info = ""
            if 'skill_match_score' in offer:
                match_score = offer['skill_match_score']
                match_level = 'High' if match_score > 0.7 else 'Medium' if match_score > 0.4 else 'Low'
                skill_match_info = f"\n- Skill Match: {match_score:.0%} ({match_level} Match)"
            
            # 添加市场薪资比较信息
            market_wage = self.get_market_average_wage(job_title, std_jobs=std_jobs)
            wage_competitiveness = offered_wage / market_wage if market_wage > 0 else 1.0
            
            if wage_competitiveness > 1.15:
                wage_comparison = f"Excellent (+{(wage_competitiveness-1)*100:.0f}% above market)"
            elif wage_competitiveness > 1.05:
                wage_comparison = f"Above Market (+{(wage_competitiveness-1)*100:.0f}%)"
            elif wage_competitiveness > 0.95:
                wage_comparison = f"Market Rate (±{abs(wage_competitiveness-1)*100:.0f}%)"
            elif wage_competitiveness > 0.85:
                wage_comparison = f"Below Market (-{(1-wage_competitiveness)*100:.0f}%)"
            else:
                wage_comparison = f"Poor (-{(1-wage_competitiveness)*100:.0f}% below market)"
            
            market_info = f"\n- Market Average: ${market_wage:.2f}/hour\n- Wage Level: {wage_comparison}"
            
            # 添加就业紧迫性信息
            urgency_info = ""
            if family_context.get('current_balance', 0) < family_context.get('monthly_expenses', 2000):
                urgency_info = f"\n- Job Search Urgency: High (financial pressure)"
            
            offers_info += f"""
Offer {i}:
- Position: {job_title}
- Company: {offer.get('company_id', 'Unknown')}
- Wage: ${offered_wage:.2f}/hour
- Monthly Income: ${monthly_income:.2f}
- Hours: {offer.get('hours_per_period', 40)} per period{market_info}{skill_match_info}{urgency_info}
"""
        
        prompt = f"""You are helping a {role} choose the best job offer from multiple options.

**Available Job Offers:**{offers_info}

**Family Context:**
- Family Size: {family_context['family_size']} people
- Current Balance: ${family_context['current_balance']:.2f}
- Monthly Expenses: ${family_context['monthly_expenses']:.2f}
- Current Income: ${family_context['current_income']:.2f}
- Has Spouse: {'Yes' if family_context['has_spouse'] else 'No'}
- Number of Children: {family_context['num_children']}

**Selection Criteria:**
1. **Market Competitiveness**: Prioritize offers with above-market wages (better value)
2. **Skill Match**: Choose jobs with higher skill match scores (better career prospects)
3. **Financial Return**: Consider both immediate income and long-term earning potential
4. **Family Needs**: Factor in financial pressure and current circumstances
5. **Overall Value**: Balance market rate, skill match, and family situation

**Task:**
Choose the BEST offer from the available options, or choose to REJECT ALL if none are suitable.

**Response Format (JSON only):**
{{
    "choice": 1-{len(offers)} or "reject_all",
    "reasoning": "Detailed explanation of the choice",
    "confidence": 0.8,
    "key_factors": ["factor1", "factor2", "factor3"]
}}

**Guidelines:**
- Choose the offer that provides the best overall value for the family
- Prioritize above-market wages when possible (indicates good employer/opportunity)
- Consider skill match for long-term career growth and job satisfaction
- PREFER to accept a reasonable offer rather than reject all
- Only reject all offers if wages are significantly below market AND below family needs
- If offers are similar in market value, choose based on skill match or other factors
"""
        
        try:
            response = await client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            response_content = response.choices[0].message.content.strip()
            
            # 清理响应内容，提取JSON部分
            if response_content.startswith("```json"):
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_content = response_content[start_idx:end_idx]
            elif response_content.startswith("```"):
                lines = response_content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') or in_json:
                        in_json = True
                        json_lines.append(line)
                        if line.strip().endswith('}') and json_lines:
                            break
                response_content = '\n'.join(json_lines)
            
            result = json.loads(response_content)
            
            choice = result.get("choice", "reject_all")
            reasoning = result.get("reasoning", "No reasoning provided")
            confidence = result.get("confidence", 0.5)
            
            print(f"        💭 LLM选择: {choice}")
            print(f"        📝 理由: {reasoning[:100]}...")
            print(f"        📊 信心度: {confidence:.1%}")
            
            if choice == "reject_all":
                print(f"        ❌ LLM决定拒绝所有 offers")
                
                # 合理性检查：如果有合理的offer，不应该全部拒绝
                reasonable_offers = []
                monthly_expenses = family_context.get('monthly_expenses', 2000)  # 默认月支出
                for i, offer in enumerate(offers, 1):
                    offered_wage = self._clean_wage_data(offer.get('offered_wage', 0))
                    monthly_income = offered_wage * offer.get('hours_per_period', 40) * 4
                    # 如果月收入 >= 月支出的50%，认为是合理的offer
                    if monthly_income >= monthly_expenses * 0.5:
                        reasonable_offers.append(i)
                
                if reasonable_offers:
                    # 有合理offer但LLM拒绝了，随机选择一个合理的
                    import random
                    chosen_offer = random.choice(reasonable_offers)
                    print(f"        🔄 合理性检查：发现合理offer，随机选择 offer #{chosen_offer}")
                    return offers[chosen_offer - 1]
                else:
                    print(f"        ✅ 合理性检查通过：确实没有合适的offers")
                    return None
            else:
                print(f"        ✅ LLM决定接受 offer #{choice}")
            
            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(offers):
                    return offers[choice_index]
                else:
                    print(f"        ⚠️  LLM选择的索引超出范围: {choice} (有效范围: 1-{len(offers)})")
                    return None
            except (ValueError, TypeError):
                print(f"        ⚠️  无法解析LLM选择的索引: {choice}")
                return None
                
        except Exception as e:
            print(f"        ❌ LLM选择失败: {e}")
            # 默认策略：选择薪资最高的
            best_offer = max(offers, key=lambda x: self._clean_wage_data(x.get('offered_wage', 0)))
            offered_wage = self._clean_wage_data(best_offer.get('offered_wage', 0))
            monthly_income = offered_wage * best_offer.get('hours_per_period', 40) * 4
            if monthly_income > family_context['monthly_expenses'] * 0.5:
                print(f"        🔄 使用默认策略: 选择薪资最高的offer (${offered_wage:.2f}/小时)")
                return best_offer
            else:
                print(f"        🔄 使用默认策略: 拒绝所有offers (最高薪资${offered_wage:.2f}/小时仍不足以满足需求)")
                return None
    
    async def submit_job_applications_to_market(self, job_applications: List[JobApplication], current_month):
        """
        将工作申请提交到劳动力市场
        
        Args:
            job_applications: JobApplication对象列表
        """
        for application in job_applications:
            try:
                success = await self.labormarket.submit_job_application.remote(application, current_month)
                if success:
                    # logger.info(f"Job application submitted successfully: {application.household_id} -> {application.job_id}")
                    pass
                else:
                    logger.warning(f"Failed to submit job application: {application.household_id} -> {application.job_id}")
            except Exception as e:
                logger.error(f"Error submitting job application: {e}")
        
    async def update_labor_hours(self, job: Job, lh_type: str):
        """
        更新指定类型的labor_hour状态为已就业
        """
        for lh in self.labor_hours:
            if lh.lh_type == lh_type and lh.is_valid:
                lh.is_valid = False
                lh.job_title = job.title
                lh.job_SOC = job.SOC
                lh.company_id = job.company_id
                if lh_type == 'head':
                    self.head_job = job
                elif lh_type == 'spouse':
                    self.spouse_job = job
                logger.debug(f"✅ 更新 {lh_type} labor_hour状态: is_valid=False, company_id={job.company_id}")
                
                # 不在此处通知企业，由主循环统一分发
                break
        else:
            logger.warning(f"❌ 警告: 家庭 {self.household_id} 没有找到匹配的 {lh_type} labor_hour (is_valid=True)")
    
    
    # Household 不再直接通知企业，由主循环处理
    
    # Household 不再直接通知企业，由主循环处理

    async def dismiss_worker(self, lh_type: str, company_id: str, job_soc: str) -> bool:
        """
        辞退指定的家庭成员工人
        
        Args:
            lh_type: 劳动力类型 ('head' 或 'spouse')
            company_id: 公司ID
            job_soc: 工作SOC代码
            
        Returns:
            bool: 是否成功辞退
        """
        try:
            for labor_hour in self.labor_hours:
                if (labor_hour.lh_type == lh_type and 
                    labor_hour.company_id == company_id and 
                    labor_hour.job_SOC == job_soc and 
                    not labor_hour.is_valid):  # 当前已被雇佣
                    
                    # 记录修改前的状态
                    print(f"🔍 修改前状态: 家庭 {self.household_id} ({lh_type}) - is_valid={labor_hour.is_valid}, company_id={labor_hour.company_id}")
                                        
                    # 恢复劳动力为可用状态
                    labor_hour.is_valid = True
                    labor_hour.company_id = None
                    labor_hour.job_title = None
                    labor_hour.job_SOC = None
                    
                    # 更新家庭的head_job/spouse_job状态
                    if lh_type == 'head':
                        self.head_job = None
                    elif lh_type == 'spouse':
                        self.spouse_job = None
                    
                    # 记录修改后的状态
                    print(f"🔍 修改后状态: 家庭 {self.household_id} ({lh_type}) - is_valid={labor_hour.is_valid}, company_id={labor_hour.company_id}")
                    print(f"✅ 家庭 {self.household_id} ({lh_type}) 被辞退，恢复为可用状态 (对象ID: {id(self)})")
                    return True
            
            logger.warning(f"❌ 家庭 {self.household_id} 没有找到匹配的已雇佣 {lh_type} labor_hour")
            return False
            
        except Exception as e:
            logger.error(f"❌ 辞退家庭 {self.household_id} ({lh_type}) 失败: {e}")
            return False
    
    def get_consume_budget_data(self) -> Dict[int, Dict]:
        """获取消费预算数据"""
        return self.consume_budget
    
    def get_household_id(self) -> str:
        """获取家庭ID"""
        return self.household_id
    
    def _clean_wage_data(self, wage_data) -> float:
        """
        清理薪资数据，确保返回浮点数
        
        Args:
            wage_data: 薪资数据，可能是字符串或数字
            
        Returns:
            float: 清理后的薪资数值
        """
        if isinstance(wage_data, (int, float)):
            return float(wage_data)
        
        if isinstance(wage_data, str):
            # 移除美元符号和其他非数字字符，只保留数字和小数点
            cleaned_wage = ''.join(c for c in str(wage_data) if c.isdigit() or c == '.')
            try:
                return float(cleaned_wage) if cleaned_wage else 0.0
            except ValueError:
                print(f"        ⚠️  无法解析薪资字符串: '{wage_data}'，使用默认值: 0.0")
                return 0.0
        
        # 其他类型，返回默认值
        print(f"        ⚠️  未知薪资数据类型: {type(wage_data)}，值: {wage_data}，使用默认值: 0.0")
        return 0.0

    def commit_labor_hours(self, labor_asset_id: str, hours_worked: float):
        """
        Updates the local record of labor hours potential after working. surplus working hours
        """
        for lh_potential in self.labor_hours_potential:
            if lh_potential.id == labor_asset_id: # Use 'id' from Asset base
                lh_potential.amount = max(0, lh_potential.amount - hours_worked)
                # If labor potential runs out, remove it from the list
                if lh_potential.amount <= 1e-6:
                    self.labor_hours_potential.remove(lh_potential)
                return

    # async def query_purchase_record(self):
    #     month_spent = {}
    #     for record in self.purchase_history:
    #         product_name = record.product_name
    #         month = record.month
    #         if month not in month_spent:
    #             month_spent[month] = 0.0
    #         month_spent[month] += record.total_spent

    #         logger.info(f"Month {month}: Household {self.household_id} purchased  {product_name}. Spent: ${record.total_spent:.2f}")
    #         # print(f"Month {month}: Household {self.household_id} purchased  {product_name}. Spent: ${record.total_spent:.2f}")
    #         #根据月份计算每个月总支出
    #     return month_spent

    # ===== 原有方法（已移动到文件末尾并修复） =====
    # def calculate_consumption_budget(wealth: float, consumption_propensity: float, wealth_exponent: float) -> float:
    #     """
    #     Calculates the consumption budget based on a given wealth value,
    #     a consumption propensity multiplier, and a wealth exponent.
    #     """
    #     # 此方法已移动到文件末尾并添加了@staticmethod装饰器

    # ===== 储蓄相关方法 =====
    async def make_savings_decision(self, bank, month: int) -> float:
        """
        家庭储蓄决策：将所有余额存入银行（简化版本）
        
        Args:
            bank: 银行代理引用
            month: 当前月份
            
        Returns:
            float: 实际存款金额
        """
        current_balance = await self.get_balance_ref()
        
        # 简化逻辑：将所有余额都存入银行
        if current_balance > 0:
            success = await bank.deposit.remote(self.household_id, current_balance, month)
            if success:
                logger.info(f"Household {self.household_id} saved all balance ${current_balance:.2f} to bank")
                return current_balance
        
        return 0.0
    
    # ===== 消费方法：统一入口 =====
    async def consume(self, product_market: ProductMarket, economic_center: EconomicCenter, ex_info=None):
        """
        统一消费入口：根据配置模式选择消费策略
        集成储蓄决策
        """
        # 执行消费
        result = await self.consume_advanced(product_market, economic_center, ex_info)
        # 确保总是返回一个数值
        return result if isinstance(result, (int, float)) else 0.0
    
    # ===== 原有消费逻辑（简单模式）=====
    async def consume_simple(self, product_market: ProductMarket):
        """
        原有的简单LLM消费逻辑（简化版，使用固定预算比例）
        Household consumes products from the product market.
        """
        balance = await self.get_balance_ref()

        if balance <= 1e-6:
            # print(f"Household {self.household_id}: No balance to consume.")
            return 0.0

        # 使用固定的80%消费比例
        spendable_budget = max(0.0, balance * 0.8)
        if spendable_budget <= 1e-6:
            # print(f"Household {self.household_id}: No spendable budget.")
            return 0.0

        market_listings: List[Product] = await product_market.get_all_listings.remote()
        if not market_listings:
            # print(f"Household {self.household_id}: No products available in market.")
            return 0.0

        money_left_to_spend = spendable_budget
        total_spent = 0.0  # 添加总消费统计
        
        # Prompt for LLM
        prompt = "As an intelligent and budget-conscious household, your task is to select products from the market to enhance your household's well-being and satisfaction.\n"
        prompt += f"You currently have a budget of ${money_left_to_spend:.2f} to spend.\n"
        prompt += "When making your decisions, consider:\n"
        prompt += "1.  Priorities: What products are most needed or offer the best long-term value and utility for your household? Think about immediate needs vs. wants, and how products contribute to quality of life.\n"
        prompt += "2.  Budget Management: You are not required to spend your entire budget. It is perfectly rational to save money if current offerings don't align with your needs, if the value isn't compelling, or if your primary needs are met with less spending\n"
        prompt += "3.  Combination: You can purchase multiple different products, or multiple quantities of the same product. Aim for a balanced purchase that brings diverse benefits if desired.\n"
        prompt += "4.  Rationality: Base your decisions on maximizing overall satisfaction and value for your household, respecting available quantities and your budget.\n"
        prompt += "Respond in JSON: [{\"product_id\": \"...\", \"quantity\": ...}, ...]\n"
        prompt += "Here are the available products:\n"

        product_data = [
            {
                "product_id": listing.product_id,
                "name": listing.name,
                "description": listing.description,
                "price": listing.price,
                "available_quantity": listing.amount,
                "seller_id": listing.owner_id
            }
            for listing in market_listings
            if listing.price and listing.price > 0 and listing.amount > 1e-6
        ]

        prompt += json.dumps(product_data, indent=2)

        try:
            response = await client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            response_content = response.choices[0].message.content.strip()
            
            # 清理响应内容，提取JSON部分
            if response_content.startswith("```json"):
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_content = response_content[start_idx:end_idx]
            elif response_content.startswith("```"):
                lines = response_content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') or in_json:
                        in_json = True
                        json_lines.append(line)
                        if line.strip().endswith('}') and json_lines:
                            break
                response_content = '\n'.join(json_lines)
            
            decision = json.loads(response_content)
        except json.JSONDecodeError:
            print("LLM response could not be parsed as JSON:", response)
            decision = []

        for item in decision:
            product_id = item.get("product_id")
            quantity_to_buy = item.get("quantity")

            # if no product_id or quantity_to_buy, skip this item
            market_listing_asset = next((p for p in market_listings if p.product_id == product_id), None)
            if not market_listing_asset: continue
            if quantity_to_buy <= 1e-6 or market_listing_asset.price <= 0: continue

            #  Calculate max affordable quantity
            max_affordable_qty = money_left_to_spend / market_listing_asset.price
            quantity_to_buy = min(quantity_to_buy, market_listing_asset.amount, max_affordable_qty)

            # 🏷️ 获取商品分类
            classification = None
            if self.budget_allocator and hasattr(market_listing_asset, 'id'):
                try:
                    classification = self.budget_allocator.find_classification_by_product_id(market_listing_asset.id)
                except Exception as e:
                    logger.warning(f"获取商品分类失败 (product_id={market_listing_asset.id}): {e}")
            
            product_kwargs = dict(
                asset_type='products',
                product_id=getattr(market_listing_asset, "product_id", None),
                name=market_listing_asset.name,
                description=market_listing_asset.description,
                price=market_listing_asset.price,
                amount=quantity_to_buy,  # This is the key change: amount being bought
                owner_id=market_listing_asset.owner_id,  # Seller's ID initial
                classification=classification
            )
            product_kwargs = self._enrich_product_kwargs(product_kwargs, market_listing_asset)
            product_to_buy = Product(**product_kwargs)

            purchase_ref = self.economic_center.process_purchase.remote(
                month=self.current_month,  # Pass current month for record keeping
                buyer_id=self.household_id,
                seller_id=market_listing_asset.owner_id,
                product=product_to_buy
            )

            record = PurchaseRecord(
                product_id=market_listing_asset.product_id,
                product_name=market_listing_asset.name,
                quantity=quantity_to_buy,
                price_per_unit=market_listing_asset.price,
                total_spent=market_listing_asset.price * quantity_to_buy,
                seller_id=market_listing_asset.owner_id,
                tx_id=tx_id,  # 使用Transaction对象的id属性
                timestamp=date.today(),  # Use today's date for the purchase record
                month=self.current_month
            )
            self.purchase_history.append(record)  # Add to purchase history
            
            try:
                tx_id = await purchase_ref # Await the transaction result
                if tx_id:  # Purchase successful
                    cost = tx_id.amount
                    money_left_to_spend -= cost
                    total_spent += cost  # 累计消费金额
                    # print(f"Household {self.household_id} bought {quantity_to_buy:.2f} of {market_listing_asset.name} for ${cost:.2f}.")
                else:
                    # print(f"Household {self.household_id}: Purchase failed for {market_listing_asset.name}.")
                    pass # Purchase failed (e.g., insufficient funds already handled by EC)
            except Exception as e:
                # print(f"Household {self.household_id}: Error during purchase of {market_listing_asset.name}: {e}")
                pass # Ray remote call failed for other reasons
        
        return total_spent  # 返回总消费金额


    async def perform_tasks(self):
        print(f"Household {self.household_id} performing tasks...")
    
    # ===== 新增：高级消费决策系统 =====
    
    async def consume_advanced(self, product_market: ProductMarket, economic_center: EconomicCenter,ex_info=None):
        """
        使用高级消费决策系统，基于月度预算分配进行商品选择
        优化：添加性能监控，提升并发性能
        """
        consumption_start = time.time()
        timing_records: List[tuple] = []

        def record_step(step_name: str, started_at: float):
            duration = time.time() - started_at
            timing_records.append((step_name, duration))
            logger.info(
                f"[ConsumptionTiming] Household {self.household_id} - {step_name}: {duration:.3f}s"
            )
        
        # 性能监控点1：预算分配
        budget_start = time.time()

        # 确保属性系统已初始化（需要在 BudgetAllocator 之前）
        if not self.attribute_initialized:
            attr_init_start = time.time()
            await self.initialize_attributes()
            record_step("initialize_attributes", attr_init_start)

        # 初始化BudgetAllocator（传入 attribute_system 用于营养引导）
        if self.budget_allocator is None:
            allocator_init_start = time.time()
            self.budget_allocator = BudgetAllocator(
                product_market=product_market, 
                economic_center=economic_center,
                attribute_manager=self.attribute_system  # 🔧 传入属性系统
            )
            logger.info(f"✅ 家庭 {self.household_id} 初始化 BudgetAllocator，已传入 attribute_system")
            record_step("init_budget_allocator", allocator_init_start)
        
        # 获取当前余额和上个月工资 - 优化：减少远程调用
        context_start = time.time()
        balance_ref = self.get_balance_ref()
        balance = await balance_ref

        # 获取上个月工资 - 只在需要时查询
        last_month_income = 0
        if self.current_month > 1:
            try:
                last_month_income = await self.economic_center.query_income.remote(self.household_id, self.current_month - 1)
            except Exception as e:
                logger.warning(f"Failed to query last month income for household {self.household_id}: {e}")
                last_month_income = 0
        record_step("fetch_financial_context", context_start)

        # last_month_income = await self.get_last_month_income()
        # if last_month_income is not None:
        #     self.budget_allocator.set_last_month_income(last_month_income)
        # last_month_income=2500
        # print(f"consume")    
        # result1 = self.budget_allocator.allocate(
        #     family_id=self.household_id,
        #     current_month=self.current_month,
        #     current_income=last_month_income,
        #     total_balance=balance
        #         )
        # 如果没有提供ex_info，则生成就业状况信息
        if ex_info is None:
            ex_info = self.generate_employment_ex_info()
        
        state_prep_start = time.time()
        
        # 获取当前属性状态和需求（新版）
        current_state = None
        needs = None
        if self.attribute_initialized and self.attribute_system:
            current_state = self.attribute_system.get_current_state()
            needs = self.attribute_system.calculate_needs()
        
        # 获取社会基准数据（参考其他家庭的平均属性）
        benchmark_data = None
        try:
            # 获取所有可能的家庭ID（这里简化处理，实际应该从系统获取）
            # 从输出目录扫描所有家庭
            output_dir = self.attribute_system.config.get('output_dir', 'output') if self.attribute_system else 'output'
            all_family_ids = []
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    if item.startswith('family_') and os.path.isdir(os.path.join(output_dir, item)):
                        family_id = item.replace('family_', '')
                        all_family_ids.append(family_id)
            
            if len(all_family_ids) > 1:  # 至少需要2个家庭才能计算基准
                benchmark_manager = AttributeBenchmarkManager(output_dir)
                # 使用上个月的数据作为参考（更稳定）
                target_month = self.current_month - 1 if self.current_month > 0 else 0
                benchmark_data = benchmark_manager.get_benchmark(
                    family_ids=all_family_ids,
                    exclude_family_id=self.household_id,  # 排除自己
                    target_month=target_month
                )
                if benchmark_data:
                    logger.info(f"✅ 获取社会基准数据成功: 参考{benchmark_data['statistics']['sample_size']}个家庭的第{target_month}月数据")
        except Exception as e:
            logger.debug(f"获取社会基准数据失败（非致命错误）: {e}")
        record_step("prepare_context", state_prep_start)
        
        # ========================================
        # 🔧 新增：准备上月预算和属性数据
        # ========================================
        last_month_attrs = self._prepare_last_month_attributes()
        last_month_budget = self.last_month_budget
        
        allocation_start = time.time()
        result1 = await self.budget_allocator.allocate_with_metrics(
            family_id=self.household_id,
            current_month=self.current_month,
            current_income=last_month_income,
            total_balance=balance,
            ex_info=ex_info,
            nutrition_stock=current_state.get('nutrition_stock') if current_state else None,
            life_quality=current_state.get('life_quality') if current_state else None,
            needs=needs,
            benchmark_data=benchmark_data,  # 传递基准数据
            last_month_budget=last_month_budget,  # 🔧 新增：传入上月预算
            last_month_attributes=last_month_attrs  # 🔧 新增：传入上月属性
                )
        record_step("allocate_with_metrics", allocation_start)
        
        # ========================================
        # 🔧 新增：保存本月预算供下月使用
        # ========================================
        monthly_budget_raw = result1.get('monthly_budget', None)
        # 🔧 修复：确保类型安全，防止字符串类型传播
        if monthly_budget_raw is not None:
            try:
                self.last_month_budget = float(monthly_budget_raw)
            except (TypeError, ValueError) as e:
                logger.warning(f"⚠️ 家庭{self.household_id}: last_month_budget类型转换失败: {monthly_budget_raw} ({type(monthly_budget_raw)}), 错误: {e}")
                self.last_month_budget = None
        else:
            self.last_month_budget = None
        
        budget_duration = time.time() - budget_start
        logger.debug(f"Household {self.household_id} budget allocation: {budget_duration:.3f}s")

        self.consume_budget[self.current_month] = result1['category_budget']
        # 转换为购买操作并执行
        raw_shopping_plan = result1.get("shopping_plan")
        normalized_shopping_list = []
        # 将 consumer_decision 返回的字典结构规范化为 List[Dict] 结构
        if isinstance(raw_shopping_plan, dict):
            for category, sub in raw_shopping_plan.items():
                if isinstance(sub, dict):
                    for subcat, products in sub.items():
                        if isinstance(products, list):
                            normalized_shopping_list.append({
                                "category": category,
                                "subcategory": subcat,
                                "products": products
                            })
        elif isinstance(raw_shopping_plan, list):
            normalized_shopping_list = raw_shopping_plan
        else:
            normalized_shopping_list = []

        # 执行商品购买 - 性能监控点2
        purchase_start = time.time()
        total_product_spent, purchased_items = await self.execute_budget_based_purchases(normalized_shopping_list, product_market)
        purchase_duration = time.time() - purchase_start
        logger.debug(f"Household {self.household_id} purchases: {purchase_duration:.3f}s")
        record_step("execute_budget_based_purchases", purchase_start)
        
        # 更新属性值（基于实际购买）
        if purchased_items:
            attr_update_start = time.time()
            await self.update_attributes_after_purchase(purchased_items, raw_shopping_plan)
            record_step("update_attributes_after_purchase", attr_update_start)
        # ========== 新增：处理非商品支出 ==========
        category_budget = result1.get("category_budget", {})
        
        # 提取非商品预算
        non_product_budget = {}
        for category, amount in category_budget.items():
            if category in self.budget_allocator.no_subcat_categories:
                non_product_budget[category] = amount
        
        # 执行政府服务支付
        total_service_spent = 0.0
        if non_product_budget:
            service_start = time.time()
            total_service_spent = await self.pay_government_services(non_product_budget, self.current_month)
            record_step("pay_government_services", service_start)
        
        # 计算总支出
        total_spent = total_product_spent + total_service_spent
        
        # 性能监控总结
        total_consumption_duration = time.time() - consumption_start
        logger.info(f"Household {self.household_id} completed advanced consumption in {total_consumption_duration:.3f}s:")
        logger.info(f" - Budget allocation: {budget_duration:.3f}s")
        logger.info(f" - Product purchases: {purchase_duration:.3f}s")
        logger.info(f" - Product spending: ${total_product_spent:.2f}")
        logger.info(f" - Government service spending: ${total_service_spent:.2f}")
        logger.info(f" - Total spending: ${total_spent:.2f}")
        if timing_records:
            timing_summary = ", ".join(f"{name}={duration:.2f}s" for name, duration in timing_records)
            logger.info(f"[ConsumptionTiming] Household {self.household_id} timeline -> {timing_summary}")
        
        # 更新返回结果
        result1.update({
            "total_product_spent": total_product_spent,
            "total_service_spent": total_service_spent,
            "total_spent": total_spent,
            "non_product_budget": non_product_budget
        })
        
        return result1
    
    # ===== 家庭属性管理方法 =====
    
    async def initialize_attributes(self):
        """初始化家庭属性系统（新版）"""
        if self.attribute_initialized:
            return
        
        try:
            # 创建属性系统实例
            self.attribute_system = FamilyAttributeSystem(
                family_id=self.household_id,
                family_size=self.family_profile.get('family_size', 1)
            )
            
            # 尝试从文件加载历史数据（使用单一文件）
            output_dir = self.attribute_system.config.get('output_dir', 'output')
            state_file = os.path.join(
                output_dir,
                f"family_{self.household_id}",
                f"family_state.json"  # 改为单一文件，不带月份
            )
            
            if os.path.exists(state_file):
                # 加载已有状态
                self.attribute_system.load_from_file(state_file)
                logger.info(f"✅ 家庭 {self.household_id} 从单一文件加载属性状态 (当前月份: {self.current_month})")
            else:
                # 首次初始化，所有属性默认为0（已在FamilyAttributeSystem.__init__中设置）
                logger.info(f"📝 家庭 {self.household_id} 初始化新的属性系统（所有属性默认为0）")
                
                # 保存初始状态到文件
                self.attribute_system.save_to_file()
                logger.info(f"💾 家庭 {self.household_id} 初始属性状态已保存到单一文件")
            
            self.attribute_initialized = True
            # logger.info(f"家庭 {self.household_id} 属性系统初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 家庭 {self.household_id} 属性初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 🔧 新增：准备上月属性满足率数据
    # ========================================
    def _prepare_last_month_attributes(self) -> Optional[Dict]:
        """
        提取上月营养满足率，供LLM预算决策使用
        
        Returns:
            {
                'carbohydrate': 79.3,  # 满足率%
                'protein': 49.2,
                'fat': 30.5,
                'water': 16.4
            }
            如果没有上月数据，返回None
        """
        if not self.attribute_system:
            return None
        
        try:
            nutrition_ref = self.attribute_system.nutrition_reference
            last_supply = nutrition_ref.get('last_month_supply', {})
            last_consumption = nutrition_ref.get('last_month_consumption', {})
            
            # 检查是否有有效数据
            if not last_supply or not last_consumption:
                return None
            
            result = {}
            for attr in ['carbohydrate_g', 'protein_g', 'fat_g', 'water_g']:
                supply = last_supply.get(attr, 0)
                consumption = last_consumption.get(attr, 1)  # 避免除0
                
                if consumption > 0:
                    rate = (supply / consumption * 100)
                    # 限制在0-200%之间
                    rate = max(0, min(rate, 200))
                else:
                    rate = 0
                
                # 简化属性名（去掉_g后缀）
                attr_name = attr.replace('_g', '')
                result[attr_name] = rate
            
            # 如果所有值都是0，返回None
            if all(v == 0 for v in result.values()):
                return None
            
            return result
            
        except Exception as e:
            logger.debug(f"提取上月属性失败: {e}")
            return None
    
    def advance_to_next_month(self):
        """推进到下一个月（包含属性系统月度更新）"""
        self.current_month += 1
        
        # 属性系统月度更新 - 已移至主循环统一处理（joint_debug_test.py 步骤16）
        # 注释原因：需要传入 all_families 参数以启用 v4.0 社会比较功能
        # if self.attribute_initialized and self.attribute_system:
        #     try:
        #         # 执行月度更新（消耗营养、应用非食物效用、衰减、移除过期）
        #         self.attribute_system.monthly_update(self.current_month)
        #         
        #         # 保存月度状态
        #         self.attribute_system.save_to_file()
        #         
        #         logger.info(f"✅ 家庭 {self.household_id} 进入第 {self.current_month} 月，属性系统已更新")
        #     except Exception as e:
        #         logger.error(f"❌ 家庭 {self.household_id} 月度属性更新失败: {e}")
        # else:
        #     logger.info(f"家庭 {self.household_id} 进入第 {self.current_month} 月")
        
        logger.info(f"家庭 {self.household_id} 进入第 {self.current_month} 月")
    
    async def update_attributes_after_purchase(self, purchased_items: List[Dict], shopping_plan: Dict = None):
        """
        购买完成后更新家庭属性值（新版）
        
        Args:
            purchased_items: 实际购买的商品列表 [{'product_id', 'name', 'quantity', 'price'}, ...]
            shopping_plan: 原始购物计划（可选，暂未使用）
        """
        if not self.attribute_initialized or not self.attribute_system:
            logger.warning(f"⚠️ 家庭 {self.household_id} 属性系统未初始化，跳过属性更新")
            return
        
        try:
            # 添加购买的商品到属性系统
            # 食物会立即转换为营养值，非食物会添加到商品清单
            self.attribute_system.add_purchased_products(purchased_items)
            
            # 保存状态到文件
            self.attribute_system.save_to_file()
            
            logger.info(
                f"✅ 家庭 {self.household_id} 第 {self.current_month} 月购买属性更新完成 | "
                f"商品数: {len(purchased_items)}"
            )
            
        except Exception as e:
            logger.error(f"❌ 家庭 {self.household_id} 购买后属性更新失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_family_profile_dict(self) -> Dict[str, Any]:
        """提取家庭画像为字典格式"""
        if isinstance(self.family_profile, dict):
            return self.family_profile
        elif isinstance(self.family_profile, str):
            try:
                return json.loads(self.family_profile)
            except:
                return {'family_size': 1}
        else:
            return {'family_size': 1}
    
    def find_market_match(self, product_info: Dict, market_dict: Dict) -> Optional[Product]:
        """
        在市场中查找与计划商品最匹配的商品
        """
        product_name = product_info.get("name", "").lower()
        
        # 1. 精确匹配
        if product_name in market_dict:
            return market_dict[product_name]
        
        # 2. 部分匹配
        for market_name, listing in market_dict.items():
            if product_name in market_name or market_name in product_name:
                return listing
        
        # 3. 关键词匹配
        keywords = product_name.split()
        for keyword in keywords:
            if len(keyword) > 3:  # 忽略太短的词
                for market_name, listing in market_dict.items():
                    if keyword in market_name:
                        return listing
        
        return None
    
    async def execute_purchases(self, purchases: List[Dict], product_market: ProductMarket) -> float:
        """
        执行购买操作（复用原有的购买逻辑）
        """
        total_spent = 0.0
        
        for purchase_info in purchases:
            try:
                market_listing = purchase_info["matched_listing"]
                quantity_to_buy = purchase_info["quantity"]
                
                # 使用原有的购买验证和执行逻辑
                if quantity_to_buy <= 1e-6 or market_listing.price <= 0:
                    continue
                
                # 计算最大可负担数量
                remaining_budget = purchase_info.get("budget", market_listing.price * quantity_to_buy)
                max_affordable_qty = remaining_budget / market_listing.price
                quantity_to_buy = min(quantity_to_buy, market_listing.amount, max_affordable_qty)
                
                if quantity_to_buy <= 1e-6:
                    continue
                
                # 🏷️ 获取商品分类
                classification = None
                if self.budget_allocator and hasattr(market_listing, 'id'):
                    try:
                        classification = self.budget_allocator.find_classification_by_product_id(market_listing.id)
                    except Exception as e:
                        logger.warning(f"获取商品分类失败 (product_id={market_listing.id}): {e}")
                
                # 创建购买产品对象
                product_kwargs = dict(
                    asset_type='products',
                    product_id=getattr(market_listing, "product_id", None),
                    name=market_listing.name,
                    description=market_listing.description,
                    price=market_listing.price,
                    amount=quantity_to_buy,
                    owner_id=market_listing.owner_id,
                    classification=classification  # 添加商品分类
                )
                product_kwargs = self._enrich_product_kwargs(product_kwargs, market_listing)
                product_to_buy = Product(**product_kwargs)
                
                # 处理购买交易
                purchase_ref = self.economic_center.process_purchase.remote(
                    month=self.current_month,  # Pass current month for record keeping
                    buyer_id=self.household_id,
                    seller_id=market_listing.owner_id,
                    product=product_to_buy
                )
                
                # 🔧 等待交易完成并获取Transaction对象
                tx = await purchase_ref
                if not tx or not hasattr(tx, 'id'):
                    logger.warning(f"Purchase failed for {market_listing.name}")
                    continue
                
                # 创建购买记录
                record = PurchaseRecord(
                    product_id=market_listing.product_id,
                    product_name=market_listing.name,
                    quantity=quantity_to_buy,
                    price_per_unit=market_listing.price,
                    total_spent=market_listing.price * quantity_to_buy,
                    seller_id=market_listing.owner_id,
                    tx_id=tx.id,  # 使用Transaction对象的id属性
                    timestamp=date.today(),
                    month=self.current_month
                )
                self.purchase_history.append(record)
                
                # 等待交易完成
                tx_id = await purchase_ref
                if tx_id:  # 购买成功
                    cost = market_listing.price * quantity_to_buy
                    total_spent += cost
                    logger.debug(f"Advanced purchase: {quantity_to_buy:.2f} of {market_listing.name} for ${cost:.2f}")
                
            except Exception as e:
                logger.warning(f"Error executing purchase: {e}")
                continue
        
        return total_spent
    
    async def execute_budget_based_purchases(self, shopping_list: List[Dict], product_market: ProductMarket):
        """
        根据商品清单执行购买操作，支持consumer_decision.py生成的商品格式
        优化：并发执行所有购买操作，大幅提升性能
        
        Returns:
            Tuple[float, List[Dict]]: (总花费, 实际购买的商品列表)
        """
        total_spent = 0.0
        purchased_items = []  # 记录实际购买的商品
        
        try:
            # 🔧 修复：处理不同格式的shopping_list
            all_products = []
            
            # 检查shopping_list的类型
            if isinstance(shopping_list, dict):
                # 如果是字典格式，转换为列表
                for category, items in shopping_list.items():
                    if isinstance(items, dict):
                        products = items.get("products", [])
                        all_products.extend(products)
                    elif isinstance(items, list):
                        all_products.extend(items)
            elif isinstance(shopping_list, list):
                # 如果是列表格式
                for category_item in shopping_list:
                    if isinstance(category_item, dict):
                        products = category_item.get("products", [])
                        all_products.extend(products)
                    elif isinstance(category_item, str):
                        logger.warning(f"跳过字符串格式的shopping_list项: {category_item}")
                        continue
            else:
                logger.error(f"不支持的shopping_list格式: {type(shopping_list)}")
                return 0.0, []
            
            if not all_products:
                return 0.0, []
            
            # 补全缺失的 product_id 和 firm_id
            for product_info in all_products:
                # 如果缺少 product_id，通过商品名称查找
                if not product_info.get("product_id") or not product_info.get("owner_id"):
                    name = product_info.get("name", "")
                    if name and self.budget_allocator:
                        # 使用 budget_allocator 中的方法查找 product_id
                        product_id = self.budget_allocator.find_product_id_by_name(name, self.budget_allocator.df)
                        if product_id:
                            product_info["product_id"] = product_id
                            # 通过 product_id 查找 firm_id（传入economic_center支持竞争模式）
                            try:
                                firm_id = self.budget_allocator.find_firm_id_by_name(product_id, self.economic_center)
                                if firm_id:
                                    product_info["owner_id"] = firm_id
                            except Exception as e:
                                logger.warning(f"Failed to find firm_id for product {name}: {e}")
                        else:
                            logger.warning(f"Failed to find product_id for product: {name}")
                
                # 如果仍然缺失，记录警告但不影响后续处理
                if not product_info.get("product_id"):
                    logger.warning(f"Product missing product_id: {product_info.get('name', 'Unknown')}")
                if not product_info.get("owner_id"):
                    logger.warning(f"Product missing owner_id/firm_id: {product_info.get('name', 'Unknown')}")
            
            # 🔧 批量购买：一次性发送所有购买请求，只需一次Ray远程调用
            purchase_list = []
            for product_info in all_products:
                firm_id = product_info.get("owner_id")
                product_id = product_info.get("product_id")
                name = product_info.get("name", "Unknown Product")
                price = product_info.get("price", 0.0)
                quantity = product_info.get("quantity", 1)
                
                # 🔧 跳过无效商品（缺少必要的ID）
                if not product_id or not firm_id:
                    logger.warning(f"Skipping purchase due to missing IDs: {name} (product_id={product_id}, firm_id={firm_id})")
                    continue
                
                # 🏷️ 获取商品分类（用于毛利率计算）
                classification = None
                if self.budget_allocator and product_id:
                    try:
                        classification = self.budget_allocator.find_classification_by_product_id(product_id)
                    except Exception as e:
                        logger.warning(f"获取商品分类失败 (product_id={product_id}): {e}")
                
                # 创建购买产品对象
                product_kwargs = dict(
                    asset_type='products',
                    product_id=product_id,
                    name=name,
                    description=f"Product from shopping list: {name}",
                    price=price,
                    amount=quantity,
                    owner_id=firm_id,
                    classification=classification  # 添加商品分类
                )
                product_kwargs = self._enrich_product_kwargs(product_kwargs)
                product_to_buy = Product(**product_kwargs)
                
                purchase_list.append({
                    'seller_id': firm_id,
                    'product': product_to_buy,
                    'quantity': quantity,
                    'product_info': product_info  # 保存原始信息用于记录
                })
            
            # 一次性批量处理所有购买（只需1次Ray调用）
            if purchase_list:
                tx_results = await self.economic_center.process_batch_purchases.remote(
                    self.current_month,
                    self.household_id,
                    purchase_list
                )
                
                # 处理结果并创建购买记录
                for idx, tx in enumerate(tx_results):
                    if tx:  # 购买成功
                        product_info = purchase_list[idx]['product_info']
                        product_to_buy = purchase_list[idx]['product']
                        
                        record = PurchaseRecord(
                            product_id=product_info.get("product_id"),
                            product_name=product_info.get("name", "Unknown"),
                            quantity=product_info.get("quantity", 1),
                            price_per_unit=product_info.get("price", 0.0),
                            total_spent=product_to_buy.price * product_to_buy.amount,
                            seller_id=product_info.get("owner_id"),
                            tx_id=tx,
                            timestamp=date.today(),
                            month=self.current_month
                        )
                        self.purchase_history.append(record)
                        total_spent += record.total_spent
                        
                        # 记录实际购买的商品
                        purchased_items.append({
                            'product_id': record.product_id,
                            'name': record.product_name,
                            'quantity': record.quantity,
                            'price': record.price_per_unit,
                            'attributes': product_to_buy.attributes,
                            'is_food': product_to_buy.is_food,
                            'nutrition_supply': product_to_buy.nutrition_supply,
                            'satisfaction_attributes': product_to_buy.satisfaction_attributes,
                            'duration_months': product_to_buy.duration_months
                        })
                        
                        logger.debug(f"Batch purchase: {record.quantity} of {record.product_name} for ${record.total_spent:.2f}")
                    else:
                        # 购买失败，尝试部分购买或替代品
                        product_info = purchase_list[idx]['product_info']
                        logger.warning(f"Batch purchase failed for {product_info.get('name', 'Unknown')}, trying partial purchase...")
                        
                        # 调用部分购买逻辑
                        partial_spent = await self._try_partial_purchase(product_info, product_market)
                        if partial_spent > 0:
                            total_spent += partial_spent
                            logger.info(f"✅ Partial purchase succeeded for {product_info.get('name', 'Unknown')}: ${partial_spent:.2f}")
                        else:
                            logger.warning(f"❌ Partial purchase also failed for {product_info.get('name', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Error in execute_budget_based_purchases: {e}")
        
        return total_spent, purchased_items
    
    async def _try_partial_purchase(self, product_info: Dict, product_market: ProductMarket) -> float:
        """
        尝试购买部分数量或寻找替代品
        """
        try:
            # 补全缺失的 product_id 和 firm_id（如果需要）
            if (not product_info.get("product_id") or not product_info.get("owner_id")) and self.budget_allocator:
                name = product_info.get("name", "")
                if name:
                    product_id = self.budget_allocator.find_product_id_by_name(name, self.budget_allocator.df)
                    if product_id:
                        product_info["product_id"] = product_id
                        try:
                            firm_id = self.budget_allocator.find_firm_id_by_name(product_id, self.economic_center)
                            if firm_id:
                                product_info["owner_id"] = firm_id
                        except Exception:
                            pass
            
            firm_id = product_info.get("owner_id")
            product_id = product_info.get("product_id")
            name = product_info.get("name", "Unknown Product")
            price = product_info.get("price", 0.0)
            original_quantity = product_info.get("quantity", 1)
            
            # 🏷️ 获取商品分类
            classification = None
            if self.budget_allocator and product_id:
                try:
                    classification = self.budget_allocator.find_classification_by_product_id(product_id)
                except Exception as e:
                    logger.warning(f"获取商品分类失败 (product_id={product_id}): {e}")
            
            # 1. 尝试购买部分数量（从最大可能数量开始递减）
            for attempt_quantity in range(int(original_quantity), 0, -1):
                if attempt_quantity <= 0:
                    break
                    
                # 创建部分购买的产品对象
                product_kwargs = dict(
                    asset_type='products',
                    product_id=product_id,
                    name=name,
                    description=f"Partial purchase: {name}",
                    price=price,
                    amount=attempt_quantity,
                    owner_id=firm_id,
                    classification=classification  # 添加商品分类
                )
                product_kwargs = self._enrich_product_kwargs(product_kwargs)
                partial_product = Product(**product_kwargs)
                
                # 尝试购买部分数量
                purchase_ref = self.economic_center.process_purchase.remote(
                    month=self.current_month,
                    buyer_id=self.household_id,
                    seller_id=firm_id,
                    product=partial_product
                )
                
                tx = await purchase_ref
                if tx and hasattr(tx, 'id'):  # 部分购买成功
                    # 创建购买记录
                    record = PurchaseRecord(
                        product_id=product_id,
                        product_name=name,
                        quantity=attempt_quantity,
                        price_per_unit=price,
                        total_spent=tx.amount if hasattr(tx, 'amount') else (price * attempt_quantity),
                        seller_id=firm_id,
                        tx_id=tx.id,  # 确保使用Transaction.id
                        timestamp=date.today(),
                        month=self.current_month
                    )
                    self.purchase_history.append(record)
                    
                    logger.info(f"Partial purchase: {attempt_quantity}/{original_quantity} of {name} for ${tx.amount:.2f}")
                    return tx.amount
            
            # 2. 如果部分购买也失败，尝试寻找替代品
            return await self._try_alternative_product(product_info, product_market)
            
        except Exception as e:
            logger.warning(f"Partial purchase failed for {product_info.get('name', 'Unknown')}: {e}")
            return 0.0
    
    async def _try_alternative_product(self, original_product_info: Dict, product_market: ProductMarket) -> float:
        """
        尝试寻找替代品
        """
        try:
            name = original_product_info.get("name", "")
            price = original_product_info.get("price", 0.0)
            quantity = original_product_info.get("quantity", 1)
            
            # 搜索类似产品
            similar_products = await product_market.search_products.remote(
                query=name,
                max_price=price * 1.2,  # 允许20%的价格差异
                top_k=3,
                economic_center=self.economic_center
            )
            
            # 尝试购买第一个可用的替代品
            for alternative in similar_products:
                if alternative.product_id != original_product_info.get("product_id"):
                    # 🏷️ 获取商品分类
                    classification = None
                    if self.budget_allocator and alternative.product_id:
                        try:
                            classification = self.budget_allocator.find_classification_by_product_id(alternative.product_id)
                        except Exception as e:
                            logger.warning(f"获取商品分类失败 (product_id={alternative.product_id}): {e}")
                    
                    # 创建替代品购买对象
                    product_kwargs = dict(
                        asset_type='products',
                        product_id=alternative.product_id,
                        name=alternative.name,
                        description=f"Alternative to {name}",
                        price=alternative.price,
                        amount=quantity,
                        owner_id=alternative.owner_id,
                        classification=classification  # 添加商品分类
                    )
                    product_kwargs = self._enrich_product_kwargs(product_kwargs, alternative)
                    alternative_product = Product(**product_kwargs)
                    
                    # 尝试购买替代品
                    purchase_ref = self.economic_center.process_purchase.remote(
                        month=self.current_month,
                        buyer_id=self.household_id,
                        seller_id=alternative.owner_id,
                        product=alternative_product
                    )
                    
                    tx = await purchase_ref
                    if tx and hasattr(tx, 'id'):  # 替代品购买成功
                        # 创建购买记录
                        record = PurchaseRecord(
                            product_id=alternative.product_id,
                            product_name=alternative.name,
                            quantity=quantity,
                            price_per_unit=alternative.price,
                            total_spent=tx.amount if hasattr(tx, 'amount') else (alternative.price * quantity),
                            seller_id=alternative.owner_id,
                            tx_id=tx.id,  # 确保使用Transaction.id
                            timestamp=date.today(),
                            month=self.current_month
                        )
                        self.purchase_history.append(record)
                        
                        logger.info(f"Alternative purchase: {quantity} of {alternative.name} (instead of {name}) for ${tx.amount:.2f}")
                        return tx.amount
            
            logger.warning(f"No alternatives found for {name}")
            return 0.0
            
        except Exception as e:
            logger.warning(f"Alternative purchase failed for {original_product_info.get('name', 'Unknown')}: {e}")
            return 0.0

    def query_purchase_record(self, month: int) -> float:
        """
        查询指定月份的购买记录
        """
        total_spent = 0.0
        for record in self.purchase_history:
            if record.month == month:
                total_spent += record.total_spent
        return total_spent
    
    def query_total_spent(self) -> float:
        """
        查询累积总支出（所有月份）
        """
        total_spent = 0.0
        for record in self.purchase_history:
            total_spent += record.total_spent
        return total_spent
    
    def query_all_months_spent(self) -> Dict[int, float]:
        """
        查询所有月份的支出记录
        """
        month_spent = {}
        for record in self.purchase_history:
            month = record.month
            if month not in month_spent:
                month_spent[month] = 0.0
            month_spent[month] += record.total_spent
        return month_spent
    
    async def pay_government_services(self, non_product_budget: Dict[str, float], month: int) -> float:
        """
        将非商品预算转给政府（使用固定政府ID）
        复用现有的商品购买逻辑和记录系统
        
        Args:
            non_product_budget: 非商品预算字典 {category: amount}
            month: 当前月份
        
        Returns:
            float: 实际支付的总金额
        """
        GOVERNMENT_ID = "gov_main_simulation"  # 固定政府ID
        total_paid = 0.0
        
        try:
            for service_category, amount in non_product_budget.items():
                if amount <= 0:
                    continue
                    
                # 🔧 改为英文服务名称
                service_name_mapping = {
                    '教育': 'Education',
                    '医疗/保健': 'Healthcare',
                    '交通/通讯': 'Transportation',
                    '水电煤/其他': 'Utilities',
                    '电话/互联网支出': 'Telecom/Internet'
                }
                service_name_zh = self.budget_allocator.no_subcat_categories.get(service_category, service_category)
                service_name_en = service_name_mapping.get(service_name_zh, service_name_zh)
                
                # 创建虚拟的政府服务"商品"
                product_kwargs = dict(
                    product_id=f"gov_service_{service_category}_{month}_{self.household_id}",
                    name=f"Government Service - {service_name_en}",
                    price=amount,
                    amount=1.0,  # 数量为1
                    owner_id=GOVERNMENT_ID,
                    classification="government_service"
                )
                product_kwargs = self._enrich_product_kwargs(product_kwargs)
                service_product = Product(**product_kwargs)
                
                # 检查家庭余额是否足够
                current_balance = await self.economic_center.query_balance.remote(self.household_id)
                if current_balance < amount:
                    logger.warning(f"Household {self.household_id} insufficient balance for {service_name_en}: ${current_balance:.2f} < ${amount:.2f}")
                    continue
                
                # 使用新的服务交易方法，直接更新账本并记录交易历史
                try:
                    # 使用add_tx_service直接处理政府服务支付
                    tx_id = await self.economic_center.add_tx_service.remote(
                        month=month,
                        sender_id=self.household_id,
                        receiver_id=GOVERNMENT_ID,
                        amount=amount
                    )
                    
                    # 记录到购买历史（保持兼容性）
                    record = PurchaseRecord(
                        product_id=f"gov_service_{service_category}_{month}_{self.household_id}",
                        product_name=f"Government Service - {service_name_en}",
                        quantity=1.0,
                        price_per_unit=amount,
                        total_spent=amount,
                        seller_id=GOVERNMENT_ID,
                        tx_id=tx_id,
                        timestamp=datetime.now(),
                        month=month
                    )
                    self.purchase_history.append(record)
                    
                    total_paid += amount
                    # logger.info(f"Household {self.household_id} paid ${amount:.2f} for {service_name_en}")
                    
                except Exception as transfer_error:
                    logger.error(f"Failed to transfer ${amount:.2f} for {service_name_zh}: {transfer_error}")
                    
        except Exception as e:
            logger.error(f"Error in pay_government_services: {e}")
        
        return total_paid
    
    
    @staticmethod
    def calculate_simple_consumption_budget(wealth: float, consumption_ratio: float = 0.8) -> float:
        """
        简化的预算计算方法，使用固定比例
        """
        if not (0.0 <= consumption_ratio <= 1.0):
            raise ValueError("Consumption ratio must be in the range [0, 1] inclusive.")
        
        effective_wealth = max(wealth, 0.0)
        calculated_budget = consumption_ratio * effective_wealth
        
        # 预算不能超过实际财富
        final_budget = min(calculated_budget, wealth)
        
        return final_budget
    
    @staticmethod
    def load_psid_family_data():
        """
        加载PSID家庭数据，用于初始化家庭财富
        """
        data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'consumer_modeling', 'household_data', 'PSID', 
            'extracted_data', 'processed_data', 'integrated_psid_families_data.json'
        )
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                psid_data = json.load(f)
            return psid_data
        except FileNotFoundError:
            print(f"PSID data file not found at: {data_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing PSID data: {e}")
            return None
    
    @staticmethod
    def get_initial_wealth_from_psid_2021_expenditure(household_id: Optional[str] = None) -> float:
        """
        从PSID数据中获取2021年消费支出作为初始财富，并乘以1-1.5倍的随机数
        
        Args:
            household_id: 指定的家庭ID，如果为None则随机选择
        
        Returns:
            2021年消费支出金额乘以随机倍数(1.0-1.5)作为初始财富
        """
        psid_data = Household.load_psid_family_data()
        if not psid_data:
            # 如果数据加载失败，返回默认值乘以随机倍数
            base_wealth = 50000.0
            random_multiplier = random.uniform(1.0, 1.5)
            return base_wealth * random_multiplier
        
        families = psid_data.get('families', {})
        if not families:
            base_wealth = 50000.0
            random_multiplier = random.uniform(1.0, 1.5)
            return base_wealth * random_multiplier
        
        # 如果没有指定household_id，随机选择一个家庭
        if household_id is None:
            family_id = random.choice(list(families.keys()))
        else:
            # 尝试使用指定的household_id
            family_id = household_id if household_id in families else random.choice(list(families.keys()))
        
        family_data = families[family_id]
        
        # 直接使用2021年的支出数据（索引5对应2021年）
        total_expenditure = family_data.get('total_income_expenditure', {}).get('total_expenditure', [])
        base_expenditure = 50000.0  # 默认基础支出
        
        if len(total_expenditure) >= 6:  # 确保有2021年的数据
            expenditure_2021 = total_expenditure[5]  # 索引5对应2021年
            if expenditure_2021 is not None and expenditure_2021 > 0:
                base_expenditure = max(expenditure_2021, 1000.0)  # 最低保证1000元
            else:
                # 如果2021年数据不可用，尝试使用其他年份的数据
                for expenditure in reversed(total_expenditure):  # 从最新年份开始
                    if expenditure is not None and expenditure > 0:
                        base_expenditure = max(expenditure, 1000.0)
                        break
        else:
            # 如果没有足够的数据，尝试使用现有数据
            for expenditure in reversed(total_expenditure):  # 从最新年份开始
                if expenditure is not None and expenditure > 0:
                    base_expenditure = max(expenditure, 1000.0)
                    break
        
        # 生成1.0到1.5之间的随机倍数
        random_multiplier = random.uniform(1.0, 1.5)
        final_wealth = base_expenditure * random_multiplier
        
        # logger.info(f"PSID family {family_id}: base expenditure ${base_expenditure:.2f}, multiplier {random_multiplier:.3f}, final wealth ${final_wealth:.2f}")
        
        return final_wealth
    
    @staticmethod
    def get_family_profile_from_psid(household_id: Optional[str] = None) -> Dict:
        """
        从PSID数据中获取家庭画像信息
        
        Args:
            household_id: 指定的家庭ID，如果为None则随机选择
        
        Returns:
            家庭画像字典
        """
        psid_data = Household.load_psid_family_data()
        if not psid_data:
            return {"family_size": 3, "income": "middle"}
        
        families = psid_data.get('families', {})
        if not families:
            return {"family_size": 3, "income": "middle"}
        
        # 如果没有指定household_id，随机选择一个家庭
        if household_id is None:
            family_id = random.choice(list(families.keys()))
        else:
            family_id = household_id if household_id in families else random.choice(list(families.keys()))
        
        family_data = families[family_id]
        basic_info = family_data.get('basic_family_info', {})
        income_expenditure = family_data.get('total_income_expenditure', {})
        
        # 构建家庭画像
        profile = {
            "psid_family_id": family_id,
            "family_size": basic_info.get('family_size', 3),
            "head_age": basic_info.get('head_age', 40),
            "head_gender": basic_info.get('head_gender', 'unknown'),
            "marital_status": basic_info.get('head_marital_status', 'unknown'),
            "num_children": basic_info.get('num_children', 0),
            "num_vehicles": basic_info.get('num_vehicles', 1),
            "state_code": basic_info.get('state_code', 0)
        }
        
        # 直接使用2021年收入数值
        total_income = income_expenditure.get('total_income', [])
        if len(total_income) >= 6 and total_income[5] is not None:
            profile["income"] = total_income[5]  # 直接使用2021年收入数值
        else:
            # 如果2021年收入数据不可用，尝试使用其他年份的收入
            income_value = None
            for income in reversed(total_income):  # 从最新年份开始
                if income is not None and income > 0:
                    income_value = income
                    break
            profile["income"] = income_value if income_value is not None else 50000  # 默认值
        
        return profile
    
    def update_monthly_job_status(self, month: int):
        """
        更新指定月份的工作状态记录
        
        Args:
            month: 月份
        """
        if month not in self.monthly_job_tracking:
            self.monthly_job_tracking[month] = {}
        
        # 更新head的工作状态
        head_job_info = self._get_labor_job_info('head')
        self.monthly_job_tracking[month]['head'] = head_job_info
        
        # 更新spouse的工作状态
        spouse_job_info = self._get_labor_job_info('spouse')
        self.monthly_job_tracking[month]['spouse'] = spouse_job_info
    
    def _get_labor_job_info(self, lh_type: str) -> Dict[str, Any]:
        """
        获取指定类型劳动力的工作信息
        
        Args:
            lh_type: 'head' 或 'spouse'
            
        Returns:
            Dict: 工作信息字典
        """
        for labor_hour in self.labor_hours:
            if labor_hour.lh_type == lh_type:
                if not labor_hour.is_valid:  # 已被雇佣
                    return {
                        'company_id': labor_hour.company_id,
                        'job_title': labor_hour.job_title,
                        'job_SOC': labor_hour.job_SOC,
                        'employed': True,
                        'wage': getattr(labor_hour, 'wage_per_hour', 0.0)  # 如果有工资信息
                    }
                else:  # 未被雇佣
                    return {
                        'company_id': None,
                        'job_title': None,
                        'job_SOC': None,
                        'employed': False,
                        'wage': 0.0
                    }
        
        # 如果没有找到对应的劳动力
        return {
            'company_id': None,
            'job_title': None,
            'job_SOC': None,
            'employed': False,
            'wage': 0.0
        }
    
    def get_monthly_job_status(self, month: int) -> Dict[str, Dict[str, Any]]:
        """
        获取指定月份的工作状态
        
        Args:
            month: 月份
            
        Returns:
            Dict: 该月份的工作状态
        """
        return self.monthly_job_tracking.get(month, {})
    
    def get_job_history(self, lh_type: str = None) -> Dict[int, Dict[str, Any]]:
        """
        获取工作历史记录
        
        Args:
            lh_type: 'head', 'spouse', 或 None (获取全部)
            
        Returns:
            Dict: 工作历史记录
        """
        if lh_type is None:
            return self.monthly_job_tracking
        
        history = {}
        for month, job_data in self.monthly_job_tracking.items():
            if lh_type in job_data:
                history[month] = {lh_type: job_data[lh_type]}
        return history
    
    def get_employment_statistics(self) -> Dict[str, Any]:
        """
        获取就业统计信息
        
        Returns:
            Dict: 就业统计信息
        """
        total_months = len(self.monthly_job_tracking)
        if total_months == 0:
            return {
                'total_months_tracked': 0,
                'head_employment_rate': 0.0,
                'spouse_employment_rate': 0.0,
                'household_employment_months': 0,
                'both_employed_months': 0
            }
        
        head_employed_months = 0
        spouse_employed_months = 0
        both_employed_months = 0
        household_employed_months = 0
        
        for month_data in self.monthly_job_tracking.values():
            head_employed = month_data.get('head', {}).get('employed', False)
            spouse_employed = month_data.get('spouse', {}).get('employed', False)
            
            if head_employed:
                head_employed_months += 1
            if spouse_employed:
                spouse_employed_months += 1
            if head_employed and spouse_employed:
                both_employed_months += 1
            if head_employed or spouse_employed:
                household_employed_months += 1
        
        return {
            'total_months_tracked': total_months,
            'head_employment_rate': head_employed_months / total_months,
            'spouse_employment_rate': spouse_employed_months / total_months,
            'household_employment_months': household_employed_months,
            'both_employed_months': both_employed_months,
            'household_employment_rate': household_employed_months / total_months
        }
    
    def add_wage_info_to_job_tracking(self, month: int, lh_type: str, wage: float):
        """
        为工作追踪记录添加工资信息
        
        Args:
            month: 月份
            lh_type: 'head' 或 'spouse'
            wage: 工资金额
        """
        if month in self.monthly_job_tracking and lh_type in self.monthly_job_tracking[month]:
            self.monthly_job_tracking[month][lh_type]['wage'] = wage
    
    def get_monthly_employment_summary(self) -> str:
        """
        获取月度就业情况的文字摘要
        
        Returns:
            str: 就业情况摘要
        """
        stats = self.get_employment_statistics()
        if stats['total_months_tracked'] == 0:
            return f"家庭 {self.household_id}: 暂无就业记录"
        
        return (f"家庭 {self.household_id} 就业统计 (共{stats['total_months_tracked']}个月):\n"
                f"  户主就业率: {stats['head_employment_rate']:.1%}\n"
                f"  配偶就业率: {stats['spouse_employment_rate']:.1%}\n"
                f"  家庭整体就业率: {stats['household_employment_rate']:.1%}\n"
                f"  双人就业月数: {stats['both_employed_months']}个月")
    
    def enhance_labor_skills(self, month: int, job_skills_data):
        """
        基于工作经验提升劳动力的技能和能力profile
        
        Args:
            month: 当前月份
            job_skills_data: 标准职业技能数据 {SOC: {'skills': {...}, 'abilities': {...}}}
        """
        
        for labor_hour in self.labor_hours:
            if not labor_hour.is_valid and labor_hour.job_SOC:  # 已被雇佣且有职业代码
                # 获取该职业的标准技能和能力要求
                job_data = job_skills_data[job_skills_data['O*NET-SOC Code'] == labor_hour.job_SOC].iloc[0]
                if job_data.empty:
                    continue
                
                # 提升技能

                self._enhance_profile(
                    labor_hour.skill_profile, 
                    job_data['skills'], 
                    enhancement_rate=0.05  # 每月15%的提升
                )
            
                # 提升能力

                self._enhance_profile(
                    labor_hour.ability_profile, 
                    job_data['abilities'], 
                    enhancement_rate=0.05  # 每月10%的提升
                    )
                
                # print(f"📈 家庭 {self.household_id} ({labor_hour.lh_type}) 在职业 {labor_hour.job_title} 中获得技能提升")
    
    def _enhance_profile(self, current_profile: Dict[str, float], target_profile: Dict[str, Any], enhancement_rate: float = 0.02):
        """
        提升技能或能力profile
        
        Args:
            current_profile: 当前的技能/能力profile
            target_profile: 目标职业的技能/能力要求 (包含mean, std, importance)
            enhancement_rate: 提升比例
        """
        for skill_name, skill_data in target_profile.items():
            if isinstance(skill_data, dict) and 'mean' in skill_data:
                target_level = skill_data['mean']
                
                # 使用真实的importance字段，如果没有则使用mean值作为重要性
                if 'importance' in skill_data:
                    importance = skill_data['importance'] / 5.0  # 标准化重要性 (假设最大值为5)
                else:
                    importance = skill_data['mean'] / 5.0  # 回退到使用mean值
                
                # 当前技能水平
                current_level = current_profile.get(skill_name, 0.0)
                
                # 计算提升量：基于重要性和当前与目标的差距
                gap = max(0, target_level - current_level)
                enhancement = enhancement_rate * importance * (1 + gap * 0.1)  # 差距越大，提升越快
                
                # 更新技能水平，但不超过5
                new_level = min(current_level + enhancement, 5)
                current_profile[skill_name] = round(new_level, 3)
    
    def get_skill_development_summary(self, lh_type: str = None) -> Dict[str, Any]:
        """
        获取技能发展摘要
        
        Args:
            lh_type: 'head', 'spouse', 或 None (获取全部)
            
        Returns:
            Dict: 技能发展摘要
        """
        summary = {}
        
        for labor_hour in self.labor_hours:
            if lh_type and labor_hour.lh_type != lh_type:
                continue
            
            # 计算技能总水平
            skill_total = sum(labor_hour.skill_profile.values()) if labor_hour.skill_profile else 0
            skill_count = len(labor_hour.skill_profile) if labor_hour.skill_profile else 0
            skill_avg = skill_total / skill_count if skill_count > 0 else 0
            
            # 计算能力总水平
            ability_total = sum(labor_hour.ability_profile.values()) if labor_hour.ability_profile else 0
            ability_count = len(labor_hour.ability_profile) if labor_hour.ability_profile else 0
            ability_avg = ability_total / ability_count if ability_count > 0 else 0
            
            summary[labor_hour.lh_type] = {
                'current_job': labor_hour.job_title if not labor_hour.is_valid else None,
                'job_SOC': labor_hour.job_SOC if not labor_hour.is_valid else None,
                'skill_average': skill_avg,  # 保持原始精度用于比较
                'skill_average_display': round(skill_avg, 2),  # 显示用的四舍五入值
                'ability_average': ability_avg,  # 保持原始精度用于比较
                'ability_average_display': round(ability_avg, 2),  # 显示用的四舍五入值
                'total_skills': skill_count,
                'total_abilities': ability_count,
                'employed': not labor_hour.is_valid
            }
        
        return summary
    
    def get_basic_employment_info(self) -> Dict[str, Any]:
        """
        获取家庭基本工作情况信息，用于消费决策
        
        Returns:
            Dict: 包含家庭劳动力和就业状况的基本信息
        """
        try:
            # 获取当前就业信息
            head_job = self._get_labor_job_info('head')
            spouse_job = self._get_labor_job_info('spouse')
            
            # 计算劳动力数量
            total_labor_force = len(self.labor_hours)
            employed_count = sum(1 for job in [head_job, spouse_job] if job.get('employed', False))
            
            # 计算家庭总月薪
            total_monthly_salary = 0.0
            head_monthly_salary = 0.0
            spouse_monthly_salary = 0.0
            
            if head_job.get('employed', False):
                # 假设每月工作160小时 (40小时/周 * 4周)
                head_monthly_salary = head_job.get('wage', 0.0) * 160
                total_monthly_salary += head_monthly_salary
                
            if spouse_job.get('employed', False):
                spouse_monthly_salary = spouse_job.get('wage', 0.0) * 160
                total_monthly_salary += spouse_monthly_salary
            
            employment_info = {
                'labor_force_summary': {
                    'total_labor_force': total_labor_force,
                    'employed_count': employed_count,
                },
                'head_employment': {
                    'employed': head_job.get('employed', False),
                    'job_title': head_job.get('job_title', '待业'),
                    'job_soc': head_job.get('job_SOC', ''),
                    'company_id': head_job.get('company_id', ''),
                    'hourly_wage': head_job.get('wage', 0.0),
                    'estimated_monthly_salary': head_monthly_salary
                },
                'spouse_employment': {
                    'employed': spouse_job.get('employed', False),
                    'job_title': spouse_job.get('job_title', '待业'),
                    'job_soc': spouse_job.get('job_SOC', ''),
                    'company_id': spouse_job.get('company_id', ''),
                    'hourly_wage': spouse_job.get('wage', 0.0),
                    'estimated_monthly_salary': spouse_monthly_salary
                },
                'household_income': {
                    'total_estimated_monthly_salary': total_monthly_salary,
                    'primary_earner': self._determine_primary_earner(head_job, spouse_job),
                    'income_diversification': 'dual_income' if employed_count == 2 else 'single_income' if employed_count == 1 else 'no_income'
                },
            }
            
            return employment_info
            
        except Exception as e:
            logger.warning(f"获取家庭 {self.household_id} 基本就业信息失败: {e}")
            # 返回默认信息
            return {
                'labor_force_summary': {
                    'total_labor_force': 0,
                    'employed_count': 0
                },
                'head_employment': {'employed': False, 'job_title': '待业', 'estimated_monthly_salary': 0.0},
                'spouse_employment': {'employed': False, 'job_title': '待业', 'estimated_monthly_salary': 0.0},
                'household_income': {'total_estimated_monthly_salary': 0.0, 'primary_earner': 'none', 'income_diversification': 'no_income'}
            }
    
    def _determine_primary_earner(self, head_job: Dict, spouse_job: Dict) -> str:
        """
        确定家庭主要收入来源
        
        Args:
            head_job: 户主工作信息
            spouse_job: 配偶工作信息
            
        Returns:
            str: 主要收入来源 ('head', 'spouse', 'equal', 'none')
        """
        head_wage = head_job.get('wage', 0.0) if head_job.get('employed', False) else 0.0
        spouse_wage = spouse_job.get('wage', 0.0) if spouse_job.get('employed', False) else 0.0
        
        if head_wage == 0 and spouse_wage == 0:
            return 'none'
        elif head_wage > spouse_wage * 1.2:  # 户主收入明显更高
            return 'head'
        elif spouse_wage > head_wage * 1.2:  # 配偶收入明显更高
            return 'spouse'
        else:
            return 'equal'  # 收入相近
    
    def _get_tax_info(self) -> Dict[str, float]:
        """
        获取税率信息并计算实际影响
        
        Returns:
            Dict包含:
            - income_tax_rate: 个人所得税率
            - vat_rate: 消费税率
            - combined_burden: 综合税负
            - gross_income: 税前总收入
            - after_tax_income: 税后收入
            - effective_purchasing_power: 有效购买力（考虑消费税）
        """
        try:
            # 使用实例属性中的税率
            income_tax_rate = self.income_tax_rate
            vat_rate = self.vat_rate
            
            # 获取家庭收入信息
            employment_info = self.get_basic_employment_info()
            gross_income = employment_info['household_income']['total_estimated_monthly_salary']
            
            # 计算税后收入和购买力
            after_tax_income = gross_income * (1 - income_tax_rate)
            effective_purchasing_power = after_tax_income / (1 + vat_rate)
            combined_burden = income_tax_rate + vat_rate
            
            return {
                'income_tax_rate': income_tax_rate,
                'vat_rate': vat_rate,
                'combined_burden': combined_burden,
                'gross_income': gross_income,
                'after_tax_income': after_tax_income,
                'effective_purchasing_power': effective_purchasing_power
            }
            
        except Exception as e:
            logger.warning(f"获取家庭 {self.household_id} 税率信息失败: {e}, 使用默认值")
            return {
                'income_tax_rate': self.income_tax_rate,
                'vat_rate': self.vat_rate,
                'combined_burden': self.income_tax_rate + self.vat_rate,
                'gross_income': 0,
                'after_tax_income': 0,
                'effective_purchasing_power': 0
            }
    
    def generate_employment_ex_info(self) -> str:
        """
        生成用于消费决策的就业状况ex_info（包含税率信息）
        
        Returns:
            str: 格式化的就业状况和税率信息
        """
        try:
            employment_info = self.get_basic_employment_info()
            
            # 提取关键信息
            labor_summary = employment_info['labor_force_summary']
            head_emp = employment_info['head_employment']
            spouse_emp = employment_info['spouse_employment']
            household_income = employment_info['household_income']
            
            # 计算就业率
            employment_rate = labor_summary['employed_count'] / labor_summary['total_labor_force'] if labor_summary['total_labor_force'] > 0 else 0
            
            # 获取税率信息
            tax_info = self._get_tax_info()
            
            # 构建ex_info文本 (英文版，包含税率信息)
            ex_info = f"""=== Current Household Employment Status ===
Labor Force Overview:
- Total household labor force: {labor_summary['total_labor_force']} people
- Currently employed: {labor_summary['employed_count']} people
- Household employment rate: {employment_rate:.1%}

Employment Details:
- Head: {'Employed' if head_emp['employed'] else 'Unemployed'} | Position: {head_emp['job_title']} | Monthly salary: ${head_emp['estimated_monthly_salary']:.0f}
- Spouse: {'Employed' if spouse_emp['employed'] else 'Unemployed'} | Position: {spouse_emp['job_title']} | Monthly salary: ${spouse_emp['estimated_monthly_salary']:.0f}

Income Status:
- Total estimated monthly income: ${household_income['total_estimated_monthly_salary']:.0f}
- Primary income source: {self._translate_primary_earner_en(household_income['primary_earner'])}
- Income structure: {self._translate_income_diversification_en(household_income['income_diversification'])}

=== Tax Environment ===
Tax Rates: Income {tax_info['income_tax_rate']:.1%} + Sales {tax_info['vat_rate']:.1%} = {tax_info['combined_burden']:.1%} burden
After-Tax: Gross ${tax_info['gross_income']:.0f} → Net ${tax_info['after_tax_income']:.0f} → Purchasing Power ${tax_info['effective_purchasing_power']:.0f}
Note: Product prices exclude {tax_info['vat_rate']:.1%} sales tax. Budget on net income ${tax_info['after_tax_income']:.0f}

=== Please consider employment status and tax impact in consumption decisions ==="""

            return ex_info
            
        except Exception as e:
            logger.error(f"生成家庭 {self.household_id} 就业ex_info失败: {e}")
            return "=== Current Household Employment Status ===\nFailed to retrieve employment information, adopting conservative consumption strategy\n=== Please consider employment status impact in consumption decisions ==="

    
    def _translate_primary_earner_en(self, primary_earner: str) -> str:
        """Translate primary income source to English"""
        translations = {
            'head': 'Head of household',
            'spouse': 'Spouse',
            'equal': 'Both equally',
            'none': 'No income'
        }
        return translations.get(primary_earner, primary_earner)
    
    def _translate_income_diversification_en(self, diversification: str) -> str:
        """Translate income structure to English"""
        translations = {
            'dual_income': 'Dual-income household',
            'single_income': 'Single-income household',
            'no_income': 'No-income household'
        }
        return translations.get(diversification, diversification)
