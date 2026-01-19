# Import necessary libraries.
import asyncio  # For running asynchronous operations.
from typing import Dict, Optional, List, Any  # For type hinting.
from collections import defaultdict
import json  # For parsing JSON data from LLM responses.
import pandas as pd  # For data manipulation, especially for managing relationships.
import numpy as np  # For numerical operations, used in financial simulation.

from ..center.jobmarket import Job, LaborMarket
from ..center.assetmarket import ProductMarket
from ..center.model import Product
from ..center.ecocenter import EconomicCenter
from ..llm.llm import LLM, ChatCompletionMessageParam
# Define a type alias for chat messages, making the code more readable.
ChatCompletionMessageParam = Dict[str, Any]
from ..utils.log_utils import setup_global_logger
# Initialize the logger for this module.
logger = setup_global_logger(name="firm")

def _convert_for_json(value: Any) -> Any:
    """
    A helper function to convert special data types (like numpy or pandas types)
    into JSON-serializable formats.
    """
    # Convert pandas Not a Number to Python's None.
    if pd.isna(value): return None
    # Convert numpy generic types (like np.int64) to standard Python types.
    if isinstance(value, np.generic): return value.item()
    # Convert pandas Timestamp to a string format.
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d')
    # Return the value as is if no conversion is needed.
    return value

# @ray.remote
class Firm:
    """
    ### An autonomous agent representing a company in an economic simulation.

    This class, designed as a `Ray` actor, encapsulates the behavior of a firm. It combines both strategic
    long-term planning and annual operational execution. The firm interacts with a simulated economic
    environment, including markets and a central authority, and leverages a Large Language Model (LLM)
    for complex decision-making processes.

    - **Core Functionality**:
        - **Strategic Analysis**: Uses an LLM to analyze the supply chain and customer market, assessing risks, identifying opportunities, and selecting partners.
        - **Operational Execution**: Manages annual tasks like hiring employees, making production decisions, and managing finances.
        - **State Management**: Maintains its own financial history, a list of business relationships (suppliers/customers), and profiles of other market participants.
        - **Market Interaction**: Publishes updates to and consumes information from a central `EconomicCenter`, and interacts with `ProductMarket` and `LaborMarket`.

    - **Attributes**:
        - `company_id` (str): A unique identifier for the firm.
        - `economic_center` (EconomicCenter): A handle to the central economic simulation hub.
        - `llm` (LLM): A handle to the language model used for decision-making.
        - `financial_history` (List[Dict]): A record of the firm's annual financial statements.
        - `relationships` (pd.DataFrame): A table tracking suppliers and customers.
        - `partner_profiles` (Dict): An intelligence database on other companies.
    """
    def __init__(self,
                 company_id: str,
                 company_name: str,
                 main_business: str,
                 economic_center: "EconomicCenter",
                 llm: Optional["LLM"] = None,
                 initial_funds: float = 1000000.0,
                 production_capacity: float = 1000.0,
                 supply_chain: Optional[str] = None,
                 product_market: Optional[ProductMarket] = None,
                 labor_market: Optional[LaborMarket] = None,
                 hiring_enabled: bool = True
               ):
        # --- Basic Company Information ---
        self.company_id = company_id
        self.company_name = company_name
        self.main_business = main_business
        self.opening_jobs:List[Job] = []  # A list to store job openings for the firm.
        self.labor_market = labor_market  # Reference to the LaborMarket for job management.
        # --- External System Handles ---
        self.economic_center = economic_center
        self.llm = llm
        self.product_market = product_market
        
        # --- Operational Attributes ---
        self.production_capacity = production_capacity
        self.supply_chain = supply_chain

        # --- Internal State ---
        # A list to store annual financial statements.
        self.financial_history: List[Dict[str, Any]] = []
        # 收支跟踪
        self.total_income: float = 0.0
        self.total_expenses: float = 0.0
        # 月度财务记录
        self.monthly_financials: Dict[int, Dict[str, float]] = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})
        self._initialize_financials(initial_funds)
        self.initial_funds = initial_funds
        # A pandas DataFrame to manage relationships with suppliers and customers.
        self.relationships = pd.DataFrame(columns=[
            'partner_id', 'partner_name', 'relationship_type', 'status',
            'start_year'
        ])
        
        # A dictionary to store intelligence data gathered on other companies.
        self.partner_profiles: Dict[str, Dict[str, Any]] = {}
        
        # --- Employee Management ---
        self.hiring_enabled = hiring_enabled
        self.employees: int = 0  # 记录当前员工数量
        self.employee_list: List[Dict] = []  # 员工详细信息列表
        self.employee_lookup: Dict[str, Dict] = {}  # 员工查找字典，key为"household_id_lh_type"
        
        # logger.info(f"Firm {self.company_id} ('{self.company_name}') initialized with both strategic and operational capabilities.")

    async def initialize(self):
        """
        Initializes the firm's presence in the economic simulation by registering
        it with the EconomicCenter's ledgers.
        """
        if self.economic_center:
            try:
                # 给企业设置大额初始资金，避免资金不足问题
                
                # Asynchronously register for financial, product, and labor ledgers.
                await asyncio.gather(
                    self.economic_center.init_agent_ledger.remote(self.company_id, self.initial_funds),
                    self.economic_center.init_agent_product.remote(self.company_id),
                    self.economic_center.init_agent_labor.remote(self.company_id),
                    self.economic_center.register_id.remote(self.company_id, 'firm')
                )
                # logger.info(f"Firm {self.company_id} registered in EconomicCenter's sub-ledgers.")
            except Exception as e:
                logger.warning(f"Firm {self.company_id} failed to register in sub-ledgers: {e}")
        else:
            logger.warning("EconomicCenter not available for Firm registrregistered in EconomicCenter'sation.")

    @classmethod
    def parse_dicts(cls, company_data):
        if isinstance(company_data['relationships_history'], str):
            company_data['relationships_history'] = json.loads(company_data['relationships_history'])
        else:
            company_data['relationships_history'] = []
        return {
            'company_id': company_data['factset_entity_id'],
            'company_name': company_data.get('current_proper_name', ''),
            'main_business': company_data.get('industry_category', ''),
            'production_capacity': company_data.get('production_capacity', 1000.0),
            'supply_chain': company_data['relationships_history'],
        }


    async def query_info(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "main_business": self.main_business,
            "production_capacity": self.production_capacity,
            "supply_chain": self.supply_chain,
            "current_employees": self.employees,  # 添加当前员工数量
            "employee_history_count": 0  # 添加员工历史记录数量
        }
    
    def _initialize_financials(self, initial_balance: float):
        """Sets up the initial financial statement for the firm at Year 0."""
        initial_statement = {
            "year": 0,
            "balance": initial_balance,
            "revenue": 0.0,
            "cogs": 0.0,  # Cost of Goods Sold
            "net_income": 0.0,
            "debt_to_equity_ratio": 0.0,
            "net_profit_margin": 0.0,
            "current_ratio": 10.0,
            "sales_growth_yoy": 0.0 # Year-over-Year
        }
        self.financial_history.append(initial_statement)


    async def define_job_openings(self, job_dis, std_job, labor_market, num_jobs: int) -> List[Job]: 
        self.opening_jobs = await labor_market.query_jobs.remote(self.company_id)
        jobs = job_dis[job_dis['Industry'] == self.main_business]
        codes = jobs['SOC Code'].tolist()
        proportions = jobs['Proportion'].tolist()
        job_openings = []
        created_job_codes = set()  # 跟踪已创建的工作类型
        
        for _ in range(num_jobs):
            job_code = np.random.choice(codes, p=proportions)
            
            # 检查是否已经有这种类型的工作（包括本次循环中创建的）
            existing_job_found = False
            # 先检查本次循环中创建的工作
            for job in job_openings:
                if job.SOC == job_code:
                    existing_job_found = True
                    job.positions_available += 1
                    print(f"    🔄 企业 {self.company_id} 为本次创建的职位 '{job.title}' 增加1个职位")
                    break
            
            # 如果本次循环中没有，再检查已存在的工作
            if not existing_job_found:
                for job in self.opening_jobs:
                    if job.SOC == job_code:
                        existing_job_found = True
                        await labor_market.add_job_position.remote(self.company_id, job)
                        print(f"    🔄 企业 {self.company_id} 为现有职位 '{job.title}' 增加1个职位")
                        break
            
            # 如果没有现有工作且没有创建过这种类型，创建新工作
            if not existing_job_found and job_code not in created_job_codes:
                job_info = std_job[std_job['O*NET-SOC Code'] == job_code].iloc[0]
                job = Job.create(
                    soc=job_code,
                    title=job_info['Title'],
                    wage_per_hour=job_info['Average_Wage'],
                    company_id=self.company_id,
                    description= job_info['Description'],
                    hours_per_period=40.0,  # 默认每周40小时
                    required_skills=job_info['skills'],
                    required_abilities=job_info['abilities']
                )
                job_openings.append(job)
                self.opening_jobs.append(job)  # 立即添加到opening_jobs中，供后续循环使用
                created_job_codes.add(job_code)
                print(f"    ✨ 企业 {self.company_id} 创建新职位: '{job.title}' (${job.wage_per_hour:.2f}/小时)")
        if job_openings:
            await labor_market.publish_job.remote(job_openings)
    
    async def run_cycle(self, current_year: int):
        """
        Executes a full annual cycle of observation, strategy, and operation for the firm.
        This is the main loop for the agent's yearly activities.
        """
        logger.info(f"--- FIRM {self.company_id} | YEAR {current_year} | STARTING CYCLE ---")

        # 1. Observe: Get public information from the market.
        logger.info(f"[{self.company_id}] Observing market and updating intelligence...")
        market_updates = await self.economic_center.get_public_market_updates.remote()
        self._update_partner_profiles_from_market(market_updates)

        # 2. Strategize: Analyze supply chain and customer market using LLM.
        self_profile = self._build_self_profile(current_year)

        supply_actions = await self.analyze_supply_chain(self_profile)
        logger.info(f"[{self.company_id}] Supply Chain Actions: {supply_actions}")

        market_actions = await self.analyze_customer_market(self_profile)
        logger.info(f"[{self.company_id}] Customer Market Actions: {market_actions}")
        
        # 3. Operate: Execute hiring and production based on strategy and needs.
        logger.info(f"[{self.company_id}] Executing operational tasks for year {current_year}...")
        await self.post_job_openings()
        await self.make_production_decision()

        # 4. Update State: Simulate the financial impact of the year's actions.
        self._simulate_operations_and_update_state(current_year, supply_actions, market_actions)
        
        # 5. Publish: Broadcast a public update of the firm's status.
        my_public_update = self._prepare_public_update(current_year)
        await self.economic_center.publish_update.remote(self.company_id, my_public_update)
        logger.info(f"[{self.company_id}] Published public update for year {current_year}.")

        logger.info(f"--- FIRM {self.company_id} | YEAR {current_year} | CYCLE COMPLETE ---\n")

    def _update_partner_profiles_from_market(self, market_updates: Dict[str, Dict]):
        """
        Updates the internal intelligence database (`partner_profiles`) with new
        public information received from the EconomicCenter.
        """
        for partner_id, update in market_updates.items():
            # Ignore updates about oneself.
            if partner_id == self.company_id:
                continue 

            # If this is a new partner, create a basic profile.
            if partner_id not in self.partner_profiles:
                self.partner_profiles[partner_id] = {
                    "company_id": partner_id,
                    "company_name": update.get("company_name"),
                    "main_business": update.get("main_business"),
                    "financial_history": [],
                    "business_segments": {}
                }
            
            # Append new financial data, avoiding duplicates.
            if "public_financials" in update:
                new_financials = update["public_financials"]
                profile = self.partner_profiles[partner_id]
                existing_years = {f['year'] for f in profile["financial_history"]}
                if new_financials['year'] not in existing_years:
                    profile["financial_history"].append(new_financials)
                    # Keep financial history sorted by year descending.
                    profile["financial_history"].sort(key=lambda x: x['year'], reverse=True)

            # Update business segment information.
            if "public_segments" in update:
                self.partner_profiles[partner_id]["business_segments"] = update["public_segments"]
    
    def _build_self_profile(self, current_year: int) -> Dict:
        """Constructs a dictionary containing the firm's own current profile."""
        latest_financials = self.financial_history[-1]
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "main_business": self.main_business,
            "decision_year": current_year,
            "latest_financials": latest_financials,
            "financial_history": self.financial_history.copy(),
            "business_segments": {"main": {"revenue_share": 1.0, "description": self.main_business}}
        }

    def _build_partner_profile(self, partner_id: str, decision_year: int) -> Optional[Dict]:
        """
        Constructs a dictionary containing a partner's profile from the intelligence database.
        Returns None if the partner has no data or no financial history.
        """
        if partner_id not in self.partner_profiles:
            return None
        
        partner_data = self.partner_profiles[partner_id]
        
        # A partner is not viable for analysis without financial history.
        if not partner_data.get("financial_history"):
            return None
            
        return {
            "company_id": partner_data["company_id"],
            "company_name": partner_data["company_name"],
            "main_business": partner_data["main_business"],
            "decision_year": decision_year,
            "latest_financials": partner_data["financial_history"][0], # The most recent one
            "financial_history": partner_data["financial_history"],
            "business_segments": partner_data.get("business_segments", {})
        }

    async def analyze_supply_chain(self, self_profile: dict) -> Dict:
        """
        Performs a multi-step strategic analysis of the supply chain using the LLM.
        """
        logger.info(f"[{self.company_id}] Analyzing supply chain...")
        
        # Step 1: Identify all potential suppliers from the intelligence database.
        candidate_ids = [pid for pid in self.partner_profiles.keys() if self.partner_profiles[pid].get("financial_history")]
        if not candidate_ids:
            return {"actions": "No viable candidates found in intelligence database."}

        # Step 2: Create assessment tasks for each candidate to be run in parallel.
        assessment_tasks = []
        for partner_id in candidate_ids:
            partner_profile = self._build_partner_profile(partner_id, self_profile["decision_year"])
            if partner_profile:
                prompt = self._create_supplier_analysis_prompt(self_profile, partner_profile)
                assessment_tasks.append(self.llm.atext_request([{"role": "user", "content": prompt}]))
        
        # Run all LLM assessments concurrently.
        llm_responses = await asyncio.gather(*assessment_tasks)
        
        assessments = [json.loads(resp) for resp in llm_responses if resp and resp.strip()]
        if not assessments:
            return {"error": "LLM assessment of suppliers failed."}
            
        # Step 3: Ask the LLM to determine the optimal number of suppliers.
        quantity_prompt = self._create_portfolio_size_prompt('supplier', self_profile, assessments)
        quantity_resp = await self.llm.atext_request([{"role": "user", "content": quantity_prompt}])
        target_count = json.loads(quantity_resp).get('estimated_supplier_count', len(assessments))
        
        # Step 4: Ask the LLM to select the best suppliers based on the assessments and target count.
        selection_prompt = self._create_selection_prompt('supplier', self_profile, assessments, target_count)
        selection_resp = await self.llm.atext_request([{"role": "user", "content": selection_prompt}])
        selection_result = json.loads(selection_resp)
        
        # Step 5: Convert the selection into concrete actions (establish, maintain, terminate).
        return self._formulate_actions('supplier', selection_result)

    async def analyze_customer_market(self, self_profile: dict) -> Dict:
        """
        Performs a multi-step strategic analysis of the customer market using the LLM.
        (The logic is parallel to `analyze_supply_chain`).
        """
        logger.info(f"[{self.company_id}] Analyzing customer market...")
        
        # Step 1: Identify potential customers.
        candidate_ids = [pid for pid in self.partner_profiles.keys() if self.partner_profiles[pid].get("financial_history")]
        if not candidate_ids:
            return {"actions": "No viable candidates found in intelligence database."}

        # Step 2: Create and run parallel assessment tasks.
        assessment_tasks = []
        for partner_id in candidate_ids:
            partner_profile = self._build_partner_profile(partner_id, self_profile["decision_year"])
            if partner_profile:
                prompt = self._create_customer_analysis_prompt(self_profile, partner_profile)
                assessment_tasks.append(self.llm.atext_request([{"role": "user", "content": prompt}]))
        
        llm_responses = await asyncio.gather(*assessment_tasks)
        assessments = [json.loads(resp) for resp in llm_responses if resp and resp.strip()]
        if not assessments:
            return {"error": "LLM assessment of customers failed."}
            
        # Step 3: Determine optimal number of customers.
        quantity_prompt = self._create_portfolio_size_prompt('customer', self_profile, assessments)
        quantity_resp = await self.llm.atext_request([{"role": "user", "content": quantity_prompt}])
        target_count = json.loads(quantity_resp).get('estimated_customer_count', len(assessments))
        
        # Step 4: Select the best customers.
        selection_prompt = self._create_selection_prompt('customer', self_profile, assessments, target_count)
        selection_resp = await self.llm.atext_request([{"role": "user", "content": selection_prompt}])
        selection_result = json.loads(selection_resp)
            
        # Step 5: Formulate actions.
        return self._formulate_actions('customer', selection_result)

    def _create_supplier_analysis_prompt(self, self_profile: dict, supplier_profile: dict) -> str:
        """Generates a detailed prompt for the LLM to analyze a single potential supplier."""
        # Convert profiles to JSON strings for inclusion in the prompt.
        self_profile_json = json.dumps(self_profile, indent=2, default=_convert_for_json)
        supplier_profile_json = json.dumps(supplier_profile, indent=2, default=_convert_for_json)
        
        # Calculate the duration of the existing relationship, if any.
        duration_years = "N/A"
        rel_info = self.relationships[(self.relationships['partner_id'] == supplier_profile['company_id']) & (self.relationships['relationship_type'] == 'supplier')]
        if not rel_info.empty:
            start_year = rel_info.iloc[0]['start_year']
            duration_years = f"{(self_profile['decision_year'] - start_year):.1f}"

        # The prompt provides identity, context, data, strict logic, and output format.
        return f"""
# IDENTITY and PURPOSE
You are a world-class supply chain strategist and risk analyst. Your task is to analyze a company's relationship with a SINGLE supplier based on the comprehensive data provided, and generate strategic assessments in a specific JSON format by strictly following the provided logic.

# INPUT DATA
## Self Profile (The Company You Work For, as of year {self_profile.get('decision_year')}):
{self_profile_json}

## Supplier Profile (The Company You Are Analyzing):
This profile includes basic info, a history of latest financials, and top business segments based on your intelligence.
{supplier_profile_json}

# DECISION LOGIC GUIDELINES (You MUST follow these rules precisely)
1.  **Supplier Tiering**:
    *   The relationship duration is approximately **{duration_years} years**.
    *   **Tier 1: Strategic**: (Rule for this simulation) Assign if the supplier's latest `net_profit_margin` is greater than 0.15 AND the relationship duration is 3 years or more.
    *   **Tier 2: Core**: Assign if the supplier's latest `net_profit_margin` is greater than 0.05 OR the relationship duration is 1 year or more.
    *   **Tier 3: Tactical**: This is the default tier for all other cases.

2.  **Financial Risk Assessment**: (Use the MOST RECENT record in the supplier's `financial_history`)
    *   **High**: Assign if `debt_to_equity_ratio` > 2.5 OR `net_profit_margin` < 0 OR `current_ratio` < 1.
    *   **Medium**: Assign if (`debt_to_equity_ratio` > 1.5) OR (`net_profit_margin` < 0.03). This is also the default if financial data is missing.
    *   **Low**: Assign for all other cases.

3.  **Geopolitical Risk Flag**: (Simplified rule for this simulation) Assign **Y** if `company_name` contains "Global" or "International", otherwise assign **N**.

4.  **Single Source Flag**: (Simplified rule for this simulation) Assign **Y** if `main_business` of the supplier is "Unique Patented Material" or "Proprietary Component", otherwise assign **N**.

5.  **Concentration Risk Flag**: (Simplified rule for this simulation, from our perspective) Assign **Y** if our company has less than 5 active suppliers in total, otherwise assign **N**.

6.  **Negotiation Leverage Score (for Self)**:
    *   **High**: Assign if Self's latest `revenue` is more than 10 times the Supplier's latest `revenue`.
    *   **Low**: Assign if the Supplier is a single source ('Y') OR Self's latest `revenue` is less than the Supplier's latest `revenue`.
    *   **Medium**: This is the default for all other cases.

7.  **Sourcing Recommendation**:
    *   **"Diversify Now"**: Recommend if `Single_Source_Flag` is 'Y' AND (`Financial_Risk_Score` is 'High' OR `Geopolitical_Risk_Flag` is 'Y').
    *   **"Seek Alternatives"**: Recommend if `Financial_Risk_Score` is 'High'.
    *   **"Deepen Partnership"**: Recommend if `Supplier_Tier` is 'Tier 1: Strategic' AND `Financial_Risk_Score` is 'Low' AND `Geopolitical_Risk_Flag` is 'N'.
    *   **"Maintain & Monitor"**: This is the default recommendation for all other cases.

# OUTPUT SPECIFICATION
You MUST respond with a single, valid JSON object containing ALL of the following keys:
{{
  "supplier_name": "{supplier_profile.get('company_name', 'N/A')}",
  "supplier_entity_id": "{supplier_profile['company_id']}",
  "Supplier_Tier": "Tier 1: Strategic" | "Tier 2: Core" | "Tier 3: Tactical",
  "Financial_Risk_Score": "High" | "Medium" | "Low",
  "Geopolitical_Risk_Flag": "Y" | "N",
  "Single_Source_Flag": "Y" | "N",
  "Concentration_Risk_Flag": "Y" | "N",
  "Negotiation_Leverage_Score": "High" | "Medium" | "Low",
  "Sourcing_Recommendation": "Diversify Now" | "Deepen Partnership" | "Maintain & Monitor" | "Seek Alternatives"
}}
"""

    def _create_customer_analysis_prompt(self, self_profile: dict, customer_profile: dict) -> str:
        """Generates a detailed prompt for the LLM to analyze a single potential customer."""
        self_profile_json = json.dumps(self_profile, indent=2, default=_convert_for_json)
        customer_profile_json = json.dumps(customer_profile, indent=2, default=_convert_for_json)

        return f"""
# Identity and Goal
You are a world-class Chief Revenue Officer (CRO) and market strategist. Your task is to analyze your company's relationship with a [single customer] based on the provided data, strictly adhering to the given logic to generate a strategic assessment in a specific JSON format.

# Input Data
## Own Company Profile (as of year {self_profile.get('decision_year')}):
{self_profile_json}

## Customer Profile (The company you are analyzing):
This profile contains basic information, a history of the latest financial data, and top business segments based on your intelligence.
{customer_profile_json}

# Decision Logic Guide (You must strictly follow these rules)
1.  **Customer Tiering**:
    *   (Simplified Rule for simulation) **Tier 1: Strategic**: Assign if the customer's latest `revenue` places them in the top 10% among all companies you know.
    *   **Tier 2: Key**: Assign if the customer's latest `revenue` places them in the top 50% among all companies you know.
    *   **Tier 3: General**: This is the default tier.

2.  **Customer Credit Risk Score**: (Use the MOST RECENT record in the customer's `financial_history`)
    *   **High**: Assign if `debt_to_equity_ratio` > 3.0 OR `net_profit_margin` < 0 OR `current_ratio` < 1.
    *   **Medium**: Assign if (`debt_to_equity_ratio` > 1.5) OR (`net_profit_margin` < 0.05). This is also the default if financial data is missing.
    *   **Low**: Assign for all other cases.

3.  **Market Decline Risk Flag**: Assign **Y** if the MOST RECENT `sales_growth_yoy` (Year-over-Year) in the customer's financials is not null and is less than 0, otherwise assign **N**.

4.  **Customer Concentration Risk Flag (Risk to our company)**: (Simplified Rule for simulation) Assign **Y** if this single customer represents more than 20% of our total active customer relationships, otherwise assign **N**.

5.  **Up-sell/Cross-sell Opportunity**: Based on the customer's `main_business`, identify up to 2 of [our company's] products or services that could be relevant but are not part of the current relationship. For example, if we sell 'AI Chips' and the customer's business is 'Electric Vehicles', an opportunity might be 'Autonomous Driving Software'. If no clear opportunities, return an empty list.

6.  **Sales Strategy Recommendation**:
    *   **"Manage Risk & Reduce Exposure"**: Recommend if `Customer_Credit_Risk_Score` is 'High' OR `Market_Decline_Risk_Flag` is 'Y'.
    *   **"Expand Account (Strategic)"**: Recommend if `Customer_Tier` is 'Tier 1: Strategic' AND `Customer_Credit_Risk_Score` is 'Low'.
    *   **"Maintain & Optimize"**: Recommend if `Customer_Tier` is 'Tier 2: Key' AND `Customer_Credit_Risk_Score` is 'Low' or 'Medium'.
    *   **"Automate & Scale"**: This is the default recommendation for all other cases.

# Output Format Requirements
You must return a single, valid JSON object containing all the following keys:
{{
  "customer_name": "{customer_profile.get('company_name', 'N/A')}",
  "customer_entity_id": "{customer_profile['company_id']}",
  "Customer_Tier": "Tier 1: Strategic" | "Tier 2: Key" | "Tier 3: General",
  "Customer_Credit_Risk_Score": "High" | "Medium" | "Low",
  "Market_Decline_Risk_Flag": "Y" | "N",
  "Customer_Concentration_Risk_Flag": "Y" | "N",
  "Up_sell_Cross_sell_Opportunity": ["Product/Segment Name 1", "Product/Segment Name 2"],
  "Sales_Strategy_Recommendation": "Manage Risk & Reduce Exposure" | "Expand Account (Strategic)" | "Maintain & Optimize" | "Automate & Scale"
}}
"""

    def _create_portfolio_size_prompt(self, entity_type: str, self_profile: dict, assessments: List[Dict]) -> str:
        """
        Creates a prompt for the LLM to decide the optimal number of suppliers or customers.
        """
        role = "Chief Procurement Officer (CPO)" if entity_type == 'supplier' else "Chief Revenue Officer (CRO)"
        summary = { "total_candidates_assessed": len(assessments) }
        
        return f"""
# IDENTITY AND GOAL
You are the {role}. Your goal is to make a high-level strategic decision on the optimal size of your core {entity_type} portfolio for the upcoming year.

# YOUR COMPANY'S PROFILE
{json.dumps(self_profile, indent=2, default=_convert_for_json)}

# CANDIDATE POOL SUMMARY
Your team has provided assessments for {len(assessments)} potential {entity_type}s.
{json.dumps(summary, indent=2, default=_convert_for_json)}

# TASK
Based on your company's scale, strategic objectives, and the quality of the candidate pool, determine the ideal number of core {entity_type}s to engage with. Consider the trade-offs between risk diversification, relationship depth, management overhead, and negotiation leverage.

# OUTPUT SPECIFICATION
Respond with a single JSON object with one key: "estimated_{entity_type}_count", which must be an integer.
"""

    def _create_selection_prompt(self, entity_type: str, self_profile: dict, assessments: List[Dict], target_count: int) -> str:
        """
        Creates a prompt for the LLM to select the best partners from a list of assessments.
        """
        role = "CPO" if entity_type == 'supplier' else "CRO"
        id_key = f"{entity_type}_entity_id" 
        
        return f"""
# IDENTITY AND GOAL
You are the {role}, and your final task is to select the top {target_count} {entity_type}s to form your core portfolio for the next year, based on your team's assessments.

# YOUR COMPANY'S PROFILE
{json.dumps(self_profile, indent=2, default=_convert_for_json)}

# CANDIDATE ASSESSMENTS (Prepared by your team)
{json.dumps(assessments, indent=2, default=_convert_for_json)}

# TASK
Review all the assessments provided. Based on a holistic view of your strategic goals, select the best {target_count} {entity_type}s to form your portfolio. You should aim for a balance of high performance, low risk, and strategic alignment.

# OUTPUT SPECIFICATION
Respond with a single JSON object with one key: "selected_{entity_type}_ids". The value must be a list of strings, where each string is an ID taken from the `{id_key}` field of your chosen candidates.
"""

    def _formulate_actions(self, entity_type: str, selection_result: Dict) -> Dict:
        """
        Compares the LLM's selected partners with current active partners to determine
        which relationships to establish, maintain, or terminate.
        """
        selected_ids_key = f"selected_{entity_type}_ids"
        selected_ids = set(selection_result.get(selected_ids_key, []))
        
        # Get the set of currently active partners of the given type.
        current_active_ids = set(self.relationships[
            (self.relationships['relationship_type'] == entity_type) &
            (self.relationships['status'] == 'active')
        ]['partner_id'])
        
        # Use set operations to find differences and intersections.
        return {
            "establish": list(selected_ids - current_active_ids),
            "maintain": list(selected_ids & current_active_ids),
            "terminate": list(current_active_ids - selected_ids)
        }

    def _simulate_operations_and_update_state(self, year: int, supply_actions: Dict, market_actions: Dict):
        """
        Applies strategic decisions and simulates the firm's financial performance for the year.
        Updates the `relationships` and `financial_history` state.
        """
        # Update the status of terminated and new supplier relationships.
        for partner_id in supply_actions.get('terminate', []):
            self.relationships.loc[(self.relationships['partner_id'] == partner_id) & (self.relationships['relationship_type'] == 'supplier'), 'status'] = 'inactive'
        for partner_id in supply_actions.get('establish', []):
             if not ((self.relationships['partner_id'] == partner_id) & (self.relationships['relationship_type'] == 'supplier')).any():
                new_row = pd.DataFrame([{'partner_id': partner_id, 'partner_name': self.partner_profiles.get(partner_id, {}).get('company_name'), 'relationship_type': 'supplier', 'status': 'active', 'start_year': year}])
                self.relationships = pd.concat([self.relationships, new_row], ignore_index=True)
        # Note: Customer relationship updates are omitted in this snippet but would follow a similar pattern.

        # Get the previous year's financial data for calculations.
        last_year_financials = self.financial_history[-1]
        
        # --- Simplified Financial Simulation ---
        # Calculate the number of active customers and suppliers.
        num_customers = len(market_actions.get('establish', [])) + len(market_actions.get('maintain', []))
        num_suppliers = len(supply_actions.get('establish', [])) + len(supply_actions.get('maintain', []))
        
        # Simulate revenue and costs based on the number of partners.
        new_revenue = num_customers * np.random.uniform(45000, 55000) 
        new_cogs = num_suppliers * np.random.uniform(18000, 22000)
        net_income = new_revenue - new_cogs
        
        # Create the new financial statement for the current year.
        new_statement = {
            "year": year,
            "balance": last_year_financials['balance'] + net_income,
            "revenue": new_revenue,
            "cogs": new_cogs,
            "net_income": net_income,
            "debt_to_equity_ratio": last_year_financials.get('debt_to_equity_ratio', 0.1) * np.random.uniform(0.9, 1.1),
            "net_profit_margin": net_income / new_revenue if new_revenue > 0 else 0,
            "current_ratio": last_year_financials.get('current_ratio', 5.0) * np.random.uniform(0.95, 1.05),
            "sales_growth_yoy": (new_revenue - last_year_financials['revenue']) / last_year_financials['revenue'] if last_year_financials['revenue'] > 0 else 0,
        }
        # Append the new statement to the financial history.
        self.financial_history.append(new_statement)
        logger.info(f"[{self.company_id}] New financial state for year {year} generated.")
        
    def _prepare_public_update(self, year: int) -> Dict:
        """
        Prepares a dictionary of public-facing information to be broadcast to the EconomicCenter.
        """
        # Find the financial data for the specified year.
        financials_for_year = next((f for f in self.financial_history if f['year'] == year), None)
        
        if not financials_for_year:
             logger.error(f"[{self.company_id}] Cannot find financial data for year {year} to publish.")
             return {}

        # Construct the public update object with key financial metrics.
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "main_business": self.main_business,
            "public_financials": {
                "year": financials_for_year["year"],
                "revenue": financials_for_year["revenue"],
                "net_income": financials_for_year["net_income"],
                "debt_to_equity_ratio": financials_for_year["debt_to_equity_ratio"],
                "net_profit_margin": financials_for_year["net_profit_margin"],
                "current_ratio": financials_for_year["current_ratio"],
                "sales_growth_yoy": financials_for_year["sales_growth_yoy"]
            },
            "public_segments": {"main": {"revenue_share": 1.0, "description": self.main_business}}
        }
    
    async def determine_hiring_needs(self) -> Dict[str, any]:
        """
        Uses the LLM to determine hiring needs based on the company's business and financial state.
        """
        current_balance = await self.get_liquid_balance()
        
        # Create a prompt asking for a hiring plan.
        prompt = f"""
Analyze business needs and determine hiring requirements based on our current state.
Our main business is: {self.main_business}.
Current budget: {current_balance:.2f}
Please provide a detailed hiring plan including job titles, descriptions, wage ranges, and number of positions.
Format your response as JSON with the following structure:
{{\"hiring_plan\": [{{\"title\": \"...\", \"description\": \"...\", \"wage_min\": ..., \"wage_max\": ..., \"positions\": ...}}]}}
"""
        try:
            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
            response = await self.llm.atext_request(messages)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Failed to get LLM hiring plan: {e}. Using fallback.")
            # Provide a default hiring plan if the LLM fails.
            return {"hiring_plan": [{"title": "General Worker", "description": "General production support", "wage_min": 15.0, "wage_max": 25.0, "positions": 1}]}

    async def post_job_openings(self):
        """
        Determines hiring needs and posts the corresponding job openings to the LaborMarket.
        """
        if not self.labor_market:
            logger.warning(f"[{self.company_id}] Labor market not configured. Skipping job postings.")
            return

        hiring_plan = await self.determine_hiring_needs()
        
        # Iterate through the hiring plan and publish each job.
        for job_info in hiring_plan.get('hiring_plan', []):
            job = Job(
                title=job_info['title'],
                description=job_info.get('description', ''),
                wage_per_hour=(job_info.get('wage_min', 15) + job_info.get('wage_max', 25)) / 2,
                hours_per_period=40.0,
                positions_available=job_info.get('positions', 1),
                company_id=self.company_id
            )
            await self.labor_market.publish_job.remote(job)
            logger.info(f"[{self.company_id}] Posted job: {job.title} ({job.positions_available} positions)")

    async def evaluate_candidate(self, household_id: str, resume_data: Dict[str, any], current_month: int = None) -> bool:
        """
        Uses the LLM to evaluate a job candidate's resume against open positions.
        Considers company's sales performance and profitability.
        """
        if not self.labor_market:
            logger.warning("Labor market not available for candidate evaluation.")
            return False

        try:
            # Fetch the company's own open jobs from the market.
            all_jobs_ref = await self.labor_market.get_open_jobs.remote()
            company_jobs = [j for j in all_jobs_ref if getattr(j, 'company_id', None) == self.company_id]
            if not company_jobs:
                logger.warning(f"No open jobs found for company {self.company_id} to evaluate candidate.")
                return False
        except Exception as e:
            logger.error(f"Error querying open jobs for candidate evaluation: {e}", exc_info=True)
            return False
        
        # 🔧 获取企业的销售和利润情况
        financial_context = self._get_financial_context_for_hiring(current_month)
        
        # Create a prompt for the LLM to make a hiring decision.
        prompt = f"""
Evaluate the following candidate for our open position, considering our company's current business performance.

Job requirements: {company_jobs[0].description}
Candidate resume: {json.dumps(resume_data)}

{financial_context}

**Important**: If sales are very low or non-existent (especially "no_sales" status), you should be VERY CAUTIOUS about hiring and may reject all applicants to conserve cash flow.

Return a JSON object with a boolean 'decision' and a string 'reasoning'.
Example: {{"decision": true, "reasoning": "The candidate's experience aligns with job requirements and we have healthy sales."}}
Or: {{"decision": false, "reasoning": "Despite good qualifications, we have no recent sales and cannot afford new hires."}}
"""
        try:
            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
            response = await self.llm.atext_request(messages)
            result = json.loads(response)
            logger.info(f"Evaluation for {household_id}: {result.get('reasoning', 'No reasoning provided.')}")
            return result.get('decision', False)
        except Exception as e:
            logger.error(f"Error evaluating candidate {household_id}: {e}")
            return False

    async def make_production_decision(self) -> bool:
        """
        Uses the LLM to create a production plan based on market conditions and budget,
        then executes it.
        """
        if not self.product_market:
            logger.warning(f"[{self.company_id}] Product market not configured. Skipping production.")
            return False

        current_balance = await self.get_liquid_balance()
        market_prices = await self.product_market.get_current_prices.remote()
        
        # Create a prompt for the LLM to devise a production plan.
        prompt = f"""
Analyze current economic conditions and formulate production decisions for our company.
Our main business is: {self.main_business}.
Current budget: {current_balance:.2f}
Current market prices for relevant goods: {market_prices}
Please provide a detailed production plan, including product type and quantity.
Format your response as JSON with the following structure:
{{\"production_plan\": [{{\"product_type\": \"...\", \"quantity\": ...}}]}}
"""
        messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        
        try:
            # Request plan from LLM and parse it.
            response = await self.llm.atext_request(messages)
            plan = json.loads(response)
            results = []
            # Execute each item in the production plan.
            for item in plan.get('production_plan', []):
                product = Product(
                    name=item['product_type'],
                    quantity=item['quantity'],
                    price=market_prices.get(item['product_type'], 10.0) # Use market price or a default
                )
                success = await self.produce_and_list_product(product)
                results.append(success)
            return all(results)
        except Exception as e:
            # If the LLM fails, use a simple fallback production plan.
            logger.warning(f"Failed to parse LLM production plan: {e}. Attempting fallback.")
            return await self._simple_production_fallback()
    
    async def get_liquid_balance(self) -> float:
        """
        Retrieves the firm's current liquid balance, preferably from the EconomicCenter,
        with a fallback to its internal records.
        """
        if self.economic_center:
            try:
                # Query the authoritative source first.
                return await self.economic_center.query_balance.remote(self.company_id)
            except Exception as e:
                logger.warning(f"Failed to query balance from EconomicCenter: {e}. Falling back to internal history.")
        
        # Fallback to the last known balance from internal financial history.
        return self.financial_history[-1]['balance'] if self.financial_history else 0.0
    
    def record_income(self, amount: float, description: str = "", month: Optional[int] = None):
        """记录企业收入"""
        self.total_income += amount
        if month is not None:
            self.monthly_financials[month]["income"] += amount
        # logger.info(f"Firm {self.company_id} recorded income: ${amount:.2f} - {description}")
    
    def record_expense(self, amount: float, description: str = "", month: Optional[int] = None):
        """记录企业支出"""
        self.total_expenses += amount
        if month is not None:
            self.monthly_financials[month]["expenses"] += amount
        # logger.info(f"Firm {self.company_id} recorded expense: ${amount:.2f} - {description}")
    
    def get_financial_summary(self) -> Dict[str, float]:
        """获取企业财务摘要"""
        return {
            "total_income": self.total_income,
            "total_expenses": self.total_expenses,
            "net_profit": self.total_income - self.total_expenses
        }
    
    def get_monthly_financial_summary(self, month: int) -> Dict[str, float]:
        """获取企业指定月份的财务摘要"""
        monthly_data = self.monthly_financials[month]
        return {
            "monthly_income": monthly_data["income"],
            "monthly_expenses": monthly_data["expenses"],
            "monthly_profit": monthly_data["income"] - monthly_data["expenses"]
        }
    
    def get_all_monthly_financials(self) -> Dict[int, Dict[str, float]]:
        """获取企业所有月份的财务数据"""
        result = {}
        for month, data in self.monthly_financials.items():
            result[month] = {
                "monthly_income": data["income"],
                "monthly_expenses": data["expenses"],
                "monthly_profit": data["income"] - data["expenses"]
            }
        return result
    

    async def produce_and_list_product(self, product: Product) -> bool:
        """
        Simulates producing a product and listing it for sale on the ProductMarket.
        """
        await self.product_market.publish_product.remote(product)
        logger.info(f"[{self.company_id}] Produced and listed {product.quantity} of {product.name}.")
        return True

    async def _simple_production_fallback(self) -> bool:
        """
        A simple, non-LLM-based production plan to be used if the main method fails.
        """
        logger.info(f"[{self.company_id}] Executing simple production fallback.")
        product = Product(
            name=f"{self.main_business.replace(' ', '_')}_Component",
            quantity=10,
            price=50.0 
        )
        return await self.produce_and_list_product(product)

    async def run_operations(self):
        """
        A legacy or standalone method to run operational tasks.
        Note: This functionality is now integrated into the main `run_cycle` method.
        """
        logger.info(f"Running standalone operations for firm {self.company_id}. This is now part of the annual 'run_cycle'.")
        await self.post_job_openings()
        await self.make_production_decision()

    # ==================== Simple Employee Management ====================
    
    def get_employees(self) -> int:
        """获取当前员工数量"""
        return self.employees
    
    def set_employees(self, count: int):
        """设置员工数量"""
        self.employees = max(0, count)
        logger.info(f"[{self.company_id}] 员工数量设置为: {self.employees}")
    
    def add_employees(self, count: int):
        """增加员工数量"""
        self.employees += max(0, count)
        logger.info(f"[{self.company_id}] 员工数量增加到: {self.employees}")
    
    def remove_employees(self, count: int):
        """减少员工数量"""
        self.employees = max(0, self.employees - count)
        logger.info(f"[{self.company_id}] 员工数量减少到: {self.employees}")
    
    def add_employee(self, employee_data: Dict):
        """
        添加员工到企业员工列表
        
        Args:
            employee_data: 员工信息字典，包含household_id, lh_type, job_title等
        """
        household_id = str(employee_data.get("household_id", ""))
        lh_type = employee_data.get("lh_type", "head")
        employee_key = f"{household_id}_{lh_type}"
        
        # 检查是否已经存在
        if employee_key in self.employee_lookup:
            logger.warning(f"[{self.company_id}] 员工 {employee_key} 已存在，跳过添加")
            return False
        
        # 添加员工信息
        employee_info = {
            "household_id": household_id,
            "lh_type": lh_type,
            "job_title": employee_data.get("job_title", ""),
            "job_soc": employee_data.get("job_soc", ""),
            "wage_per_hour": employee_data.get("wage_per_hour", 0.0),
            "hours_per_period": employee_data.get("hours_per_period", 40),
            "skills": employee_data.get("skills", {}),
            "abilities": employee_data.get("abilities", {}),
            "hire_date": employee_data.get("hire_date", ""),
            "status": "active"  # active, resigned, dismissed
        }
        
        self.employee_list.append(employee_info)
        self.employee_lookup[employee_key] = employee_info
        self.employees += 1
        
        logger.info(f"[{self.company_id}] 添加员工: {employee_key} - {employee_info['job_title']} @ ${employee_info['wage_per_hour']:.2f}/小时")
        return True
    
    def remove_employee(self, household_id: str, lh_type: str, reason: str = "resigned"):
        """
        从企业员工列表中移除员工
        
        Args:
            household_id: 家庭ID
            lh_type: 劳动力类型 (head/spouse)
            reason: 离职原因 (resigned/dismissed)
        """
        employee_key = f"{household_id}_{lh_type}"
        
        if employee_key not in self.employee_lookup:
            logger.warning(f"[{self.company_id}] 员工 {employee_key} 不存在，无法移除")
            return False
        
        # 更新员工状态
        employee_info = self.employee_lookup[employee_key]
        employee_info["status"] = reason
        employee_info["resign_date"] = ""  # 可以添加具体日期
        
        # 从查找字典中移除
        del self.employee_lookup[employee_key]
        
        # 从列表中移除
        self.employee_list = [emp for emp in self.employee_list if emp.get("household_id") != household_id or emp.get("lh_type") != lh_type]
        
        self.employees = max(0, self.employees - 1)
        
        logger.info(f"[{self.company_id}] 移除员工: {employee_key} - 原因: {reason}")
        return True
    
    def get_employee(self, household_id: str, lh_type: str) -> Optional[Dict]:
        """获取员工信息"""
        employee_key = f"{household_id}_{lh_type}"
        return self.employee_lookup.get(employee_key)
    
    def get_all_employees(self) -> List[Dict]:
        """获取所有活跃员工列表"""
        return [emp for emp in self.employee_list if emp.get("status") == "active"]
    
    def get_employee_count_by_role(self) -> Dict[str, int]:
        """按角色统计员工数量"""
        role_count = {"head": 0, "spouse": 0}
        for emp in self.employee_list:
            if emp.get("status") == "active":
                role = emp.get("lh_type", "head")
                role_count[role] += 1
        return role_count