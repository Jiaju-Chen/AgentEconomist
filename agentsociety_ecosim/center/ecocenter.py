from typing import Literal, Dict, List, Any
from uuid import uuid4
from collections import defaultdict
import copy
from .model import * 
import ray
from .utils import safe_call
from agentsociety_ecosim.utils.log_utils import setup_global_logger
from agentsociety_ecosim.utils.product_attribute_loader import inject_product_attributes
from agentsociety_ecosim.center.model import FirmInnovationConfig, FirmInnovationEvent
import numpy as np
import random
import os 
from dotenv import load_dotenv
load_dotenv()
logger = setup_global_logger(name="economic_center")

@ray.remote(num_cpus=8)
class EconomicCenter:
    def __init__(self, income_tax_rate: float = 0.225, vat_rate: float = 0.08, corporate_tax_rate: float = 0.21, category_profit_margins: Dict[str, float] = None):
        """
        Initialize EconomicCenter with tax rates
        
        Args:
            income_tax_rate: 个人所得税率，默认22.5%
            vat_rate: 消费税率（增值税），默认8%
            corporate_tax_rate: 企业所得税率，默认21%
        """
        # 税率配置
        self.income_tax_rate = income_tax_rate
        self.vat_rate = vat_rate
        self.corporate_tax_rate = corporate_tax_rate
        
        # 💰 商品毛利率配置（基于Daily Category的12个大类）
        # 由GPT-5生成，基于行业实际情况和市场竞争程度
        # 🔧 修复：如果传入 None，初始化默认配置（类似 SimulationConfig.__post_init__）
        if category_profit_margins is None:
            print('使用默认毛利率')
            self.category_profit_margins = {
                "Beverages": 25.0,                              # 饮料
                "Confectionery and Snacks": 32.0,               # 糖果和零食
                "Dairy Products": 15.0,                         # 乳制品
                "Furniture and Home Furnishing": 30.0,          # 家具和家居装饰
                "Garden and Outdoor": 28.0,                     # 园艺和户外
                "Grains and Bakery": 18.0,                      # 谷物和烘焙
                "Household Appliances and Equipment": 30.0,     # 家用电器和设备
                "Meat and Seafood": 16.0,                       # 肉类和海鲜
                "Personal Care and Cleaning": 40.0,            # 个人护理和清洁
                "Pharmaceuticals and Health": 45.0,            # 药品和健康
                "Retail and Stores": 25.0,                      # 零售和商店
                "Sugars, Oils, and Seasonings": 20.0,           # 糖类、油类和调料
            }
        else:
            self.category_profit_margins = category_profit_margins

        # Save assets for different agents
        self.ledger: Dict[str, Ledger] = defaultdict(Ledger) 
        self.products: Dict[str, List[Product]] = defaultdict(list)
        self.laborhour: Dict[str, List[LaborHour]] = defaultdict(list)

        # Save IDs for different agents
        self.government_id: List[str] = []  # government ID
        self.household_id: List[str] = []  #  household ID
        self.company_id: List[str] = []  #  firm ID
        self.bank_id: List[str] = []  #  bank ID

        self.middleware = MiddlewareRegistry()
        self.tx_history: List[Transaction] = []  # Store transaction history
        self.wage_history: List[Wage] = []
        self.firm_financials: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total_income": 0.0, "total_expenses": 0.0})  # 企业财务记录
        self.firm_monthly_financials: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"income": 0.0, "expenses": 0.0}))  # 企业月度财务记录
        self.firm_production_stats: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"base_production": 0.0, "labor_production": 0.0}))  # 企业月度生产统计
        self.redistribution_record_per_person:Dict[int, float] = defaultdict(float)
        # 创新系统数据结构
        self.firm_innovation_strategy: Dict[str, str] = {}  # {company_id: "encouraged" or "suppressed"}
        self.firm_research_share: List[Dict[str, [float, int]]] = []  # [company_id: [research_share, month]] 研发投入比例
        
        # 创新系统数据结构
        self.firm_innovation_config: Dict[str, FirmInnovationConfig] = {}  # {company_id: innovation_config}
        self.firm_innovation_events: List[FirmInnovationEvent] = []  # [company_id: innovation_events, month: month] 创新事件历史记录
        print(f"EconomicCenter initialized with tax rates: income={income_tax_rate:.1%}, vat={vat_rate:.1%}, corporate={corporate_tax_rate:.1%}")

    @safe_call("EconomicCenter init_agent_ledger", "warning")
    def init_agent_ledger(self, agent_id: str, initial_amount: float = 0.0):
        """
        Initialize a ledger for an agent with a given initial amount.
        If the agent already exists, it will not overwrite the existing ledger.
        """
        if agent_id not in self.ledger:
            ledger = Ledger.create(agent_id, amount=initial_amount)
            self.ledger[agent_id] = ledger
            # logger.info(f"Initialized ledger for agent {agent_id} with amount {initial_amount}")
    
    @safe_call("EconomicCenter init_agent_product", "warning")
    def init_agent_product(self, agent_id: str, product: Optional[Product]=None):
        """
        Initialize a product for an agent. If the product already exists, it will merge the amounts.
        """
        if agent_id not in self.products:
            # print(f"Initialized product for agent {agent_id}")
            self.products[agent_id] = []
        
        if product:
            self._add_or_merge_product(agent_id, product)
            # logger.info(f"Initialized product {product.name} for agent {agent_id} with amount {product.amount}")

    @safe_call("EconomicCenter init_agent_labor", "warning")
    def init_agent_labor(self, agent_id:str, labor:[LaborHour]=[]):
        """
        Initialize the labor hour for an agent. 
        """  
        if agent_id not in self.laborhour:
            self.laborhour[agent_id] = []
        if labor:
            self.laborhour[agent_id] = labor

    def register_id(self, agent_id: str, agent_type: Literal['government', 'household', 'firm', 'bank']):
        """
        Register an agent ID based on its type.
        """ 
        if agent_type == 'government':
            self.government_id.append(agent_id)
        elif agent_type == 'household':
            self.household_id.append(agent_id)
        elif agent_type == 'firm':
            self.company_id.append(agent_id)
        elif agent_type == 'bank':
            self.bank_id.append(agent_id)

    def query_all_products(self):
        return self.products

    def query_all_tx(self):
        return self.tx_history
    
    def query_exsiting_agents(self, agent_type: Literal['government', 'household', 'firm']) -> List[str]:
        """
        Query existing agents based on their type.
        """
        if agent_type == 'government':
            return self.government_id
        elif agent_type == 'household':
            return self.household_id
        elif agent_type == 'firm':
            return self.company_id
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
    # query interface
    def query_balance(self, agent_id: str) -> float:
        if agent_id in self.ledger:
            return self.ledger[agent_id].amount
        else:
            return 0.0

    def query_redistribution_record_per_person(self, month: int) -> float:
        return self.redistribution_record_per_person[month]
    
    def query_products(self, agent_id: str) -> List[Product]:
        return self.products[agent_id]
    
    def query_price(self, agent_id: str, product_id: str) -> float:
        for product in self.products[agent_id]:
            if product.product_id == product_id:
                return product.price
        return 0.0
    
    def query_financial_summary(self, agent_id: str) -> Dict[str, float]:
        """查询代理的财务摘要：余额、总收入、总支出（企业适用）"""
        result = {}
        
        if agent_id in self.ledger:
            result["balance"] = self.ledger[agent_id].amount
        else:
            result["balance"] = 0.0
        
        # 如果是企业，添加收支记录
        if agent_id in self.firm_financials:
            result.update(self.firm_financials[agent_id])
            result["net_profit"] = result.get("total_income", 0.0) - result.get("total_expenses", 0.0)
        
        result['total_income'] = self.firm_financials[agent_id].get("total_income", 0.0)
        result['total_expenses'] = self.firm_financials[agent_id].get("total_expenses", 0.0)
        return result
    
    def record_firm_income(self, company_id: str, amount: float):
        """记录企业收入"""
        self.firm_financials[company_id]["total_income"] += amount
        
    def record_firm_expense(self, company_id: str, amount: float):
        """记录企业支出"""
        self.firm_financials[company_id]["total_expenses"] += amount
    
    def record_firm_monthly_income(self, company_id: str, month: int, amount: float):
        """记录企业月度收入"""
        self.firm_monthly_financials[company_id][month]["income"] += amount
        
    def record_firm_monthly_expense(self, company_id: str, month: int, amount: float):
        """记录企业月度支出"""
        self.firm_monthly_financials[company_id][month]["expenses"] += amount
    
    def query_firm_monthly_financials(self, company_id: str, month: int) -> Dict[str, float]:
        """查询企业指定月份的财务数据"""
        if company_id in self.firm_monthly_financials and month in self.firm_monthly_financials[company_id]:
            monthly_data = self.firm_monthly_financials[company_id][month]
            return {
                "monthly_income": monthly_data["income"],
                "monthly_expenses": monthly_data["expenses"],
                "monthly_profit": monthly_data["income"] - monthly_data["expenses"]
            }
        return {"monthly_income": 0.0, "monthly_expenses": 0.0, "monthly_profit": 0.0}
    
    def query_firm_production_stats(self, company_id: str, month: int) -> Dict[str, float]:
        """查询企业指定月份的生产统计数据"""
        if company_id in self.firm_production_stats and month in self.firm_production_stats[company_id]:
            production_data = self.firm_production_stats[company_id][month]
            return {
                "base_production": production_data["base_production"],
                "labor_production": production_data["labor_production"],
                "total_production": production_data["base_production"] + production_data["labor_production"]
            }
        return {"base_production": 0.0, "labor_production": 0.0, "total_production": 0.0}
    
    def query_firm_all_monthly_financials(self, company_id: str) -> Dict[int, Dict[str, float]]:
        """查询企业所有月份的财务数据"""
        result = {}
        if company_id in self.firm_monthly_financials:
            for month, data in self.firm_monthly_financials[company_id].items():
                result[month] = {
                    "monthly_income": data["income"],
                    "monthly_expenses": data["expenses"],
                    "monthly_profit": data["income"] - data["expenses"]
                }
        return result

    def query_income(self, agent_id: str, month: int) -> float:
        total_wage = 0.0
        for wage in self.wage_history:
            if wage.agent_id == agent_id and wage.month == month:
                total_wage += wage.amount
        return total_wage


    def query_labor(self, agent_id: str) -> List[LaborHour]:
        return self.laborhour[agent_id]

    def deposit_funds(self, agent_id: str, amount: float):
        self.ledger[agent_id].amount += amount
    
    def update_balance(self, agent_id: str, amount: float):
        """
        更新代理的余额（可以是正数或负数）
        
        Args:
            agent_id: 代理ID
            amount: 变动金额（正数增加，负数减少）
        """
        if agent_id not in self.ledger:
            self.ledger[agent_id] = Ledger()
        self.ledger[agent_id].amount += amount
    
    def consume_product_inventory(self, company_id: str, product_id: str, quantity: float) -> bool:
        """
        减少企业商品库存
        
        Args:
            company_id: 企业ID
            product_id: 商品ID
            quantity: 消耗数量
            
        Returns:
            bool: 是否成功消耗
        """
        if company_id not in self.products:
            logger.warning(f"企业 {company_id} 没有产品库存")
            return False
        
        for product in self.products[company_id]:
            if product.product_id == product_id:
                if product.amount >= quantity:
                    product.amount -= quantity
                    # logger.info(f"企业 {company_id} 商品 {product_id} 消耗 {quantity} 单位，剩余 {product.amount}")
                    return True
                else:
                    logger.warning(f"企业 {company_id} 商品 {product_id} 库存不足: {product.amount} < {quantity}")
                    return False
        
        logger.warning(f"企业 {company_id} 没有找到商品 {product_id}")
        return False
    
    def register_product(self, agent_id: str, product: Product):
        """
        Register a product for an agent. If the product already exists, it will merge the amounts.
        """
        if agent_id not in self.products:
            # print(f"Initialized product for agent {agent_id}")
            self.products[agent_id] = []
        
        self._add_or_merge_product(agent_id, product, product.amount)
        # logger.info(f"Registered product {product.name} for agent {agent_id} with amount {product.amount}")

    def _add_or_merge_product(self, agent_id:str, product: Product, quantity: float = 1.0):

        product.owner_id = agent_id
        product.amount = quantity
        for existing_product in self.products[agent_id]:
            if existing_product.product_id == product.product_id:
                existing_product.amount += quantity
                return
        self.products[agent_id].append(product)

    def _check_and_reserve_inventory(self, seller_id: str, product: Product, quantity: float) -> bool:
        """
        检查并预留库存，确保原子性购买操作
        返回True表示库存充足且已预留，False表示库存不足
        """
        if seller_id not in self.products:
            return False
        
        for existing_product in self.products[seller_id]:
            if existing_product.product_id == product.product_id:
                if existing_product.amount >= quantity:
                    # 库存充足，可以购买
                    return True
                else:
                    # 库存不足
                    return False
        
        # 商品不存在
        return False
    
    def _get_profit_margin(self, category: str) -> float:
        """
        根据商品大类获取毛利率（用于利润计算）
        
        Args:
            category: 商品大类名称（daily_cate）
            
        Returns:
            毛利率（百分比，如25.0表示25%）
        """
        # 如果配置中有该大类，返回配置的毛利率
        if category in self.category_profit_margins:
            return self.category_profit_margins[category]
        
        # 如果找不到该大类，返回默认毛利率25%
        logger.warning(f"未找到大类 '{category}' 的毛利率配置，使用默认值25%")
        return 25.0
    
    def _reduce_or_remove_product(self, agent_id: str, product: Product, quantity: float = 1.0):
        """
        减少商品库存（在确认库存充足后调用）
        """
        for existing_product in self.products[agent_id]:
            if existing_product.product_id == product.product_id:
                # 再次检查库存（双重保险）
                if existing_product.amount < quantity:
                    raise ValueError(f"库存不足: 需要 {quantity}，但只有 {existing_product.amount}")
                
                existing_product.amount -= quantity
                return
        raise ValueError("Asset not found or insufficient amount to reduce.")
    
    # register_middleware
    def register_middleware(self, tx_type: str, middleware_fn: Callable[[Transaction, Dict[str, float]], None], tag: Optional[str] = None):
        if tag:
            self.middleware.register(tx_type, middleware_fn, tag)
        else:
            self.middleware.register(tx_type, middleware_fn) 
    
    def process_batch_purchases(self, month: int, buyer_id: str, purchase_list: List[Dict]) -> List[Optional[str]]:
        """
        批量处理购买，减少Ray远程调用次数
        
        Args:
            month: 当前月份
            buyer_id: 购买者ID
            purchase_list: 购买列表，每项包含 {'seller_id', 'product', 'quantity'}
        
        Returns:
            交易ID列表（成功返回tx_id，失败返回None）
        """
        results = []
        for purchase in purchase_list:
            seller_id = purchase['seller_id']
            product = purchase['product']
            quantity = purchase.get('quantity', 1.0)
            tx_result = self.process_purchase(month, buyer_id, seller_id, product, quantity)
            
            # 🔧 处理返回值：Transaction对象或False
            if tx_result and hasattr(tx_result, 'id'):
                results.append(tx_result.id)  # 返回交易ID
            else:
                results.append(None)  # 购买失败
        return results
    
    def process_purchase(self, month: int, buyer_id: str, seller_id: str, product: Product, quantity: float = 1.0) -> Optional[str]:
        # 计算总费用：标价 + 消费税
        base_price = product.price * quantity
        total_cost_with_tax = base_price * (1 + self.vat_rate)  # 家庭支付标价+消费税
        
        # 检查家庭余额是否足够支付含税价格
        if self.ledger[buyer_id].amount < total_cost_with_tax:
            return False

        # 🔒 关键修复：在支付前先检查并预留库存
        if not self._check_and_reserve_inventory(seller_id, product, quantity):
            # 获取当前库存用于调试
            current_stock = 0
            for pro in self.products[seller_id]:
                if pro.product_id == product.product_id:
                    current_stock = pro.amount
                    break
            logger.warning(f"库存不足，购买失败: {product.name} 需要 {quantity}，但库存不足, 剩余库存: {current_stock}")
            return False

        # 家庭支付含税价格
        self.ledger[buyer_id].amount -= total_cost_with_tax

        # 创建消费税交易记录（税收部分）
        tax_amount = base_price * self.vat_rate
        tax_tx = Transaction(
            id=str(uuid4()),
            sender_id=buyer_id,
            receiver_id="gov_main_simulation",  # 固定政府ID
            amount=tax_amount,
            type='consume_tax',
            month=month
        )
        self.tx_history.append(tax_tx)
        
        # 政府收取消费税
        self.ledger["gov_main_simulation"].amount += tax_amount

        # 创建购买交易记录（企业收入部分）
        purchase_tx = Transaction(
            id=str(uuid4()),
            sender_id=buyer_id,
            receiver_id=seller_id,
            amount=total_cost_with_tax,  # 家庭实际支出
            assets=[product],
            type='purchase',
            month=month
        )
        self.tx_history.append(purchase_tx)

        # 💰 企业收入、成本和利润计算
        # 1. 企业收到销售收入（税前）
        revenue = base_price
        self.ledger[seller_id].amount += revenue
        
        # 记录企业收入（经济中心层面）
        self.record_firm_income(seller_id, revenue)
        # 记录企业月度收入
        self.record_firm_monthly_income(seller_id, month, revenue)
        
        # 2. 根据商品类别和毛利率计算成本和利润
        # 毛利率 = (销售收入 - 成本) / 销售收入 × 100%
        # => 成本 = 销售收入 × (1 - 毛利率)
        # => 毛利润 = 销售收入 × 毛利率
        config = self.firm_innovation_config.get(seller_id)
        if not config or config.profit_margin is None:
            # 如果没有创新配置，使用默认毛利率
            product_category = product.classification if hasattr(product, 'classification') else "Unknown"
            profit_margin = self.category_profit_margins.get(product_category, 25.0)
        else:
            profit_margin = config.profit_margin
        margin_rate = profit_margin / 100.0  # 转换为小数
        cost = revenue * (1 - margin_rate)  # 成本
        gross_profit = revenue * margin_rate  # 毛利润
        # 3. 记录成本支出
        # 企业支付成本（从账户扣除）
        if self.ledger[seller_id].amount >= cost:
            self.ledger[seller_id].amount -= cost
            # 记录企业成本支出（经济中心层面）
            self.record_firm_expense(seller_id, cost)
            # 记录企业月度支出
            self.record_firm_monthly_expense(seller_id, month, cost)
        else:
            logger.warning(f"企业 {seller_id} 余额不足以支付成本: ${self.ledger[seller_id].amount:.2f} < ${cost:.2f}")
        
        # 4. 企业需要缴纳企业所得税（基于毛利润）
        # 企业所得税 = 毛利润 × 企业所得税率
        corporate_tax = gross_profit * self.corporate_tax_rate
        
        # 企业支付所得税
        if self.ledger[seller_id].amount >= corporate_tax:
            self.ledger[seller_id].amount -= corporate_tax
            # 记录企业支出（经济中心层面）
            self.record_firm_expense(seller_id, corporate_tax)
            # 记录企业月度支出
            self.record_firm_monthly_expense(seller_id, month, corporate_tax)
        else:
            print(f"Warning: Company {seller_id} insufficient balance for corporate tax: ${self.ledger[seller_id].amount:.2f} < ${corporate_tax:.2f}")
            return purchase_tx.id
        
        # 政府收取企业所得税
        self.ledger["gov_main_simulation"].amount += corporate_tax
        
        # 记录企业所得税交易
        corp_tax_tx = Transaction(
            id=str(uuid4()),
            sender_id=seller_id,
            receiver_id="gov_main_simulation",
            amount=corporate_tax,
            type='corporate_tax',
            month=month
        )
        self.tx_history.append(corp_tax_tx)
        
        # 商品转移
        try:
            self._add_or_merge_product(buyer_id, product, quantity)
            self._reduce_or_remove_product(seller_id, product, quantity) 
        except Exception as e:
            print(f"Warning: Failed to process purchase: {e}")
            return False
        
        return purchase_tx

    def process_labor(self, month: int, wage_hour: float, household_id: str, company_id: Optional[str] = None) -> str:
        # 计算税前工资
        gross_wage = wage_hour * 160
        
        # 计算个人所得税
        income_tax = gross_wage * self.income_tax_rate
        net_wage = gross_wage - income_tax  # 税后工资
        
        # 检查企业余额
        if company_id and self.ledger[company_id].amount < gross_wage:
            print(f"Warning: Company {company_id} insufficient balance for wage payment: ${self.ledger[company_id].amount:.2f} < ${gross_wage:.2f}")
            return None

        # 创建工资支付交易记录
        wage_tx = Transaction(
            id=str(uuid4()),
            sender_id=company_id,
            receiver_id=household_id,
            amount=net_wage,  # 家庭收到税后工资
            type='labor_payment',
            month=month,
        )
        self.tx_history.append(wage_tx)
        
        # 创建个人所得税交易记录
        tax_tx = Transaction(
            id=str(uuid4()),
            sender_id=household_id,
            receiver_id="gov_main_simulation",
            amount=income_tax,
            type='labor_tax',
            month=month,
        )
        self.tx_history.append(tax_tx)

        # 更新账本
        self.ledger[household_id].amount += net_wage  # 家庭收到税后工资
        self.ledger["gov_main_simulation"].amount += income_tax  # 政府收到个人所得税
        
        # 企业支出工资
        if company_id:
            self.ledger[company_id].amount -= gross_wage
            # 记录企业支出（经济中心层面）
            self.record_firm_expense(company_id, gross_wage)
            # 记录企业月度支出
            self.record_firm_monthly_expense(company_id, month, gross_wage)

        # 记录工资历史（记录税前工资）
        self.wage_history.append(Wage.create(household_id, gross_wage, month))
        # print(f"Month {month} Processed labor payment: ${gross_wage:.2f} gross (${net_wage:.2f} net, ${income_tax:.2f} tax) from {company_id} to {household_id}")
        return wage_tx.id

    def compute_household_settlement(self, household_id: str):
        """
        Process household settlement, including asset and labor hour settlement.
        计算家庭累积收入和支出
        """
        # breakpoint()

        total_income = 0
        total_expense = 0
        for tx in self.tx_history:
            if tx.type == 'purchase' and tx.sender_id == household_id:
                total_expense += tx.amount

            elif tx.type == 'service' and tx.sender_id == household_id:
                total_expense += tx.amount  # 服务费用直接计入支出，不需要税收调整

            elif tx.type == 'labor_payment' and tx.receiver_id == household_id:
                total_income += tx.amount

            elif tx.type == 'redistribution' and tx.receiver_id == household_id:
                total_income += tx.amount

            elif tx.type == 'interest' and tx.receiver_id == household_id:
                total_income += tx.amount

        return total_income, total_expense

    def compute_household_monthly_stats(self, household_id: str, target_month: int = None):
        """
        计算家庭月度收入和支出统计(收入不统计再分配)
        如果不指定target_month，返回所有月份的统计
        """
        monthly_income = 0
        monthly_expense = 0
        
        month = target_month 


        for tx in self.tx_history:
            if tx.type == 'purchase' and tx.sender_id == household_id and tx.month == month:
                monthly_expense += tx.amount

            elif tx.type == 'service' and tx.sender_id == household_id and tx.month == month:
                monthly_expense += tx.amount

            elif tx.type == 'labor_payment' and tx.receiver_id == household_id and tx.month == month:
                monthly_income += tx.amount

            elif tx.type == 'interest' and tx.receiver_id == household_id and tx.month == month:
                monthly_income += tx.amount

            # elif tx.type == 'redistribution' and tx.receiver_id == household_id and tx.month == month:
            #     monthly_income += tx.amount

        return monthly_income, monthly_expense, self.ledger[household_id].amount
    
    def get_monthly_tax_collection(self, month: int) -> Dict[str, float]:
        """
        获取指定月份的税收收入统计
        
        Args:
            month: 目标月份
            
        Returns:
            Dict: 各类税收收入统计
        """
        tax_summary = {
            "consume_tax": 0.0,
            "labor_tax": 0.0, 
            "corporate_tax": 0.0,
            "total_tax": 0.0
        }
        
        for tx in self.tx_history:
            if tx.month == month and tx.receiver_id == "gov_main_simulation":
                if tx.type == 'consume_tax':
                    tax_summary["consume_tax"] += tx.amount
                elif tx.type == 'labor_tax':
                    tax_summary["labor_tax"] += tx.amount
                elif tx.type == 'corporate_tax':
                    tax_summary["corporate_tax"] += tx.amount
        
        tax_summary["total_tax"] = (tax_summary["consume_tax"] + 
                                   tax_summary["labor_tax"] + 
                                   tax_summary["corporate_tax"])
        
        return tax_summary
    

    
    async def redistribute_monthly_taxes(self, month: int, strategy: str = "equal", 
                                       poverty_weight: float = 0.3, 
                                       unemployment_weight: float = 0.2, 
                                       family_size_weight: float = 0.1) -> Dict[str, float]:
        """
        税收再分配：支持多种分配策略
        
        Args:
            month: 当前月份
            strategy: 分配策略 ("none", "equal", "income_proportional", "poverty_focused", "unemployment_focused", "family_size", "mixed")
            poverty_weight: 贫困权重 (0-1)
            unemployment_weight: 失业权重 (0-1) 
            family_size_weight: 家庭规模权重 (0-1)
            
        Returns:
            Dict: 再分配结果统计
        """
        # 如果策略为 "none"，不进行再分配
        if strategy == "none":
            tax_summary = self.get_monthly_tax_collection(month)
            return {
                "total_redistributed": 0.0, 
                "recipients": 0, 
                "per_person": 0.0,
                "total_tax_collected": tax_summary["total_tax"],
                "tax_breakdown": tax_summary
            }
        
        # 获取当月税收总额
        tax_summary = self.get_monthly_tax_collection(month)
        total_tax = tax_summary["total_tax"]
        
        if total_tax <= 0:
            print(f"Month {month}: No tax revenue to redistribute")
            return {"total_redistributed": 0.0, "recipients": 0, "per_person": 0.0}
        
        # 获取所有有劳动力的家庭ID（基于现有的laborhour字典）
        all_workers = [household_id for household_id, labor_hours in self.laborhour.items() 
                      if labor_hours]  # 只包括有劳动力的家庭
        if not all_workers:
            print(f"Month {month}: No households with labor hours found for tax redistribution")
            return {"total_redistributed": 0.0, "recipients": 0, "per_person": 0.0}
        
        # 根据策略计算分配金额
        household_allocations = self._calculate_redistribution_allocations(
            all_workers, total_tax, strategy, poverty_weight, unemployment_weight, family_size_weight, month
        )
        
        total_redistributed = 0.0
        successful_redistributions = 0
        
        # 执行再分配
        for household_id, allocation_amount in household_allocations.items():
            try:
                if allocation_amount > 0:
                    # 政府向家庭转账
                    tx_id = self.add_redistribution_tx(
                        month=month,
                        sender_id="gov_main_simulation",
                        receiver_id=household_id,
                        amount=allocation_amount,
                    )
                    
                    total_redistributed += allocation_amount
                    successful_redistributions += 1
        
            except Exception as e:
                print(f"Failed to redistribute to household {household_id}: {e}")

        # 计算平均分配金额（用于记录）
        avg_allocation = total_redistributed / successful_redistributions if successful_redistributions > 0 else 0
        
        result = {
            "total_tax_collected": total_tax,
            "total_redistributed": total_redistributed,
            "recipients": successful_redistributions,
            "per_person": avg_allocation,
            "strategy": strategy,
            "tax_breakdown": tax_summary
        }
        self.redistribution_record_per_person[month] = avg_allocation

        print(f"Month {month} Tax Redistribution ({strategy}):")
        print(f"  Total tax collected: ${total_tax:.2f}")
        print(f"  Redistributed to {successful_redistributions} households: ${total_redistributed:.2f}")
        print(f"  Average per household: ${avg_allocation:.2f}")
        
        return result

    def _calculate_redistribution_allocations(self, all_workers: List[str], total_tax: float, 
                                           strategy: str, poverty_weight: float, 
                                           unemployment_weight: float, family_size_weight: float, 
                                           month: int) -> Dict[str, float]:
        """
        根据策略计算每个家庭的分配金额
        
        Args:
            all_workers: 所有有劳动力的家庭ID列表
            total_tax: 税收总额
            strategy: 分配策略
            poverty_weight: 贫困权重
            unemployment_weight: 失业权重
            family_size_weight: 家庭规模权重
            month: 当前月份
            
        Returns:
            Dict[str, float]: 家庭ID到分配金额的映射
        """
        if strategy == "equal":
            return self._equal_allocation(all_workers, total_tax)
        elif strategy == "income_proportional":
            return self._income_proportional_allocation(all_workers, total_tax, month)
        elif strategy == "poverty_focused":
            return self._poverty_focused_allocation(all_workers, total_tax, month)
        elif strategy == "unemployment_focused":
            return self._unemployment_focused_allocation(all_workers, total_tax, month)
        elif strategy == "family_size":
            return self._family_size_allocation(all_workers, total_tax)
        elif strategy == "mixed":
            return self._mixed_allocation(all_workers, total_tax, poverty_weight, 
                                        unemployment_weight, family_size_weight, month)
        else:
            print(f"Unknown redistribution strategy: {strategy}, using equal allocation")
            return self._equal_allocation(all_workers, total_tax)

    def _equal_allocation(self, all_workers: List[str], total_tax: float) -> Dict[str, float]:
        """平均分配策略"""
        amount_per_household = total_tax / len(all_workers)
        return {household_id: amount_per_household for household_id in all_workers}

    def _income_proportional_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """按收入比例分配策略"""
        household_incomes = {}
        total_income = 0.0
        
        for household_id in all_workers:
            monthly_income, _, _ = self.compute_household_monthly_stats(household_id, month)
            household_incomes[household_id] = monthly_income
            total_income += monthly_income
        
        if total_income <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = household_incomes[household_id] / total_income
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _poverty_focused_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """贫困导向分配策略（收入越低分配越多）"""
        household_incomes = {}
        household_balances = {}
        
        for household_id in all_workers:
            monthly_income, _, balance = self.compute_household_monthly_stats(household_id, month)
            household_incomes[household_id] = monthly_income
            household_balances[household_id] = balance
        
        if not household_incomes:
            return self._equal_allocation(all_workers, total_tax)
        
        max_income = max(household_incomes.values())
        min_income = min(household_incomes.values())
        max_balance = max(household_balances.values()) if household_balances else 0.0
        min_balance = min(household_balances.values()) if household_balances else 0.0
        
        # 若收入与存款都无差异，则退化为均分
        if max_income == min_income and max_balance == min_balance:
            return self._equal_allocation(all_workers, total_tax)
        
        # 计算贫困权重（收入越低、存款越低权重越高）
        # 组合权重：alpha 用于控制收入与存款的权重占比
        alpha = 0.5  # 可按需调整/暴露为超参数
        poverty_weights = {}
        total_weight = 0.0
        
        for household_id, income in household_incomes.items():
            # 收入成分（越低越高）
            income_component = 0.0
            if max_income != min_income:
                income_component = (max_income - income) / (max_income - min_income)
            
            # 存款成分（越低越高）
            balance = household_balances.get(household_id, 0.0)
            balance_component = 0.0
            if max_balance != min_balance:
                balance_component = (max_balance - balance) / (max_balance - min_balance)
            
            # 综合权重
            weight = alpha * income_component + (1 - alpha) * balance_component
            poverty_weights[household_id] = weight
            total_weight += weight
        
        allocations = {}
        for household_id in all_workers:
            proportion = poverty_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _unemployment_focused_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """失业导向分配策略（失业者获得更多）"""
        unemployment_weights = {}
        total_weight = 0.0
        
        for household_id in all_workers:
            labor_hours = self.laborhour.get(household_id, [])
            employed_count = sum(1 for lh in labor_hours if not lh.is_valid and lh.company_id is not None)
            unemployed_count = len(labor_hours) - employed_count
            
            # 失业者权重更高
            weight = unemployed_count * 2.0 + employed_count * 1.0
            unemployment_weights[household_id] = weight
            total_weight += weight
        
        if total_weight <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = unemployment_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _family_size_allocation(self, all_workers: List[str], total_tax: float) -> Dict[str, float]:
        """按家庭规模分配策略"""
        family_weights = {}
        total_weight = 0.0
        
        for household_id in all_workers:
            labor_hours = self.laborhour.get(household_id, [])
            family_size = len(labor_hours)
            family_weights[household_id] = family_size
            total_weight += family_size
        
        if total_weight <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = family_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _mixed_allocation(self, all_workers: List[str], total_tax: float, 
                         poverty_weight: float, unemployment_weight: float, 
                         family_size_weight: float, month: int) -> Dict[str, float]:
        """混合分配策略"""
        # 获取各种权重
        poverty_allocations = self._poverty_focused_allocation(all_workers, total_tax, month)
        unemployment_allocations = self._unemployment_focused_allocation(all_workers, total_tax, month)
        family_size_allocations = self._family_size_allocation(all_workers, total_tax)
        equal_allocations = self._equal_allocation(all_workers, total_tax)
        
        # 计算剩余权重
        remaining_weight = 1.0 - poverty_weight - unemployment_weight - family_size_weight
        if remaining_weight < 0:
            remaining_weight = 0.0
        
        # 混合分配
        allocations = {}
        for household_id in all_workers:
            mixed_amount = (
                poverty_allocations[household_id] * poverty_weight +
                unemployment_allocations[household_id] * unemployment_weight +
                family_size_allocations[household_id] * family_size_weight +
                equal_allocations[household_id] * remaining_weight
            )
            allocations[household_id] = mixed_amount
        
        return allocations

    def add_interest_tx(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加利息交易记录
        """
        tx = Transaction(
            id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            type='interest',
            month=month
        )
        self.tx_history.append(tx)
        return tx.id
    def add_redistribution_tx(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加再分配交易记录
        """
        tx = Transaction(
            id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            type='redistribution',
            month=month
        )
        self.tx_history.append(tx)
        return tx.id

    def add_tx_service(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加服务类型交易记录，直接更新账本并记录到交易历史
        用于政府服务、基础服务等不需要商品库存的交易
        
        Args:
            month: 交易月份
            sender_id: 付款方ID
            receiver_id: 收款方ID  
            amount: 交易金额
            
        Returns:
            str: 交易ID
        """
        # 检查付款方余额是否足够
        if self.ledger[sender_id].amount < amount:
            raise ValueError(f"Insufficient balance for {sender_id}: ${self.ledger[sender_id].amount:.2f} < ${amount:.2f}")
        
        # 直接更新账本
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount
        
        # 创建服务交易记录
        tx = Transaction(
            id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            assets=[],  # 服务交易没有具体商品
            type='service',  # 使用service类型
            month=month
        )
        
        # 添加到交易历史
        self.tx_history.append(tx)
       
        return tx.id
    
    def add_inherent_market_transaction(self, month: int, sender_id: str, receiver_id: str, 
                                       amount: float, product_id: str, quantity: float,
                                       product_name: str = "Unknown", product_price: float = 0.0,
                                       product_classification: str = "Unknown") -> str:
        """
        添加固有市场交易记录（包含毛利率计算）
        用于记录政府通过固有市场购买企业商品的交易
        
        Args:
            month: 交易月份
            sender_id: 付款方ID (通常是政府)
            receiver_id: 收款方ID (企业)
            amount: 交易金额
            product_id: 商品ID
            quantity: 购买数量
            product_name: 商品名称
            product_price: 商品单价
            product_classification: 商品分类（daily_cate）
            
        Returns:
            str: 交易ID
        """
        # 检查付款方余额是否足够
        if self.ledger[sender_id].amount < amount:
            raise ValueError(f"Insufficient balance for {sender_id}: ${self.ledger[sender_id].amount:.2f} < ${amount:.2f}")
        
        # 政府支付企业
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount
        
        # 💰 企业收入、成本和利润计算（与process_purchase保持一致）
        # 1. 记录企业收入
        revenue = amount
        self.record_firm_income(receiver_id, revenue)
        self.record_firm_monthly_income(receiver_id, month, revenue)
        
        # 2. 根据商品类别和毛利率计算成本和利润
        config = self.firm_innovation_config.get(receiver_id)
        if not config or config.profit_margin is None:
            # 如果没有创新配置，使用默认毛利率
            profit_margin = self.category_profit_margins.get(product_classification, 25.0)
        else:
            profit_margin = config.profit_margin
        margin_rate = profit_margin / 100.0
        
        cost = revenue * (1 - margin_rate)  # 成本
        gross_profit = revenue * margin_rate  # 毛利润
        
        # 3. 记录成本支出
        if self.ledger[receiver_id].amount >= cost:
            self.ledger[receiver_id].amount -= cost
            self.record_firm_expense(receiver_id, cost)
            self.record_firm_monthly_expense(receiver_id, month, cost)
        else:
            logger.warning(f"企业 {receiver_id} 余额不足以支付成本: ${self.ledger[receiver_id].amount:.2f} < ${cost:.2f}")
        
        # 创建固有市场交易记录
        unit_price = product_price if product_price > 0 else (amount / quantity if quantity > 0 else 0)
        if unit_price <= 0:
            unit_price = 0.01
            
        product_kwargs = dict(
            asset_type='products',
            product_id=product_id,
            name=product_name,
            owner_id=receiver_id,
            amount=quantity,
            price=unit_price,
            classification=product_classification
        )
        product_kwargs = inject_product_attributes(product_kwargs, product_id)
        product_asset = Product(**product_kwargs)
        
        tx = Transaction(
            id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            assets=[product_asset],
            type='inherent_market',
            month=month
        )
        self.tx_history.append(tx)
        
        # 4. 企业需要缴纳企业所得税（基于毛利润）
        corporate_tax = gross_profit * self.corporate_tax_rate
        
        # 企业支付所得税
        if self.ledger[receiver_id].amount >= corporate_tax:
            self.ledger[receiver_id].amount -= corporate_tax
            self.record_firm_expense(receiver_id, corporate_tax)
            self.record_firm_monthly_expense(receiver_id, month, corporate_tax)
        else:
            logger.warning(f"企业 {receiver_id} 余额不足以支付企业所得税: ${self.ledger[receiver_id].amount:.2f} < ${corporate_tax:.2f}")
        
        # 政府收取企业所得税
        self.ledger["gov_main_simulation"].amount += corporate_tax
        
        # 记录企业所得税交易
        corp_tax_tx = Transaction(
            id=str(uuid4()),
            sender_id=receiver_id,
            receiver_id="gov_main_simulation",
            amount=corporate_tax,
            type='corporate_tax',
            month=month
        )
        self.tx_history.append(corp_tax_tx)
        
        # logger.info(f"固有市场交易: 政府购买商品 {product_name}(ID:{product_id}, {product_classification}) "
        #            f"数量 {quantity} 金额 ${amount:.2f}, 成本 ${cost:.2f}, 毛利润 ${gross_profit:.2f} (毛利率{profit_margin}%), "
        #            f"企业所得税 ${corporate_tax:.2f}")
        
        return tx.id
    
    def get_product_inventory(self, owner_id: str, product_id: str) -> float:
        """
        获取指定商品的当前库存数量
        """
        if owner_id not in self.products:
            return 0.0
        
        for product in self.products[owner_id]:
            if product.product_id == product_id:
                return product.amount
        return 0.0
    
    def get_all_product_inventory(self) -> Dict[tuple, float]:
        """
        批量获取所有商品的库存信息
        
        Returns:
            Dict[tuple, float]: {(product_id, owner_id): amount} 字典
        """
        inventory_dict = {}
        for owner_id, products in self.products.items():
            for product in products:
                key = (product.product_id, owner_id)
                inventory_dict[key] = product.amount
        return inventory_dict
    
    async def sync_product_inventory_to_market(self, product_market):
        """
        将EconomicCenter的库存信息同步到ProductMarket
        这个方法可以定期调用以保持两边数据一致
        """
        try:
            # 收集所有有库存的商品
            all_products = []
            for owner_id, products in self.products.items():
                if owner_id in self.company_id:
                    for product in products:
                        if product.amount > 0:  # 只同步有库存的商品
                            all_products.append(product)
            
            # 更新ProductMarket的商品列表
            await product_market.update_products_from_economic_center.remote(all_products)
            logger.info(f"已同步 {len(all_products)} 个商品到ProductMarket")
            return True
        except Exception as e:
            logger.error(f"同步库存到ProductMarket失败: {e}")
            return False
    
    def update_product_prices_based_on_sales(self, sales_data: Dict[tuple, Dict], price_adjustment_rate: float = 0.1) -> Dict[str, float]:
        """
        根据销量数据更新商品价格（包含库存信息）
        sales_data: {(product_id, seller_id): {"quantity_sold": float, "revenue": float, "demand_level": str}}
        price_adjustment_rate: 价格调整幅度 (0.1 = 10%)
        返回: {product_id: new_price}
        
        注意：使用 (product_id, seller_id) 作为key，支持竞争市场模式下同一商品由多个企业销售
        """
        price_changes = {}
        
        # 🔍 调试信息：检查 company_id 列表
        logger.info(f"📋 已注册的企业数量: {len(self.company_id)}")
        logger.info(f"📦 商品所有者数量: {len(self.products)}")
        
        processed_owners = 0
        skipped_owners = 0
        price_increase_count = 0
        price_decrease_count = 0
        
        for owner_id, products in self.products.items():
            if owner_id in self.company_id:  # 只处理真正的公司
                processed_owners += 1
                for product in products:
                    product_id = product.product_id
                    sales_key = (product_id, owner_id)
                    
                    # 使用 (product_id, owner_id) 作为key查找销量数据
                    if sales_key in sales_data:
                        sales_info = sales_data[sales_key]
                        quantity_sold = sales_info.get("quantity_sold", 0)
                        revenue = sales_info.get("revenue", 0)
                        demand_level = sales_info.get("demand_level", "normal")
                        current_inventory = product.amount  # 获取当前库存
                        
                        # 计算价格调整（传入库存信息）
                        old_price = product.price
                        new_price = self._calculate_new_price(
                            old_price, quantity_sold, revenue, demand_level, 
                            price_adjustment_rate, current_inventory
                        )
                        
                        # 更新价格
                        product.price = new_price
                        price_changes[product_id] = new_price
                        
                        # 统计涨价和降价商品数
                        if new_price > old_price:
                            price_increase_count += 1
                        elif new_price < old_price:
                            price_decrease_count += 1
                        
                        # 打印价格变化
                        price_change_pct = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
                        supply_demand_ratio = current_inventory / quantity_sold if quantity_sold > 0 else float('inf')
                        
                        if abs(price_change_pct) > 5:  # 只打印变化超过5%的
                            print(f"💹 {product.name[:40]:40} | "
                                  f"${old_price:6.2f} → ${new_price:6.2f} ({price_change_pct:+6.1f}%) | "
                                  f"销量:{quantity_sold:5.1f} | 库存:{current_inventory:5.1f} | "
                                  f"供需比:{supply_demand_ratio:5.2f} | {demand_level}")
            else:
                skipped_owners += 1
        
        logger.info(f"✅ 处理了 {processed_owners} 个企业的商品，跳过了 {skipped_owners} 个非企业所有者")
        print(f"\n📊 价格调整汇总: 涨价 {price_increase_count} 种商品, 降价 {price_decrease_count} 种商品")
        
        if skipped_owners > 0:
            logger.warning(f"⚠️ 跳过的所有者示例: {list(self.products.keys())[:5]}")
            logger.warning(f"⚠️ 已注册企业ID示例: {self.company_id[:5] if self.company_id else '空列表'}")
        
        return price_changes
    
    def _calculate_new_price(self, current_price: float, quantity_sold: float, revenue: float, 
                           demand_level: str, adjustment_rate: float, current_inventory: float = None) -> float:
        """
        ✨ 优化版价格调整算法 - 让供不应求的商品更明显涨价
        
        考虑因素：
        1. 销量水平（绝对值）
        2. 需求水平（high/normal/low）
        3. 供需比（库存与销量的比例）- 新增
        4. 收入效率
        """
        # 基础价格调整因子
        base_adjustment = 0.0
        
        # 1. 根据销量调整（更激进的调整）
        if quantity_sold > 100:  # 超高销量
            base_adjustment += 0.15  # 涨价15%
        elif quantity_sold > 50:  # 高销量
            base_adjustment += 0.10  # 涨价10%
        elif quantity_sold > 30:  # 中等销量
            base_adjustment += 0.05  # 涨价5%
        elif quantity_sold < 5:  # 极低销量
            base_adjustment -= 0.08  # 降价8%
        elif quantity_sold < 15:  # 低销量
            base_adjustment -= 0.05  # 降价5%
        
        # 2. 根据需求水平调整（更激进）
        demand_multipliers = {
            "high": 0.20,      # 高需求涨价20%
            "normal": 0.0,     # 正常需求不变
            "low": -0.12       # 低需求降价12%
        }
        base_adjustment += demand_multipliers.get(demand_level, 0.0)
        
        # 3. ✨ 新增：根据供需比调整（库存与销量的比例）
        # 这是关键的供不应求判断指标
        if current_inventory is not None and quantity_sold > 0:
            supply_demand_ratio = current_inventory / quantity_sold
            
            if supply_demand_ratio < 0.5:
                # 库存不足销量的一半 - 严重供不应求
                base_adjustment += 0.25  # 大幅涨价25%
                logger.debug(f"🔥 严重供不应求: 库存{current_inventory:.1f} / 销量{quantity_sold:.1f} = {supply_demand_ratio:.2f}")
            elif supply_demand_ratio < 1.0:
                # 库存不足一个周期的销量 - 供不应求
                base_adjustment += 0.15  # 涨价15%
                logger.debug(f"📈 供不应求: 库存{current_inventory:.1f} / 销量{quantity_sold:.1f} = {supply_demand_ratio:.2f}")
            elif supply_demand_ratio < 2.0:
                # 库存略高于销量 - 供需平衡
                base_adjustment += 0.02  # 小幅涨价2%
            elif supply_demand_ratio < 5.0:
                # 库存明显高于销量 - 供过于求
                base_adjustment -= 0.08  # 降价8%
                logger.debug(f"📉 供过于求: 库存{current_inventory:.1f} / 销量{quantity_sold:.1f} = {supply_demand_ratio:.2f}")
            else:
                # 库存严重过剩 - 严重供过于求
                base_adjustment -= 0.15  # 大幅降价15%
                logger.debug(f"⚠️ 严重供过于求: 库存{current_inventory:.1f} / 销量{quantity_sold:.1f} = {supply_demand_ratio:.2f}")
        
        # 4. 根据收入效率调整
        if revenue > 0 and quantity_sold > 0:
            avg_revenue_per_unit = revenue / quantity_sold
            if avg_revenue_per_unit > current_price * 1.15:  # 收入效率高（提高阈值）
                base_adjustment += 0.05
            elif avg_revenue_per_unit < current_price * 0.85:  # 收入效率低（降低阈值）
                base_adjustment -= 0.05
        
        # 5. 应用调整率（增大调整幅度，让价格变化更明显）
        # 原来是直接乘以adjustment_rate，现在增加系数让变化更明显
        price_change = current_price * base_adjustment * adjustment_rate * 1.5  # 放大1.5倍
        
        # 6. 计算新价格，放宽价格变动范围
        new_price = current_price + price_change
        min_price = current_price * 0.3   # 最低可降至原价的30%（原来是50%）
        max_price = current_price * 3.0   # 最高可涨至原价的300%（原来是200%）
        
        # 7. 确保价格合理性（不能低于成本的80%）
        absolute_min_price = current_price * 0.4  # 绝对最低价
        
        final_price = max(absolute_min_price, min(new_price, max_price))
        
        # 记录显著的价格变化
        if abs(final_price - current_price) / current_price > 0.1:  # 变化超过10%
            logger.info(f"💹 显著价格变动: ${current_price:.2f} → ${final_price:.2f} "
                       f"({((final_price - current_price) / current_price * 100):+.1f}%) | "
                       f"销量:{quantity_sold:.1f} | 库存:{current_inventory:.1f if current_inventory else 'N/A'} | "
                       f"需求:{demand_level}")
        
        return final_price
    
    async def sync_price_changes_to_market(self, product_market, price_changes: Dict[str, float]) -> bool:
        """
        将价格变更同步到ProductMarket
        """
        try:
            success = await product_market.update_product_prices.remote(price_changes)
            logger.info(f"已同步 {len(price_changes)} 个商品的价格变更到ProductMarket")
            return success
        except Exception as e:
            logger.error(f"同步价格变更到ProductMarket失败: {e}")
            return False
    
    def collect_sales_statistics(self, month: int) -> Dict[tuple, Dict]:
        """
        收集指定月份的销售统计数据
        返回: {(product_id, seller_id): {
            "product_id": str,
            "seller_id": str,
            "quantity_sold": float, 
            "revenue": float, 
            "demand_level": str,
            "household_quantity": float,  # 家庭购买数量
            "household_revenue": float,  # 家庭购买收入
            "inherent_market_quantity": float,  # 固定市场消耗数量
            "inherent_market_revenue": float  # 固有市场收入
        }}
        
        注意：使用 (product_id, seller_id) 作为key，支持竞争市场模式下同一商品由多个企业销售
        """
        sales_stats = {}
        
        # 从交易历史中收集销售数据
        for tx in self.tx_history:
            if tx.month == month:
                seller_id = tx.receiver_id
                
                # 处理家庭购买（purchase类型）
                if tx.type == 'purchase':
                    for asset in tx.assets:
                        if hasattr(asset, 'product_id') and asset.product_id:
                            product_id = asset.product_id
                            key = (product_id, seller_id)
                            
                            if key not in sales_stats:
                                sales_stats[key] = {
                                    "product_id": product_id,
                                    "seller_id": seller_id,
                                    "quantity_sold": 0.0,
                                    "revenue": 0.0,
                                    "demand_level": "normal",
                                    "household_quantity": 0.0,
                                    "household_revenue": 0.0,  # 新增：家庭购买收入
                                    "inherent_market_quantity": 0.0,
                                    "inherent_market_revenue": 0.0  # 新增：固有市场收入
                                }
                            
                            # 累计家庭销量和收入
                            household_revenue = asset.price * asset.amount
                            sales_stats[key]["quantity_sold"] += asset.amount
                            sales_stats[key]["household_quantity"] += asset.amount
                            sales_stats[key]["revenue"] += household_revenue
                            sales_stats[key]["household_revenue"] += household_revenue

                
                # 处理固定市场消耗（inherent_market类型）
                elif tx.type == 'inherent_market':
                    for asset in tx.assets:
                        if hasattr(asset, 'product_id') and asset.product_id:
                            product_id = asset.product_id
                            key = (product_id, seller_id)
                            
                            if key not in sales_stats:
                                sales_stats[key] = {
                                    "product_id": product_id,
                                    "seller_id": seller_id,
                                    "quantity_sold": 0.0,
                                    "revenue": 0.0,
                                    "demand_level": "normal",
                                    "household_quantity": 0.0,
                                    "household_revenue": 0.0,  # 新增：家庭购买收入
                                    "inherent_market_quantity": 0.0,
                                    "inherent_market_revenue": 0.0  # 新增：固有市场收入
                                }
                            
                            # 累计固定市场销量和收入
                            inherent_revenue = tx.amount  # 固定市场交易的总金额
                            sales_stats[key]["quantity_sold"] += asset.amount
                            sales_stats[key]["inherent_market_quantity"] += asset.amount
                            sales_stats[key]["revenue"] += inherent_revenue
                            sales_stats[key]["inherent_market_revenue"] += inherent_revenue
        
        # 根据销量确定需求水平
        for key, stats in sales_stats.items():
            quantity = stats["quantity_sold"]
            if quantity > 100:
                stats["demand_level"] = "high"
            elif quantity < 10:
                stats["demand_level"] = "low"
            else:
                stats["demand_level"] = "normal"
        
        print(f"📊 销售数据收集: 月份{month}, 交易记录{len(self.tx_history)}条, 销售商品-企业组合{len(sales_stats)}种")
        
        # 计算总收入统计
        total_revenue = sum(s['revenue'] for s in sales_stats.values())
        total_household_revenue = sum(s.get('household_revenue', 0) for s in sales_stats.values())
        total_inherent_revenue = sum(s.get('inherent_market_revenue', 0) for s in sales_stats.values())
        
        if total_revenue > 0:
            household_ratio = (total_household_revenue / total_revenue) * 100
            inherent_ratio = (total_inherent_revenue / total_revenue) * 100
            print(f"💰 收入统计: 总收入${total_revenue:.2f} | "
                  f"家庭购买${total_household_revenue:.2f} ({household_ratio:.1f}%) | "
                  f"固有市场${total_inherent_revenue:.2f} ({inherent_ratio:.1f}%)")
        
        if sales_stats:
            # 显示销量最高的3个商品-企业组合，并区分家庭和固定市场
            top_sales = sorted(sales_stats.items(), key=lambda x: x[1]['quantity_sold'], reverse=True)[:3]
            for (product_id, seller_id), stats in top_sales:
                household_rev = stats.get('household_revenue', 0)
                inherent_rev = stats.get('inherent_market_revenue', 0)
                total_rev = stats['revenue']
                hh_ratio = (household_rev / total_rev * 100) if total_rev > 0 else 0
                in_ratio = (inherent_rev / total_rev * 100) if total_rev > 0 else 0
                
                print(f"   - {product_id}@{seller_id}: 总销量{stats['quantity_sold']:.1f} "
                      f"(家庭:{stats['household_quantity']:.1f} | 固定市场:{stats['inherent_market_quantity']:.1f}), "
                      f"总收入${total_rev:.2f} (家庭:${household_rev:.2f} {hh_ratio:.1f}% | 固有:${inherent_rev:.2f} {in_ratio:.1f}%)")
        return sales_stats
    
    async def execute_monthly_production_cycle(self, month: int, labor_market, product_market, std_jobs, firms: List = None, production_config: Dict = None, innovation_config: Dict = None) -> Dict[str, Any]:
        """
        执行月度生产周期
        1. 所有公司基础生产
        2. 有工人的公司额外生产
        3. 根据销量调整产出
        
        Args:
            production_config: 生产配置参数字典，包含:
                - base_production_rate: 基础补货量
                - high_demand_multiplier: 高需求倍数
                - low_demand_multiplier: 低需求倍数
                - labor_productivity_factor: 劳动力生产率
                - labor_elasticity: 劳动力弹性
        """
        logger.info(f"🏭 开始第 {month} 月生产周期...")

        # 兼容旧实例：如果早期创建的 EconomicCenter 没有该属性，这里动态补上，避免 AttributeError
        if not hasattr(self, "production_stats_by_month"):
            self.production_stats_by_month = {}
        production_stats = {
            "total_companies": 0,
            "companies_with_workers": 0,
            "base_production_total": 0.0,
            "labor_production_total": 0.0,
            "products_restocked": 0
        }
        
        # 根据self.company_id统计总公司数（只统计真正的公司）
        for owner_id in self.company_id:
            if owner_id in self.products and self.products[owner_id]:
                production_stats["total_companies"] += 1
        
        try:
            # 1. 获取销售数据（用于指导生产）
            sales_data = self.collect_sales_statistics(month)
            
            # 2. 为所有公司执行基础生产
            base_production = await self._execute_base_production_for_all_firms(month, sales_data, firms, std_jobs, production_config)
            production_stats["base_production_total"] = base_production["total_output"]
            production_stats["products_restocked"] = base_production["products_restocked"]
            
            # 3. 为有工人的公司执行额外生产 (基于技能匹配的有效劳动力)
            labor_production = await self._execute_labor_based_production(
                month, sales_data, labor_market, firms, std_jobs, production_config, innovation_config
            )
            production_stats["labor_production_total"] = labor_production["total_output"]
            production_stats["companies_with_workers"] = labor_production["companies_count"]
            production_stats["firm_labor_efficiency"] = labor_production.get("firm_labor_efficiency", {})

            if "firm_innovation_arrival_rate" in labor_production:
                production_stats["firm_innovation_arrival_rate"] = labor_production["firm_innovation_arrival_rate"]
            if "firm_innovation_arrivals" in labor_production:
                production_stats["firm_innovation_arrivals"] = labor_production["firm_innovation_arrivals"]
            if "firm_research_labor" in labor_production:
                production_stats["firm_research_labor"] = labor_production["firm_research_labor"]
            if "total_research_effective_labor" in labor_production:
                production_stats["total_research_effective_labor"] = labor_production["total_research_effective_labor"]

            # 4. 记录每个企业的生产统计数据
            firm_base_production = base_production.get("firm_production", {})
            firm_labor_prod = labor_production.get("firm_labor_production", {})
            
            # 合并所有企业的基础生产和劳动力生产
            all_company_ids = set(firm_base_production.keys()) | set(firm_labor_prod.keys())
            for company_id in all_company_ids:
                base_prod = firm_base_production.get(company_id, 0.0)
                labor_prod = firm_labor_prod.get(company_id, 0.0)
                
                # 保存到企业月度生产统计
                self.firm_production_stats[company_id][month]["base_production"] = base_prod
                self.firm_production_stats[company_id][month]["labor_production"] = labor_prod
            
            # 5. 同步库存到ProductMarket
            await self.sync_product_inventory_to_market(product_market)
            
            logger.info(f"✅ 第 {month} 月生产周期完成")
            logger.info(f"   基础生产: {base_production['total_output']:.2f} 单位")
            logger.info(f"   劳动力生产: {labor_production['total_output']:.2f} 单位")
            logger.info(f"   补货商品: {base_production['products_restocked']} 种")
            
            # 🆕 缓存本月生产统计，供后续可视化/导出查询
            self.production_stats_by_month[month] = production_stats
            
            return production_stats
            
        except Exception as e:
            logger.error(f"❌ 第 {month} 月生产周期失败: {e}")
            # 失败也缓存，便于后续诊断
            self.production_stats_by_month[month] = production_stats
            return production_stats

    async def _execute_base_production_for_all_firms(self, month: int, sales_data: Dict, firms: List = None, std_jobs = None, production_config: Dict = None) -> Dict[str, Any]:
        """
        ✨ 改进版基于利润和成本的生产系统
        
        核心改进：
        1. 区分两类商品：家庭市场商品 vs 纯固定市场商品
        2. 分配生产预算：50%给家庭市场商品，20%给固定市场商品，30%作为储备
        3. 家庭市场商品：按销量生产，维持3个月库存
        4. 固定市场商品：维持最低库存（50件）
        5. 库存预警：如果库存价值超过月收入2倍，强制减产50%
        
        Args:
            month: 当前月份（用于获取当月财务数据）
        """
        # 从配置中获取生产预算比例，默认70%的利润用于再生产
        if production_config:
            profit_to_production_ratio = production_config.get('profit_to_production_ratio', 0.7)
            min_production_per_product = production_config.get('min_production_per_product', 5.0)
        else:
            profit_to_production_ratio = 0.7
            min_production_per_product = 5.0
        
        total_output = 0.0
        products_restocked = 0
        firm_production = {}  # 记录每个企业的基础生产量
        
        # 遍历所有公司
        for owner_id, products in self.products.items():
            if not products:  # 跳过没有产品的公司
                continue

            if owner_id not in self.company_id:  # 只处理真正的公司
                continue
            
            firm_base_production = 0.0  # 该企业的基础生产总量
            
            # 1. 获取企业当月财务数据
            current_financials = self.firm_monthly_financials.get(owner_id, {}).get(month, {})
            current_income = current_financials.get("income", 0.0)
            current_expenses = current_financials.get("expenses", 0.0)
            current_profit = current_income - current_expenses
            
            # 使用当月利润作为生产依据（包括第一个月）
            prev_profit = current_profit
            
            # 如果当月利润为负或没有记录，使用保底预算
            if prev_profit <= 0:
                prev_profit = 1000.0
            
            # 2. 计算初始生产预算（利润的一定比例）
            production_budget = max(0, prev_profit * profit_to_production_ratio)
            
            # 如果企业利润为负或预算太少，给予最小预算（避免企业无法生产）
            if production_budget < 1000:
                production_budget = 1000.0  # 最小生产预算
            
            # 📦 库存预警：检查总库存价值是否过高
            total_inventory_value = sum(p.amount * p.price for p in products)
            if total_inventory_value > current_income * 2 and current_income > 0:
                print(f"   ⚠️  企业 {owner_id}: 库存过高 (${total_inventory_value:.2f} > ${current_income*2:.2f})，强制减产50%")
                production_budget *= 0.5
            
            # 3. 区分两类商品
            household_products = []  # 有家庭购买的商品
            inherent_only_products = []  # 只有固定市场的商品
            
            for product in products:
                product_id = product.product_id
                # 使用 (product_id, owner_id) 作为key查找该企业该商品的销量数据
                sales_key = (product_id, owner_id)
                
                # 检查是否有销售记录（家庭购买或固定市场）
                if sales_key in sales_data:
                    sales_info = sales_data[sales_key]
                    household_quantity = sales_info.get("household_quantity", 0.0)
                    if household_quantity > 0:
                        household_products.append(product)
                    else:
                        inherent_only_products.append(product)
                else:
                    # 没有任何销售记录，归为固定市场商品
                    inherent_only_products.append(product)
            
            # 4. 分配生产预算
            household_budget = production_budget * 0.5  # 30%给家庭市场商品
            inherent_budget = production_budget * 0.5   # 40%给纯固定市场商品
            # 剩余30%作为企业储备金（不使用）
            
            print(f"   💼 企业 {owner_id}: 家庭商品{len(household_products)}个, 固定市场商品{len(inherent_only_products)}个")
            print(f"   💰 预算分配: 总预算=${production_budget:.2f} | 家庭市场=${household_budget:.2f} | 固定市场=${inherent_budget:.2f}")
            
            # 5. 家庭市场商品：按利润优先级分配预算生产，维持3个月库存
            if household_products:
                # 🎯 第一步：计算每个商品的利润优先级
                product_profit_priorities = {}
                total_profit_priority = 0.0
                
                for product in household_products:
                    product_id = product.product_id
                    sales_key = (product_id, owner_id)
                    
                    # 获取商品参数
                    config = self.firm_innovation_config.get(owner_id)
                    if not config or config.profit_margin is None:
                        # 如果没有创新配置，使用默认毛利率
                        product_category = product.classification if hasattr(product, 'classification') else "Unknown"
                        profit_margin = self.category_profit_margins.get(product_category, 25.0)
                    else:
                        profit_margin = config.profit_margin
                    unit_profit = product.price * profit_margin / 100.0  # 单件利润（毛利率需要除以100转换为小数）
                    
                    # 获取家庭销量（使用 (product_id, owner_id) 作为key）
                    household_sales = sales_data.get(sales_key, {}).get("household_quantity", 0.0)
                    
                    # 计算利润优先级分数 = 单件利润 × 月销量
                    # 这表示该商品每月能带来的利润贡献
                    profit_priority = unit_profit * household_sales
                    
                    # 如果销量为0但商品存在，给予最小优先级（避免完全不生产）
                    if profit_priority == 0:
                        profit_priority = unit_profit * 1.0  # 假设至少卖1件
                    
                    product_profit_priorities[product_id] = {
                        'priority': profit_priority,
                        'unit_profit': unit_profit,
                        'sales': household_sales,
                        'product': product
                    }
                    total_profit_priority += profit_priority
                
                print(f"   💰 家庭商品利润优先级总和: ${total_profit_priority:.2f}")
                
                # 🎯 第二步：按利润优先级分配预算并生产
                for product_id, info in product_profit_priorities.items():
                    product = info['product']
                    household_sales = info['sales']
                    unit_profit = info['unit_profit']
                    priority = info['priority']
                    
                    # 期望库存 = max(销量×倍数, 当前库存×维持率)
                    # 🔧 优化：防止高初始库存商品过度下降
                    if household_sales > 0:
                        # 基于销量的目标库存（基础倍数3个月）
                        sales_based_target = household_sales * 3
                        
                        # 基于当前库存的维持目标（保持90%库存水平）
                        inventory_maintain_target = product.amount * 0.9
                        
                        # 取两者较大值：既考虑销量需求，也防止库存过度下降
                        target_inventory = max(sales_based_target, inventory_maintain_target)
                    else:
                        # 没有销量，维持当前库存的90%（缓慢下降）
                        target_inventory = product.amount * 0.9
                    
                    # 计算需要生产的数量
                    if product.amount < target_inventory:
                        production_quantity = target_inventory - product.amount
                    else:
                        production_quantity = 0
                    
                    # 确保最小生产量
                    if production_quantity > 0:
                        production_quantity = max(production_quantity, min_production_per_product)
                    
                    # 🏠 关键规则：补货必须要达到家庭购买的数量
                    # 如果有家庭购买，补货量必须至少等于家庭购买数量
                    if household_sales > 0:
                        production_quantity = max(production_quantity, household_sales)
                    
                    # 🎯 关键改进：按利润优先级比例分配预算
                    if total_profit_priority > 0:
                        # 该商品获得的预算 = 总预算 × (该商品利润优先级 / 总利润优先级)
                        available_budget_for_product = household_budget * (priority / total_profit_priority)
                    else:
                        # 如果所有商品利润优先级都是0，则平均分配
                        available_budget_for_product = household_budget / len(household_products) if household_products else 0
                    
                    # 计算成本并检查是否超出预算
                    category = product.classification or "Retail and Stores"
                    profit_margin = self.category_profit_margins.get(category, 25.0) / 100.0
                    product_cost = product.price / (1 + profit_margin)
                    
                    required_budget = production_quantity * product_cost
                    
                    if required_budget > available_budget_for_product:
                        # 预算不足，按预算生产
                        production_quantity = available_budget_for_product / product_cost if product_cost > 0 else min_production_per_product
                    
                    # 🏠 关键规则：补货必须要达到家庭购买的数量（如果预算允许）
                    # 即使预算受限，也要优先确保至少满足家庭购买数量
                    if household_sales > 0:
                        household_required_budget = household_sales * product_cost
                        if household_required_budget <= available_budget_for_product:
                            # 预算足够，确保至少生产家庭购买数量
                            production_quantity = max(production_quantity, household_sales)
                        else:
                            # 预算不足以满足家庭购买需求，记录警告
                            logger.warning(f"⚠️ 商品 {product.name} 预算不足，无法满足家庭购买数量 {household_sales:.1f}，只能生产 {production_quantity:.1f}")
                    
                    # 更新库存
                    if production_quantity > 0:
                        old_amount = product.amount
                        product.amount += production_quantity
                        total_output += production_quantity
                        firm_base_production += production_quantity
                        products_restocked += 1
                        
                        print(f"   🏠 家庭商品: {product.name[:30]} | "
                              f"销量:{household_sales:.1f} | 单件利润:${unit_profit:.2f} | "
                              f"优先级:{priority:.1f} | 预算:${available_budget_for_product:.2f} | "
                              f"生产:{production_quantity:.1f}件 | "
                              f"库存:{old_amount:.1f}→{product.amount:.1f}")
            
            # 6. 固定市场商品：按利润优先级维持最低库存
            if inherent_only_products:
                # 🔧 动态最低库存：根据商品价值调整
                # 高价值商品维持更高库存，低价值商品维持较低库存
                min_inventory_threshold = 50  # 默认最低库存
                
                # 🎯 第一步：计算每个固定市场商品的利润优先级
                inherent_profit_priorities = {}
                total_inherent_priority = 0.0
                
                for product in inherent_only_products:
                    product_id = product.product_id
                    sales_key = (product_id, owner_id)
                    
                    # 获取商品参数
                    category = product.classification
                    profit_margin = self.category_profit_margins.get(category, 25.0) / 100.0
                    unit_profit = product.price * profit_margin  # 单件利润
                    
                    # 获取固定市场销量（使用 (product_id, owner_id) 作为key）
                    inherent_sales = sales_data.get(sales_key, {}).get("inherent_market_quantity", 0.0)
                    
                    # 计算利润优先级 = 单件利润 × 销量（如果有的话）
                    if inherent_sales > 0:
                        profit_priority = unit_profit * inherent_sales
                    else:
                        # 没有销量的商品，按单件利润作为优先级
                        profit_priority = unit_profit
                    
                    inherent_profit_priorities[product_id] = {
                        'priority': profit_priority,
                        'unit_profit': unit_profit,
                        'sales': inherent_sales,
                        'product': product
                    }
                    total_inherent_priority += profit_priority
                
                print(f"   💰 固定市场商品利润优先级总和: ${total_inherent_priority:.2f}")
                
                # 🎯 第二步：按利润优先级分配预算并生产
                for product_id, info in inherent_profit_priorities.items():
                    product = info['product']
                    unit_profit = info['unit_profit']
                    priority = info['priority']
                    inherent_sales = info['sales']
                    
                    # 🔧 动态库存目标：基于预算优先级而不是绝对销量
                    # 问题：所有商品的固有市场销量都相同（因为都按65%比例消耗）
                    # 解决：根据商品价值和预算优先级设置差异化的目标库存
                    
                    if inherent_sales > 0:
                        # 方法：根据预算分配比例设置目标库存
                        # 高优先级商品（高利润）应该有更高的目标库存
                        budget_ratio = priority / total_inherent_priority if total_inherent_priority > 0 else 0
                        
                        # 基础目标：补回上月消耗的量（假设消耗率65%）
                        base_target = product.amount + inherent_sales
                        
                        # 根据预算比例调整目标（预算多的商品目标库存更高）
                        # 预算比例高的可以达到base_target的150%，低的只能达到80%
                        budget_multiplier = 0.8 + (budget_ratio * len(inherent_only_products) * 0.7)
                        target_inventory = base_target * budget_multiplier
                        
                        # 确保不低于最小阈值
                        target_inventory = max(target_inventory, min_inventory_threshold)
                    else:
                        # 无销量：维持当前库存的85%（缓慢下降）
                        target_inventory = max(product.amount * 0.85, min_inventory_threshold)
                    
                    # 计算需要生产的数量
                    if product.amount < target_inventory:
                        production_quantity = target_inventory - product.amount
                    else:
                        production_quantity = 0
                    
                    # 🎯 按利润优先级比例分配预算
                    if total_inherent_priority > 0:
                        available_budget_for_product = inherent_budget * (priority / total_inherent_priority)
                    else:
                        available_budget_for_product = inherent_budget / len(inherent_only_products) if inherent_only_products else 0
                    
                    # 预算限制
                    category = product.classification or "Retail and Stores"
                    profit_margin = self.category_profit_margins.get(category, 25.0) / 100.0
                    product_cost = product.price / (1 + profit_margin)
                    
                    required_budget = production_quantity * product_cost
                    
                    if required_budget > available_budget_for_product:
                        production_quantity = available_budget_for_product / product_cost if product_cost > 0 else 0
                    
                    # 更新库存
                    if production_quantity > 0:
                        old_amount = product.amount
                        product.amount += production_quantity
                        total_output += production_quantity
                        firm_base_production += production_quantity
                        products_restocked += 1
                        
                        print(f"   🏭 固定市场商品: {product.name[:30]} | "
                              f"销量:{inherent_sales:.1f} | 单件利润:${unit_profit:.2f} | "
                              f"优先级:{priority:.1f} | 预算:${available_budget_for_product:.2f} | "
                              f"生产:{production_quantity:.1f}件 | "
                              f"库存:{old_amount:.1f}→{product.amount:.1f}")
            
            # 记录该企业的基础生产量
            firm_production[owner_id] = firm_base_production
            
            print(f"📦 企业 {owner_id}: 当月收入=${current_income:.2f} | 支出=${current_expenses:.2f} | "
                       f"利润=${current_profit:.2f}, 生产预算=${production_budget:.2f}, "
                       f"生产总量={firm_base_production:.1f}件")
        
        return {
            "total_output": total_output,
            "products_restocked": products_restocked,
            "firm_production": firm_production
        }

    async def _decide_research_share_with_llm(
        self, firm, month: int, llm_client=None, model: str = "deepseek-chat"
    ) -> float:
        """
        使用大模型动态决策企业的研发投入比例 ρ

        输入信息：
        - 公司行业（industry）
        - 当月利润（monthly_profit）
        - 毛利率（profit_margin）
        - 政策信号（policy_encourage_innovation）
        - 销量情况（sales_trend）

        Returns:
            float: 研发投入比例 ρ ∈ [0, 1]
        """
        try:
            # 1. 检查创新策略：抑制创新也允许LLM决策，但限制较低上限
            config = self.firm_innovation_config.get(firm.company_id)
            if not config:
                logger.warning(f"企业 {firm.company_id} 没有创新配置，使用默认值")
                return 0.0
            
            strategy = config.innovation_strategy
            is_suppressed = strategy == "suppressed"
            max_research_share = 0.05 if is_suppressed else 0.3

            # 2. 收集企业信息
            industry = getattr(firm, 'main_business', 'Unknown')

            # 获取当月财务数据
            current_financials = self.firm_monthly_financials.get(firm.company_id, {}).get(month, {})
            monthly_income = current_financials.get("income", 0.0)
            monthly_expenses = current_financials.get("expenses", 0.0)
            monthly_profit = monthly_income - monthly_expenses

            # 获取毛利率（优先从配置中获取，否则从行业映射获取）
            profit_margin = config.profit_margin if config.profit_margin is not None else self.category_profit_margins.get(industry, 25.0)

            # 获取政策信号
            policy_signal = strategy == "encouraged"

            # 获取销量趋势（对比上月）
            prev_month = month - 1
            if prev_month > 0:
                prev_financials = self.firm_monthly_financials.get(firm.company_id, {}).get(prev_month, {})
                prev_income = prev_financials.get("income", 0.0)
                if prev_income > 0:
                    sales_trend = ((monthly_income - prev_income) / prev_income) * 100
                else:
                    sales_trend = 0.0
            else:
                sales_trend = 0.0

            # 3. 构建 Prompt
            innovation_status = "suppressed (keep R&D share very small, ideally ≤ 0.05)" if is_suppressed else "encouraged/flexible"
            prompt = f"""You are a strategic advisor for a company making R&D investment decisions.

Company Information:
- Industry: {industry}
- Monthly Profit: ${monthly_profit:.2f}
- Profit Margin: {profit_margin:.1f}%
- Policy Encouragement: {'Yes' if policy_signal else 'No'}
- Innovation Status: {innovation_status}
- Sales Trend (vs last month): {sales_trend:+.1f}%
- Current Month: {month}

Task: Decide what proportion (ρ) of the company's workforce should be allocated to R&D instead of production.

Important Constraints:
- Allocating ρ of workers to R&D reduces current production capacity by the same proportion.
- Successful R&D increases future production capacity, but only probabilistically and with uncertain magnitude.
- Excessively high ρ may severely hurt current output and destabilize the company.
- Too low ρ slows innovation and can cause long-term competitiveness loss.
- You must choose ρ such that the trade-off between short-term production loss and potential long-term gains remains reasonable and sustainable for the company.

Considerations:
1. If profit is negative or very low, prioritize production (low ρ).
2. If policy encourages innovation, consider higher ρ.
3. If sales are declining, innovation may help regain market share.
4. If profit margin is high, the company can afford more R&D.
5. Different industries have different innovation needs.
6. If innovation status is "suppressed", keep ρ extremely small (≤ 0.05) but not zero.
7. Always ensure ρ does not compromise baseline operational production.

Output Format:
Provide ONLY a single number between 0.0 and 1.0 representing the R&D workforce proportion.
Example valid outputs: 0.0, 0.05, 0.1, 0.15, 0.2
Do NOT output any explanation, just the number.
"""

            # 4. 调用大模型（如果提供了client）
            try:
                from openai import AsyncOpenAI
                llm_client = AsyncOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    base_url=os.getenv("BASE_URL", ""),
                    timeout=60.0  # 设置60秒超时
                )
                model = os.getenv("MODEL", "")
                response = await llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strategic business advisor."},
                        {"role": "user", "content": prompt}
                    ]
                )

                # 解析响应
                content = response.choices[0].message.content.strip()
                research_share = float(content)

                # 限制在合理范围内，根据策略调整上限
                research_share = max(0.0, min(max_research_share, research_share))

                print(f"🤖 企业 {firm.company_id} LLM决策: 研发比例={research_share:.1%} "
                            f"(利润=${monthly_profit:.0f}, 趋势={sales_trend:+.1f}%)")

                return research_share

            except Exception as e:
                logger.warning(f"LLM决策失败 {firm.company_id}: {e}, 使用默认规则")

            # 5. 回退方案：基于规则的决策
            print(f"🔍 企业 {firm.company_id} 规则决策: 利润={monthly_profit:.0f}, 毛利率={profit_margin:.1f}%，政策信号={policy_signal}, 销量趋势={sales_trend:+.1f}%")
            research_share = self._decide_research_share_rule_based(
                monthly_profit, profit_margin, policy_signal, sales_trend
            )

            logger.debug(f"📊 企业 {firm.company_id} 规则决策: 研发比例={research_share:.1%}")

            research_share = max(0.0, min(max_research_share, research_share))
            return research_share

        except Exception as e:
            logger.error(f"决策研发比例失败 {firm.company_id}: {e}")
            return 0.0

    def _decide_research_share_rule_based(
        self, monthly_profit: float, profit_margin: float,
        policy_signal: bool, sales_trend: float
    ) -> float:
        """
        基于规则的研发投入决策（LLM的回退方案）

        Returns:
            float: 研发投入比例 ρ ∈ [0, 1]
        """
        # 基础研发比例
        base_share = 0.0

        # 1. 利润足够才考虑研发
        if monthly_profit <= 0:
            return 0.0

        # 2. 政策鼓励创新
        if policy_signal:
            base_share = 0.1  # 10%基础

        # 3. 销量下降，增加研发投入抢市场
        if sales_trend < -5:  # 销量下降超过5%
            base_share += 0.05

        # 4. 高毛利率行业可以承担更多研发
        if profit_margin > 35:
            base_share += 0.03
        elif profit_margin > 25:
            base_share += 0.01

        # 5. 利润很高，可以多投研发
        if monthly_profit > 10000:
            base_share += 0.02

        # 限制在合理范围
        return max(0.0, min(0.25, base_share))

    async def _calculate_effective_labor_force(self, firm, month:int = 0, std_jobs = None) -> Dict[str, float]:
        """
        计算企业的有效劳动力
        根据员工技能与工作要求的匹配度计算有效劳动力系数
        
        Args:
            firm: 企业对象
            month: 月份
        Returns:
            Dict: 包含总员工数、有效劳动力、平均匹配分数等信息
        """
        try:
            # 获取企业所有活跃员工
            employees = firm.get_all_employees()
            if not employees:
                return {
                    "total_employees": 0,
                    "effective_labor": 0.0,
                    "avg_match_score": 0.0,
                    "skill_details": []
                }
            
            total_match_score = 0.0
            skill_details = []
            
            # 计算每个员工的技能匹配度 (针对其具体职位)
            for employee in employees:
                employee_skills = employee.get("skills", {})
                employee_abilities = employee.get("abilities", {})
                job_title = employee.get("job_title", "")
                job_soc = employee.get("job_soc", "")
                
                # 为该员工获取其具体职位的技能要求
                job_requirements = self._get_job_requirements_by_soc(job_soc, std_jobs)
                
                # 计算该员工与其职位要求的匹配分数
                job_skills = job_requirements.get("skills", {})
                job_abilities = job_requirements.get("abilities", {})
                
                
                match_score = self._calculate_skill_match_score(
                    employee_skills, 
                    employee_abilities, 
                    job_skills,
                    job_abilities
                )
                
                total_match_score += match_score
                skill_details.append({
                    "employee": f"{employee.get('household_id')}_{employee.get('lh_type')}",
                    "job_title": job_title,
                    "job_soc": job_soc,
                    "match_score": match_score,
                    "skills_count": len(employee_skills),
                    "abilities_count": len(employee_abilities)
                })
                
                logger.debug(f"员工 {employee.get('household_id')}_{employee.get('lh_type')} ({job_soc}) 技能匹配度: {match_score:.3f}")
            
            # 计算平均匹配分数和有效劳动力
            avg_match_score = total_match_score / len(employees)
            effective_labor = total_match_score  # 有效劳动力 = 所有员工匹配分数之和

            research_share = 0.0
            try:
                research_share = await self._decide_research_share_with_llm(firm, month)
                self.firm_research_share.append({firm.company_id: [research_share, month]})
            except Exception as e:
                logger.error(f"计算企业 {firm.company_id} 研发比例失败: {e}")
                research_share = 0.0

            research_share = max(0.0, min(1.0, research_share))
            production_effective_labor = effective_labor * (1 - research_share)
            research_effective_labor = effective_labor - production_effective_labor

            return {
                "total_employees": len(employees),
                "effective_labor": effective_labor,
                "production_effective_labor": production_effective_labor,
                "research_effective_labor": research_effective_labor,
                "research_share": research_share,
                "avg_match_score": avg_match_score,
                "skill_details": skill_details
            }
            
        except Exception as e:
            logger.error(f"计算企业 {firm.company_id} 有效劳动力失败: {e}")
            return {
                "total_employees": 0,
                "effective_labor": 0.0,
                "avg_match_score": 0.0,
                "skill_details": []
            }
    
    def _get_job_requirements_by_soc(self, soc_code: str, std_jobs = None) -> Dict:
        """
        根据单个SOC Code获取具体职位的技能要求
        
        Args:
            soc_code: O*NET-SOC Code
            std_jobs: 标准工作数据
            
        Returns:
            Dict: 包含skills和abilities要求的字典
        """
        try:
            if std_jobs is None or std_jobs.empty or not soc_code:
                return self._get_default_job_requirements()
            
            # 在std_jobs中查找匹配的工作
            matching_jobs = std_jobs[std_jobs['O*NET-SOC Code'] == soc_code]
            if not matching_jobs.empty:
                job_info = matching_jobs.iloc[0]
                job_skills = job_info.get('skills', {})
                job_abilities = job_info.get('abilities', {})
                
                logger.debug(f"找到SOC {soc_code}的工作要求: {job_info.get('Title', 'Unknown')}")
                
                return {
                    "skills": job_skills if isinstance(job_skills, dict) else {},
                    "abilities": job_abilities if isinstance(job_abilities, dict) else {}
                }
            else:
                logger.debug(f"未找到SOC {soc_code}的工作要求，使用默认要求")
                
        except Exception as e:
            logger.error(f"获取SOC {soc_code}工作要求失败: {e}")


    def _calculate_skill_match_score(self, worker_skills: Dict, worker_abilities: Dict, 
                                   job_skills: Dict, job_abilities: Dict) -> float:
        """
        计算工人技能与工作要求的匹配分数
        返回 0-1 之间的分数，表示匹配度
        """
        total_score = 0
        total_weight = 0
        
        # 计算技能匹配分数
        for skill_name, skill_req in job_skills.items():
            if skill_name in worker_skills:
                required_mean = skill_req.get('mean', 50)
                required_std = skill_req.get('std', 10)
                importance = skill_req.get('importance', 1.0)

                worker_value = worker_skills[skill_name]

                # 计算匹配度，防止除零错误
                if required_std > 0 and required_mean > 0:
                    # 使用标准化距离计算匹配度（类似于jobmarket.py的算法）
                    distance = abs(worker_value - required_mean) / required_std
                    skill_score = max(0, 1 - distance / 3)  # 3个标准差外为0分
                else:
                    # 如果std或mean为0，使用简单比较
                    if required_mean > 0:
                        skill_score = min(worker_value / required_mean, 1.0)
                    else:
                        # 如果要求值为0，使用默认匹配分数
                        skill_score = 0.5

                # 如果importance为0，跳过这个技能
                if importance > 0:
                    total_score += skill_score * importance
                    total_weight += importance
            else:
                # 缺少技能的惩罚
                importance = skill_req.get('importance', 1.0)
                if importance > 0:
                    total_score += 0.3 * importance  # 给予30%的基础分
                    total_weight += importance
        
        # 计算能力匹配分数
        for ability_name, ability_req in job_abilities.items():
            if ability_name in worker_abilities:
                required_mean = ability_req.get('mean', 50)
                required_std = ability_req.get('std', 10)
                importance = ability_req.get('importance', 1.0)

                worker_value = worker_abilities[ability_name]

                # 计算匹配度，防止除零错误
                if required_std > 0 and required_mean > 0:
                    # 使用标准化距离计算匹配度
                    distance = abs(worker_value - required_mean) / required_std
                    ability_score = max(0, 1 - distance / 3)  # 3个标准差外为0分
                else:
                    # 如果std或mean为0，使用简单比较
                    if required_mean > 0:
                        ability_score = min(worker_value / required_mean, 1.0)
                    else:
                        # 如果要求值为0，使用默认匹配分数
                        ability_score = 0.5

                # 如果importance为0，跳过这个能力
                if importance > 0:
                    total_score += ability_score * importance
                    total_weight += importance
            else:
                # 缺少能力的惩罚
                importance = ability_req.get('importance', 1.0)
                if importance > 0:
                    total_score += 0.3 * importance  # 给予30%的基础分
                    total_weight += importance
        
        # 返回加权平均分数
        return total_score / total_weight if total_weight > 0 else 0.5

    async def _execute_labor_based_production(
        self, month: int, sales_data: Dict, labor_market, firms: List = None, std_jobs = None, production_config: Dict = None, innovation_config: Dict = None
    ) -> Dict[str, Any]:
        """
        为有工人的公司执行基于劳动力的额外生产
        考虑员工技能匹配度计算有效劳动力
        """
        total_output = 0.0
        companies_with_workers = 0
        firm_labor_efficiency = {}  # 记录每家企业的劳动效率
        firm_labor_production = {}  # 记录每家企业的劳动力生产量

        firm_research_labor = {}
        total_research_effective_labor = 0.0
        policy_signal = None if innovation_config is None else innovation_config.get("policy_signal", None)
        
        # 创新模块：记录每家企业创新到达率和到达次数
        firm_innovation_arrival_rate = {}  # Λ_t = λ * (research_effective_labor)^beta
        firm_innovation_arrivals = {}  # 泊松采样得到的创新到达次数
        try:
            # 计算每家企业的有效劳动力
            if firms:
                for firm in firms:
                    if firm.get_employees() > 0:  # 只处理有员工的企业
                        try:
                            effective_labor = await self._calculate_effective_labor_force(firm, month, std_jobs)
                            firm_labor_efficiency[firm.company_id] = effective_labor
                            # jiaju_add_4 start 计算创新到达率和次数 核心代码
                            if policy_signal is not None:
                                effective_labor['policy_signal'] = policy_signal
                            research_eff = effective_labor.get('research_effective_labor', 0.0)
                            firm_research_labor[firm.company_id] = research_eff
                            total_research_effective_labor += research_eff
                            
                            # 计算创新到达率 Λ_t = λ * (research_effective_labor)^beta
                            if innovation_config and innovation_config.get('enable_innovation_module', False):
                                innovation_lambda = innovation_config.get('innovation_lambda', 0.05)
                                innovation_beta = innovation_config.get('innovation_concavity_beta', 0.6)
                                
                                # Λ_t = λ * (effective_research_labor)^beta
                                if research_eff > 0:
                                    innovation_arrival_rate = innovation_lambda * (research_eff ** innovation_beta)
                                else:
                                    innovation_arrival_rate = 0.0
                                
                                # 限制到达率在合理范围内（避免过大）
                                innovation_arrival_rate = min(innovation_arrival_rate, 10.0)  # 最大每月10次
                                
                                firm_innovation_arrival_rate[firm.company_id] = innovation_arrival_rate
                                
                                # 泊松采样：从泊松分布中采样创新到达次数
                                # 泊松分布的参数为 Λ_t，采样结果表示本月创新发生的次数（非负整数）
                                # 注：P(至少发生1次) = 1 - exp(-Λ_t)，但这里我们直接采样次数（innovation_arrivals为非负整数）
                                if innovation_arrival_rate > 0:
                                    innovation_arrivals = np.random.poisson(innovation_arrival_rate)
                                else:
                                    innovation_arrivals = 0
                                
                                firm_innovation_arrivals[firm.company_id] = innovation_arrivals
                                
                                print(
                                    f"🔬 企业 {firm.company_id} 创新: 研发有效劳动力={research_eff:.2f}, "
                                    f"到达率Λ_t={innovation_arrival_rate:.4f}, 本月到达次数={innovation_arrivals}"
                                )
                            # jiaju_add_4 end
                            print(f"🏭 企业 {firm.company_id} 有效劳动力: {effective_labor['effective_labor']:.2f} (员工数: {firm.get_employees()})")
                        except Exception as e:
                            logger.error(f"计算企业 {firm.company_id} 劳动效率失败: {e}")
                            firm_labor_efficiency[firm.company_id] = {"total_employees": 0, "effective_labor": 0.0, "avg_match_score": 0.0}
                            # 初始化创新到达次数为0，避免后续访问时出现KeyError
                            firm_innovation_arrivals[firm.company_id] = 0
            
            # 获取所有有工人的公司
            companies_with_employees = await self._get_companies_with_employees(labor_market)
            
            for company_id, employee_count in companies_with_employees.items():
                if employee_count == 0:
                    continue
                    
                companies_with_workers += 1
                # jiaju_add_5 start 获取信息
                # 获取该企业的有效劳动力信息
                labor_info = firm_labor_efficiency.get(
                    company_id, {"effective_labor": employee_count, "avg_match_score": 1.0}
                )
                production_labor = labor_info.get("production_effective_labor", labor_info.get("effective_labor", employee_count))
                research_share = labor_info.get("research_share", 0.0)

                # 获取该企业的创新到达次数，如果不存在则默认为0
                innovation_arrivals = firm_innovation_arrivals.get(company_id, 0)

                # 如果有创新到达，先处理创新到达（更新 labor_productivity_factor），以便影响本月生产
                if innovation_arrivals > 0 and innovation_config and innovation_config.get('enable_innovation_module', False):
                    await self.handle_innovation_arrival(company_id, month, innovation_arrivals, innovation_config)
                
                # 计算该公司的劳动力产出 (使用有效劳动力而不是员工数量)
                # 注意：如果有创新到达，这里会使用更新后的 labor_productivity_factor
                company_output = await self._calculate_company_labor_production(
                    company_id, production_labor * (1 - research_share), sales_data, production_config
                )

                total_output += company_output
                firm_labor_production[company_id] = company_output  # 记录该企业的劳动力生产量

                logger.debug(
                    f"劳动力生产: 公司 {company_id} 员工 {employee_count} 人，产出 {company_output:.2f} | 研发份额 {research_share:.2f} | 创新到达 {innovation_arrivals}"
                )

        except Exception as e:
            logger.warning(f"劳动力生产计算失败: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "total_output": total_output,
            "companies_count": companies_with_workers,
            "firm_labor_efficiency": firm_labor_efficiency,
            "firm_labor_production": firm_labor_production,  # 新增：每个企业的劳动力生产量
            "firm_research_labor": firm_research_labor,
            "total_research_effective_labor": total_research_effective_labor,
            "firm_innovation_arrival_rate": firm_innovation_arrival_rate,  # 创新到达率 Λ_t
            "firm_innovation_arrivals": firm_innovation_arrivals  # 泊松采样的创新到达次数
        }
        # jiaju_add_5 end

    async def handle_innovation_arrival(self, company_id: str, month: int, innovation_arrivals: int, innovation_config: Dict = None):
        """
        处理创新到达 随机选择三种方式
        1. 降价+提高产量（原先的update_prices_innovation_arrival方法）
        2. 涨价+提升商品属性
        3. 提高毛利率
        """
        if innovation_arrivals > 0:
            for i in range(innovation_arrivals):
                innovation_type = random.choice([1, 2, 3])
                if innovation_type == 1:
                    await self.update_prices_innovation_arrival(company_id, innovation_config.get('innovation_gamma', 1.2), month)
                elif innovation_type == 2:
                    await self.update_product_attributes_innovation_arrival(company_id, innovation_config.get('innovation_gamma', 1.2), month)
                elif innovation_type == 3:
                    await self.update_profit_margin_innovation_arrival(company_id, innovation_config.get('innovation_gamma', 1.2), month)

    async def update_prices_innovation_arrival(self, company_id: str, gamma: float = 1.2, month: int = 0):
        """
        更新公司价格和创新到达次数
        """
        if company_id not in self.products or not self.products[company_id]:
            return
        price_change = np.sqrt(gamma) 
        for product in self.products[company_id]:
            product.price = product.price * (1/price_change)
        print(f"🔬 公司 {company_id} {month}月价格变化 {price_change}")

        if company_id not in self.firm_innovation_config:
            logger.warning(f"公司 {company_id} 没有创新配置，无法更新劳动力因素")
            return
        
        config = self.firm_innovation_config[company_id]
        old_labor_production = config.labor_productivity_factor
        new_labor_production = old_labor_production * gamma
        print(f"🔬 公司 {company_id} {month}月劳动力因素变化 {old_labor_production} -> {new_labor_production}")
        
        # 直接更新字典中存储的对象属性，确保修改立即生效
        self.firm_innovation_config[company_id].labor_productivity_factor = new_labor_production

        self.add_innovation_event(
            company_id=company_id,
            month=month,
            innovation_type='price',
            price_change=price_change
        )
        self.add_innovation_event(
            company_id=company_id,
            month=month,
            innovation_type='labor_productivity_factor',
            old_value=old_labor_production,
            new_value=new_labor_production
        )

    async def update_product_attributes_innovation_arrival(self, company_id: str, gamma: float = 1.2, month: int = 0):
        """
        更新公司商品属性
        """
        if company_id not in self.products or not self.products[company_id]:
            return
        
        def _scale_numeric_fields(payload: Any, multiplier: float):
            """
            递归放大字典/列表中的数值字段，保持其余结构不变。
            """
            if isinstance(payload, dict):
                return {k: _scale_numeric_fields(v, multiplier) for k, v in payload.items()}
            if isinstance(payload, list):
                return [_scale_numeric_fields(v, multiplier) for v in payload]
            if isinstance(payload, (int, float)):
                return payload * multiplier
            return payload

        updated_products = 0
        for product in self.products[company_id]:
            before_snapshot = {
                "attributes": copy.deepcopy(product.attributes) if isinstance(product.attributes, (dict, list)) else product.attributes,
                "nutrition": copy.deepcopy(product.nutrition_supply) if isinstance(product.nutrition_supply, (dict, list)) else product.nutrition_supply,
                "satisfaction": copy.deepcopy(product.satisfaction_attributes) if isinstance(product.satisfaction_attributes, (dict, list)) else product.satisfaction_attributes,
            }

            if product.attributes:
                product.attributes = _scale_numeric_fields(product.attributes, gamma)
            if product.nutrition_supply:
                product.nutrition_supply = _scale_numeric_fields(product.nutrition_supply, gamma)
            if product.satisfaction_attributes:
                product.satisfaction_attributes = _scale_numeric_fields(product.satisfaction_attributes, gamma)

            # 如果有任何字段发生变化，则记录
            if before_snapshot["attributes"] != product.attributes or \
               before_snapshot["nutrition"] != product.nutrition_supply or \
               before_snapshot["satisfaction"] != product.satisfaction_attributes:
                updated_products += 1

        if updated_products > 0:
            print(f"🔬 公司 {company_id} {month}月商品属性提升: 放大系数={gamma}, 受影响商品={updated_products} 件")
            self.add_innovation_event(
                company_id=company_id,
                month=month,
                innovation_type='attribute',
                attribute_change=gamma
            )

    async def update_profit_margin_innovation_arrival(self, company_id: str, gamma: float = 1.2, month: int = 0):
        """
        更新公司毛利率
        """
        if company_id not in self.firm_innovation_config:
            logger.warning(f"公司 {company_id} 没有创新配置，无法更新毛利率")
            return
        
        config = self.firm_innovation_config[company_id]
        if config.profit_margin is None:
            logger.warning(f"公司 {company_id} 毛利率为None，无法更新")
            return
        
        old_profit_margin = config.profit_margin
        new_profit_margin = old_profit_margin * gamma
        print(f"🔬 公司 {company_id} {month}月毛利率变化 {old_profit_margin} -> {new_profit_margin}")
        
        # 直接更新字典中存储的对象属性，确保修改立即生效
        self.firm_innovation_config[company_id].profit_margin = new_profit_margin

        self.add_innovation_event(
            company_id=company_id,
            month=month,
            innovation_type='profit_margin',
            old_value=old_profit_margin,
            new_value=new_profit_margin
        )

    async def _get_companies_with_employees(self, labor_market) -> Dict[str, int]:
        """
        获取所有有员工的公司及其员工数量
        """
        companies_employees = {}
        
        try:
            # 从劳动力市场获取所有匹配的工作
            matched_jobs = await labor_market.query_matched_jobs.remote()
            
            # 统计每个公司的员工数量
            for job in matched_jobs:
                company_id = job.company_id
                companies_employees[company_id] = companies_employees.get(company_id, 0) + 1
        
        except Exception as e:
            logger.warning(f"获取公司员工数据失败: {e}")
        
        return companies_employees

    async def _calculate_company_labor_production(
        self, company_id: str, employee_count: int, sales_data: Dict, production_config: Dict = None
    ) -> float:
        """
        计算单个公司基于劳动力的产出
        使用简化的柯布-道格拉斯生产函数: Q = A × L^α
        如果创新到达次数大于0，则使用创新阶梯函数: Q = A × (gamma ** innovation_arrivals)
        
        Args:
            employee_count: 有效劳动力数量
            production_config: 生产配置参数
        """
        if company_id not in self.products or not self.products[company_id]:
            return 0.0
        
        labor_elasticity = production_config.get('labor_elasticity', 0.7) if production_config else 0.7
        config = self.firm_innovation_config.get(company_id)
        if not config:
            # 如果没有创新配置，使用默认值
            firm_productivity_factor = production_config.get('labor_productivity_factor', 30.0) if production_config else 30.0
        else:
            firm_productivity_factor = config.labor_productivity_factor        

        # 计算总的劳动力产出: Q = A × L^α
        total_labor_output = firm_productivity_factor * (employee_count ** labor_elasticity)
        
        # 根据销量情况分配产出到不同产品
        # 🔧 优先按照"家庭购买过的商品"的销量占比进行分配；
        #    若当月无任何家庭购买记录，则回退到原有的销量/库存优先级规则。
        company_products = self.products[company_id]
        product_priorities = {}
        household_sum = 0.0
        
        # 计算每个产品的优先级
        for product in company_products:
            product_id = product.product_id
            sales_key = (product_id, company_id)
            
            # 计算优先级分数（基于销量和库存水平）
            # 使用 (product_id, company_id) 作为key查找销量数据
            if sales_key in sales_data:
                # 有销售记录：基于销量计算优先级
                sales_info = sales_data[sales_key]
                quantity_sold = sales_info.get("quantity_sold", 0)
                demand_level = sales_info.get("demand_level", "normal")
                
                # 计算优先级分数
                priority_score = quantity_sold
                if demand_level == "high":
                    priority_score *= 2.0
                elif demand_level == "low":
                    priority_score *= 0.5
                
                product_priorities[product_id] = priority_score
                hh_qty = float(sales_info.get("household_quantity", 0.0) or 0.0)
                household_sum += hh_qty
            else:
                # 🔧 修改：无销售记录的商品也参与劳动力生产（可能是库存为0）
                # 基于库存水平计算优先级
                if product.amount == 0:
                    # 库存为0的商品，给予中等优先级（相当于销量10）
                    priority_score = 10.0
                elif product.amount < 50:
                    # 低库存商品，给予较低优先级（相当于销量5）
                    priority_score = 5.0
                else:
                    # 高库存商品，给予最低优先级（相当于销量1）
                    priority_score = 1.0
                
                product_priorities[product_id] = priority_score
                logger.debug(f"劳动力生产: {product.name} (无销售记录, 库存{product.amount:.1f}, 优先级{priority_score})")
        
        # 若有家庭购买记录，则按家庭销量占比分配；
        # 否则回退到基于销量/库存的优先级逻辑。
        if household_sum == 0.0:
            product_priorities = {}
            for product in company_products:
                product_id = product.product_id
                sales_key = (product_id, company_id)
                if sales_key in sales_data:
                    sales_info = sales_data[sales_key]
                    quantity_sold = sales_info.get("quantity_sold", 0)
                    demand_level = sales_info.get("demand_level", "normal")
                    priority_score = quantity_sold
                    if demand_level == "high":
                        priority_score *= 2.0
                    elif demand_level == "low":
                        priority_score *= 0.5
                    product_priorities[product_id] = priority_score
                else:
                    if product.amount == 0:
                        priority_score = 10.0
                    elif product.amount < 50:
                        priority_score = 5.0
                    else:
                        priority_score = 1.0
                    product_priorities[product_id] = priority_score
                    logger.debug(f"劳动力生产: {product.name} (无销售记录, 库存{product.amount:.1f}, 优先级{priority_score})")
        
        # 按优先级分配产出
        total_priority = sum(product_priorities.values())
        actual_output = 0.0
        
        if total_priority > 0:
            for product in company_products:
                product_id = product.product_id
                
                # 只处理有优先级的产品（现在所有产品都有优先级）
                if product_id not in product_priorities:
                    continue
                
                priority = product_priorities[product_id]
                
                # 计算该产品应得的产出
                product_output = total_labor_output * (priority / total_priority)
                
                # 增加库存
                old_amount = product.amount
                product.amount += product_output
                actual_output += product_output
                
                logger.debug(f"劳动力产出: {product.name} 优先级 {priority:.2f}, 增加 {product_output:.2f}")
        else:
            # 这种情况理论上不应该发生，因为所有产品都会有优先级
            logger.warning(f"公司 {company_id} 没有产品可以分配劳动力产出")
        
        return actual_output

    def get_production_statistics(self, month: int) -> Dict[str, Any]:
        """
        获取生产统计数据
        """
        stats = {
            "total_companies": len([owner_id for owner_id in self.company_id if owner_id in self.products and self.products[owner_id]]),
            "total_products": sum(len(products) for products in self.products.values()),
            "total_inventory": 0.0,
            "products_by_category": {},
            "low_stock_products": [],
            "high_stock_products": []
        }
        
        # 统计库存情况
        for owner_id, products in self.products.items():
            if owner_id in self.company_id:    
                for product in products:
                    stats["total_inventory"] += product.amount
                    
                    # 按分类统计
                    category = product.classification or "other"
                    if category not in stats["products_by_category"]:
                        stats["products_by_category"][category] = {"count": 0, "inventory": 0.0}
                    
                    stats["products_by_category"][category]["count"] += 1
                    stats["products_by_category"][category]["inventory"] += product.amount
                    
                    # 识别库存异常的产品
                    if product.amount < 5:
                        stats["low_stock_products"].append({
                            "name": product.name,
                            "amount": product.amount,
                            "owner": owner_id
                        })
                    elif product.amount > 80:
                        stats["high_stock_products"].append({
                            "name": product.name,
                            "amount": product.amount,
                            "owner": owner_id
                        })
        
        return stats

    async def update_tax_rates(self, income_tax_rate: float = None, vat_rate: float = None, corporate_tax_rate: float = None):
        """
        更新税率
        """
        if income_tax_rate is not None:
            self.income_tax_rate = income_tax_rate
        if vat_rate is not None:
            self.vat_rate = vat_rate
        if corporate_tax_rate is not None:
            self.corporate_tax_rate = corporate_tax_rate

        logger.info(f"税率已更新: income_tax_rate={self.income_tax_rate:.1%}, vat_rate={self.vat_rate:.1%}, corporate_tax_rate={self.corporate_tax_rate:.1%}")

# ======================== 创新系统相关方法 ========================

    def register_firm_innovation_config(self, firm, strategy: str, labor_productivity_factor: float, fund_share: float = 0.0):
        """
        注册企业的创新策略

        Args:
            company_id: 企业ID
            strategy: 创新策略 ("encouraged" 或 "suppressed")
            research_share: 研发投入比例（0-1之间的浮点数）
        """
        # 根据企业的行业（main_business）设置毛利率
        # main_business 通常对应商品分类（daily_cate）
        profit_margin = self._get_profit_margin(firm.main_business)
        
        self.firm_innovation_config[firm.company_id] = FirmInnovationConfig(
            company_id=firm.company_id,
            innovation_strategy=strategy,
            labor_productivity_factor=labor_productivity_factor,
            profit_margin=profit_margin,
            fund_share=fund_share
        )
        
        logger.info(f"✅ 企业 {firm.company_id} 创新策略: {strategy}, 研发比例: {fund_share:.1%}, 毛利率: {profit_margin:.1f}%")

    def query_firm_innovation_config(self, company_id: str) -> FirmInnovationConfig:
        """
        查询企业的创新策略

        Returns:
            FirmInnovationConfig: FirmInnovationConfig对象
        """
        return self.firm_innovation_config[company_id]

    def add_innovation_event(self, **kwargs):
        """
        添加创新事件记录

        Args:
            **kwargs: 创新事件数据
        """
        self.firm_innovation_events.append(FirmInnovationEvent.create(**kwargs))



    def query_all_firm_innovation_events(self) -> List[FirmInnovationEvent]:
        """
        查询所有创新事件

        Returns:
            List: 创新事件列表
        """
        return self.firm_innovation_events


    def query_production_stats_by_month(self, month: int) -> Dict[str, Any]:
        """查询并返回某个月份的生产统计（包含劳动与创新细节）。若无则返回空字典。"""
        return self.production_stats_by_month.get(month, {})