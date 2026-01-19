import json
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import ray

# Relative imports for internal modules
from ..center.model import Asset, Transaction
from ..center.ecocenter import EconomicCenter
from ..logger import get_logger

logger = get_logger(__name__)

class SavingsAccount(BaseModel):
    """
    # 储蓄账户
    代表家庭在银行的储蓄账户信息
    
    ## Properties
    - **account_id**: 账户ID
    - **household_id**: 家庭ID
    - **balance**: 账户余额
    - **annual_interest_rate**: 年利率
    - **last_interest_date**: 上次计息日期
    """
    account_id: str
    household_id: str
    balance: float = Field(default=0.0, ge=0.0)
    annual_interest_rate: float = Field(default=0.005, ge=0.0, le=1.0)  # 0.5% 年利率
    last_interest_date: Optional[int] = Field(default=None, description="上次计息的月份")
    created_month: int = Field(default=1, description="账户创建月份")

@ray.remote
class Bank:
    """
    # 银行代理
    管理家庭储蓄和利息发放的银行系统
    
    ## Features
    - 储蓄账户管理
    - 月度利息计算和发放
    - 存款和取款操作
    - 与经济中心集成
    """
    
    def __init__(self,
                 bank_id: str = "central_bank",
                 initial_capital: float = 1000000.0,  # 银行初始资本
                 economic_center: Optional[EconomicCenter] = None):
        """
        ## 初始化银行代理
        
        ### Parameters
        - `bank_id` (str): 银行唯一标识符
        - `initial_capital` (float): 银行初始资本
        - `economic_center` (EconomicCenter): 经济中心引用
        """
        if not bank_id or not isinstance(bank_id, str):
            raise ValueError("bank_id must be a non-empty string")
        
        self.bank_id = bank_id
        self.initial_capital = initial_capital
        self.economic_center = economic_center
        
        # 储蓄账户管理
        self.savings_accounts: Dict[str, SavingsAccount] = {}  # household_id -> SavingsAccount
        
        # 银行统计数据
        self.total_deposits = 0.0
        self.total_interest_paid = 0.0
        self.interest_history: List[Dict] = []
        
        self.logger = get_logger(__name__)
        self.initialized = False
    
    async def initialize(self):
        """
        ## 初始化银行代理
        在经济中心注册银行账本和产品
        """
        if self.economic_center:
            try:
                await asyncio.gather(
                    self.economic_center.init_agent_ledger.remote(self.bank_id, self.initial_capital),
                    self.economic_center.register_id.remote(self.bank_id, 'bank'),
                    self.economic_center.init_agent_product.remote(self.bank_id)
                )
                self.logger.info(f"Bank {self.bank_id} registered in EconomicCenter with capital ${self.initial_capital:.2f}")
                print(f"Bank {self.bank_id} registered in EconomicCenter with capital ${self.initial_capital:.2f}")
            except Exception as e:
                self.logger.warning(f"[Bank Init] Failed to register bank {self.bank_id}: {e}")
        
        self.initialized = True
    
    def create_savings_account(self, household_id: str, current_month: int = 1) -> str:
        """
        为家庭创建储蓄账户
        
        Args:
            household_id: 家庭ID
            current_month: 当前月份
            
        Returns:
            str: 账户ID
        """
        if household_id in self.savings_accounts:
            self.logger.info(f"Savings account already exists for household {household_id}")
            return self.savings_accounts[household_id].account_id
        
        account_id = f"savings_{household_id}_{current_month}"
        account = SavingsAccount(
            account_id=account_id,
            household_id=household_id,
            balance=0.0,
            created_month=current_month,
            last_interest_date=current_month
        )
        
        self.savings_accounts[household_id] = account
        # self.logger.info(f"Created savings account {account_id} for household {household_id}")
        return account_id

    async def update_deposit(self, household_id: str, amount: float):
        """
        更新家庭储蓄账户余额
        """
        if household_id not in self.savings_accounts:
            self.create_savings_account(household_id)
        self.savings_accounts[household_id].balance = amount
    
    async def deposit(self, household_id: str, amount: float, month: int) -> bool:
        """
        家庭存款到储蓄账户
        
        Args:
            household_id: 家庭ID
            amount: 存款金额
            month: 当前月份
            
        Returns:
            bool: 存款是否成功
        """
        if amount <= 0:
            return False
        
        # 检查家庭余额是否足够
        household_balance = await self.economic_center.query_balance.remote(household_id)
        if household_balance < amount:
            self.logger.warning(f"Household {household_id} insufficient balance for deposit: ${household_balance:.2f} < ${amount:.2f}")
            return False
        
        # 如果没有储蓄账户，创建一个
        if household_id not in self.savings_accounts:
            self.create_savings_account(household_id, month)
        
        # 转移资金：家庭 -> 银行
        try:
            tx_id = await self.economic_center.add_tx_service.remote(
                month=month,
                sender_id=household_id,
                receiver_id=self.bank_id,
                amount=amount
            )
            
            # 更新储蓄账户余额
            self.savings_accounts[household_id].balance += amount
            self.total_deposits += amount
            
            self.logger.info(f"Household {household_id} deposited ${amount:.2f} to bank")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process deposit for household {household_id}: {e}")
            return False
    
    async def withdraw(self, household_id: str, amount: float, month: int) -> bool:
        """
        家庭从储蓄账户取款
        
        Args:
            household_id: 家庭ID
            amount: 取款金额
            month: 当前月份
            
        Returns:
            bool: 取款是否成功
        """
        if amount <= 0:
            return False
        
        if household_id not in self.savings_accounts:
            self.logger.warning(f"No savings account found for household {household_id}")
            return False
        
        account = self.savings_accounts[household_id]
        if account.balance < amount:
            self.logger.warning(f"Insufficient savings balance for household {household_id}: ${account.balance:.2f} < ${amount:.2f}")
            return False
        
        # 转移资金：银行 -> 家庭
        try:
            tx_id = await self.economic_center.add_tx_service.remote(
                month=month,
                sender_id=self.bank_id,
                receiver_id=household_id,
                amount=amount
            )
            
            # 更新储蓄账户余额
            account.balance -= amount
            self.total_deposits -= amount
            
            self.logger.info(f"Household {household_id} withdrew ${amount:.2f} from bank")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process withdrawal for household {household_id}: {e}")
            return False
    
    async def calculate_and_pay_monthly_interest(self, month: int) -> float:
        """
        计算并发放月度利息
        年利率0.5%，按月计算：月利率 = 0.5% / 12 ≈ 0.0417%
        
        Args:
            month: 当前月份
            
        Returns:
            float: 总利息支付金额
        """
        monthly_interest_rate = 0.005 / 12  # 年利率0.5%转换为月利率
        total_interest_paid = 0.0
        
        for household_id, account in self.savings_accounts.items():
            if account.balance <= 0:
                continue
            
            # 计算利息
            interest_amount = account.balance * monthly_interest_rate
            
            if interest_amount > 0:
                try:
                    # 银行支付利息给家庭
                    tx_id = await self.economic_center.add_interest_tx.remote(
                        month=month,
                        sender_id=self.bank_id,
                        receiver_id=household_id,
                        amount=interest_amount
                    )
  
                    total_interest_paid += interest_amount
                    
                    # 记录利息历史
                    self.interest_history.append({
                        "month": month,
                        "household_id": household_id,
                        "principal": account.balance - interest_amount,
                        "interest": interest_amount,
                        "new_balance": account.balance
                    })
                    
                    self.logger.debug(f"Paid ${interest_amount:.4f} interest to household {household_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to pay interest to household {household_id}: {e}")
        
        self.total_interest_paid += total_interest_paid
        if total_interest_paid > 0:
            print(f"Month {month}: Paid total interest ${total_interest_paid:.2f} to {len([a for a in self.savings_accounts.values() if a.balance > 0])} accounts")
        
        return total_interest_paid
    
    def get_account_summary(self, household_id: str) -> Optional[Dict]:
        """
        获取家庭储蓄账户摘要
        
        Args:
            household_id: 家庭ID
            
        Returns:
            Dict: 账户摘要信息
        """
        if household_id not in self.savings_accounts:
            return None
        
        account = self.savings_accounts[household_id]
        return {
            "account_id": account.account_id,
            "household_id": account.household_id,
            "balance": account.balance,
            "annual_interest_rate": account.annual_interest_rate,
            "last_interest_date": account.last_interest_date,
            "created_month": account.created_month
        }
    
    def get_bank_summary(self) -> Dict:
        """
        获取银行整体摘要信息
        
        Returns:
            Dict: 银行摘要
        """
        active_accounts = len([a for a in self.savings_accounts.values() if a.balance > 0])
        return {
            "bank_id": self.bank_id,
            "total_accounts": len(self.savings_accounts),
            "active_accounts": active_accounts,
            "total_deposits": self.total_deposits,
            "total_interest_paid": self.total_interest_paid,
            "average_balance": self.total_deposits / max(active_accounts, 1)
        }
