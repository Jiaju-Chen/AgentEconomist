from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal, Dict, Optional, List, Callable
from uuid import uuid4
from datetime import date, datetime

class TaxPolicy(BaseModel):
    """
    # Tax Policy
    Represents a comprehensive tax policy with three core tax types.
    
    ## Properties
    - **income_tax_rate**: Personal income tax rate (0-1) - 个人所得税
    - **corporate_tax_rate**: Corporate income tax rate (0-1) - 企业所得税
    - **vat_rate**: Value-added tax rate (0-1) - 消费税
    """
    income_tax_rate: float = Field(default=0.225, ge=0.0, le=1.0)  # 22.5% 个人所得税
    corporate_tax_rate: float = Field(default=0.21, ge=0.0, le=1.0)  # 21% 企业所得税
    vat_rate: float = Field(default=0.08, ge=0.0, le=1.0)  # 8% 消费税
    
    @model_validator(mode='after')
    def validate_tax_rates(self):
        # Ensure all tax rates are valid
        for rate_name in ['income_tax_rate', 'corporate_tax_rate', 'vat_rate']:
            rate = getattr(self, rate_name)
            if not 0 <= rate <= 1:
                raise ValueError(f"{rate_name.replace('_', ' ').title()} must be between 0 and 1")
        return self

class Asset(BaseModel):
    # the same goods should share the same asset_id
    name: Optional[str] = None
    asset_type: Literal['money', 'goods', 'labor_hour', 'security'] = Field(..., description="Type of the asset")
    classification: Optional[str] = Field(None, description="classification of the asset")
    expiration_date: Optional[date] = Field(None, description="expiration date of the asset")
    manufacturer: Optional[str] = Field(None, description="manufacturer of the asset; could be None for labor hour and money") 
    price: Optional[float] = Field(None, gt=0, description="price of the asset; could be None for labor hour and money")
    amount: float = Field(..., gt=0, description="amount of the asset")
    description: Optional[str] = Field(None, description="description of the asset")

class Ledger(Asset):
    asset_type: Literal['money'] = Field(default='money', description="Type of the asset")
    amount: float = Field(..., ge=0, description="Amount of money")
    
    @classmethod
    def create(cls, agent_id: str, amount: float = 0.0) -> 'Ledger':
        if amount < 0:
            raise ValueError("Initial amount cannot be negative")
        return cls(agent_id=agent_id, asset_type='money', amount=amount)
    
class LaborHour(BaseModel):
    agent_id: str  
    skill_profile: Dict[str, float] = Field(None, description="Type of skill")
    ability_profile: Dict[str, float] = Field(None, description="Type of ability")
    asset_type: Literal['labor_hour'] = Field(default='labor_hour', description="Type of the asset")
    total_hours: float = Field(..., gt=0, description="Total hours available")
    start_date: Optional[date] = Field(None, description="Start date")
    lh_type: Literal['head', 'spouse'] = Field(default='head', description="Type of the labor hour")
    is_recurring: bool = Field(default=False, description="Whether the labor is recurring")
    cycle: Optional[Literal['daily', 'weekly', 'monthly']] = Field(
        None, description="If recurring, specify the cycle frequency"
    )
    template: str
    daily_hours: Optional[float] = Field(None, description="Daily work hours (calculated if possible)")
    is_valid: bool = Field(default=True, description="Whether the labor hour is valid")
    job_title: Optional[str] = Field(None, description="The ID of the job")
    job_SOC: Optional[str] = Field(None, description="The SOC of the job")
    company_id: Optional[str] = Field(None, description="The company ID that posted the job")
    @model_validator(mode='after')
    def compute_daily_hours(self) -> 'LaborHour':
        if self.is_recurring:
            self.daily_hours = None  # Optional: could compute default if needed
        elif self.start_date and self.end_date:
            days = (self.end_date - self.start_date).days + 1
            if days <= 0:
                raise ValueError("End date must be after start date")
            self.daily_hours = self.total_hours / days
        else:
            self.daily_hours = None
        return self

    @classmethod
    def create(cls, agent_id: str, total_hours: float, template:str, skill_profile: Dict[str, float] = None,
               ability_profile: Dict[str, float] = None,
               start_date: Optional[date] = None, end_date: Optional[date] = None,
               is_recurring: bool = False, cycle: Optional[Literal['daily', 'weekly', 'monthly']] = None, lh_type: Literal['head', 'spouse'] = 'head') -> 'LaborHour':
        if total_hours <= 0:
            raise ValueError("Total hours must be greater than zero")
        return cls(
            agent_id=agent_id,
            skill_profile=skill_profile,
            ability_profile=ability_profile,
            total_hours=total_hours,
            start_date=start_date,
            end_date=end_date,
            is_recurring=is_recurring,
            cycle=cycle,
            template=template,
            lh_type=lh_type
        )

class Wage(BaseModel):
    agent_id: str
    amount: float
    month: int
    @classmethod
    def create(cls, agent_id: str, amount: float, month: int) -> 'Wage':
        return cls(
            agent_id=agent_id,
            amount=amount,
            month=month
        )

class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique job identifier")
    SOC: str = Field(..., description="Standard Occupational Classification code")
    title: str
    description: Optional[str] = None
    wage_per_hour: float = Field(..., description="Posted wage by the company")
    required_skills: Dict[str, Dict[str, float]] = Field(None, description="Required skills for the job")
    required_abilities: Dict[str, Dict[str, float]] = Field(None, description="Required abilities for the job")

    company_id: Optional[str]   # The company ID that posted the job
    is_valid: bool = Field(default=True, description="Whether the job is currently available")
    positions_available: int = Field(default=1, description="Number of positions available for this job")
    hours_per_period: Optional[float] = Field(None, description="Hours per work period")

    @classmethod
    def create(cls, soc: str, title: str, wage_per_hour: float,
                company_id: Optional[str] = None, description: Optional[str] = None,
                hours_per_period: Optional[float] = None,
                required_skills: Optional[Dict[str, Dict[str, float]]] = None, 
                required_abilities: Optional[Dict[str, Dict[str, float]]] = None,
                job_id: Optional[str] = None) -> 'Job':
          return cls(
                job_id=job_id or str(uuid4()),
                SOC=soc,
                title=title,
                wage_per_hour=wage_per_hour,
                company_id=company_id,
                description=description,
                hours_per_period=hours_per_period,
                required_skills=required_skills or {},
                required_abilities=required_abilities or {}
          )

class JobApplication(BaseModel):
    job_id: str = Field(..., description="ID of the job being applied for")
    household_id: str = Field(..., description="ID of the household applying")
    lh_type: Literal['head', 'spouse'] = Field(..., description="Type of labor hour (head or spouse)")
    expected_wage: float = Field(..., description="Expected wage by the job seeker")
    worker_skills: Dict[str, float] = Field(default_factory=dict, description="Worker's skill profile")
    worker_abilities: Dict[str, float] = Field(default_factory=dict, description="Worker's ability profile")
    application_timestamp: datetime = Field(default_factory=datetime.now, description="When the application was submitted")
    month:int = Field(default=1, description="Month of the application")
    @classmethod
    def create(cls, job_id: str, household_id: str, lh_type: Literal['head', 'spouse'],
               expected_wage: float, worker_skills: Optional[Dict[str, float]] = None,
               worker_abilities: Optional[Dict[str, float]] = None, month:int = 1) -> 'JobApplication':
        return cls(
            job_id=job_id,
            household_id=household_id,
            lh_type=lh_type,
            expected_wage=expected_wage,
            worker_skills=worker_skills or {},
            worker_abilities=worker_abilities or {},
            month=month
        )

class MatchedJob(BaseModel):
    job: Job
    average_wage: float
    household_id: str
    lh_type: Literal['head', 'spouse']
    company_id: str
    @classmethod
    def create(cls, job: Job, average_wage: float, household_id: str, lh_type: Literal['head', 'spouse'], company_id: str) -> 'MatchedJob':
        return cls(
            job=job,
            average_wage=average_wage,
            household_id=household_id,
            lh_type=lh_type,
            company_id=company_id
        )

class Product(Asset):
    asset_type: Literal['products'] = Field(default='products', description="Type of the asset")
    product_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Name of the product")
    description: Optional[str] = Field(None, description="Description of the product")    
    price: float = Field(..., gt=0, description="Price of the product")
    owner_id: str = Field(..., description="ID of the owner")
    brand: Optional[str] = Field(None, description="Brand of the product")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Raw attribute payload for the product")
    is_food: Optional[bool] = Field(default=None, description="Whether the product is classified as food")
    nutrition_supply: Optional[Dict[str, float]] = Field(default=None, description="Nutrition supply data when the product is food")
    satisfaction_attributes: Optional[Dict[str, Any]] = Field(default=None, description="Satisfaction attributes when the product is non-food")
    duration_months: Optional[int] = Field(default=None, description="Duration (in months) the product provides satisfaction/nutrition")
    
    @classmethod
    def create(cls, name: str, price: float, owner_id: str, amount: float = 1.0,
               classification: Optional[str] = None, expiration_date: Optional[date] = None,
               manufacturer: Optional[str] = None, description: Optional[str] = None, brand: Optional[str] = None,
               product_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None,
               is_food: Optional[bool] = None, nutrition_supply: Optional[Dict[str, float]] = None,
               satisfaction_attributes: Optional[Dict[str, Any]] = None, duration_months: Optional[int] = None) -> 'Product':
        if price <= 0:
            raise ValueError("Price must be greater than zero")
        return cls(
            name=name,
            asset_type='products',
            product_id=product_id,
            price=price,
            owner_id=owner_id,
            amount=amount,
            classification=classification,
            expiration_date=expiration_date,
            manufacturer=manufacturer,
            description=description,
            brand=brand,
            attributes=attributes,
            is_food=is_food,
            nutrition_supply=nutrition_supply,
            satisfaction_attributes=satisfaction_attributes,
            duration_months=duration_months
        )
        

class Transaction(BaseModel):
    id: str
    sender_id: str
    receiver_id: str
    amount: float
    assets: Optional[List[Any]] = Field(default_factory=list)
    labor_hours: List[LaborHour] = Field(default_factory=list)
    type: Literal['purchase', 'interest', 'service', 'redistribution', 'consume_tax', 'labor_tax', 'corporate_tax', 'labor_payment', 'inherent_market'] = Field(default='purchase', description="Type of transaction")
    month: Optional[int] = Field(default=0, description="Month number")

class PurchaseRecord(BaseModel):
    product_id: str
    product_name: str
    quantity: float
    price_per_unit: float
    total_spent: float
    seller_id: str
    tx_id: str
    timestamp: datetime
    month: Optional[int] = Field(default=0, description="Month number")

class MiddlewareRegistry:
    def __init__(self):
        self.middlewares_by_type: Dict[str, List[Callable[[Transaction, Dict[str, float]], None]]] = {}

    def register(self, tx_type: str, func: Callable, tag: Optional[str] = None):
        if tag:
            self.middlewares_by_type[tx_type] = [f for f in self.middlewares_by_type.get(tx_type, []) if getattr(f, "_tag", None) != tag]
            func._tag = tag
        self.middlewares_by_type.setdefault(tx_type, []).append(func)

    def execute_all(self, tx_type: str, transaction: Transaction, ledger: Dict[str, Ledger]):
        for mw in self.middlewares_by_type.get(tx_type, []):
            mw(transaction, ledger)

class FirmInnovationConfig(BaseModel):
    company_id: str
    innovation_strategy: Literal['encouraged', 'suppressed']
    labor_productivity_factor: float
    profit_margin: Optional[float] = None
    fund_share: float

class FirmInnovationEvent(BaseModel):
    company_id: str
    innovation_type: Optional[Literal['price', 'attribute', 'profit_margin', 'labor_productivity_factor']] = None
    month: int
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    price_change: Optional[float] = None
    attribute_change: Optional[float] = None

    @classmethod
    def create(cls, company_id: str, innovation_type: Optional[Literal['price', 'attribute', 'profit_margin', 'labor_productivity_factor']] = None, month: int = 0, old_value: Optional[float] = None, new_value: Optional[float] = None, price_change: Optional[float] = None, attribute_change: Optional[float] = None) -> 'FirmInnovationEvent':
        return cls(
            company_id=company_id,
            innovation_type=innovation_type,
            month=month,
            old_value=old_value,
            new_value=new_value,
            price_change=price_change,
            attribute_change=attribute_change
        )