import json
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Dict, List, Any, Optional, Union

# Relative imports for internal modules
from ..center.model import Asset, TaxPolicy
from ..center.ecocenter import EconomicCenter
from ..center.middleware.middleware import *

# LLM and logging
from ..llm.llm import LLM, LLMConfig, LLMProviderType, ChatCompletionMessageParam
from ..logger import get_logger

# Third-party libraries
import ray
import copy
import asyncio

# Initialize Ray if not already initialized
# try:
#     ray.init(ignore_reinit_error=True)
# except Exception:
#     pass

logger = get_logger(__name__)

@ray.remote
class Government:
    """
    # Government Agent
    Distributed government entity managing fiscal policy and taxation using Ray framework.
    
    ## Features
    - Tax policy management
    - Middleware integration for tax collection
    - LLM-assisted policy updates
    - Budget tracking via EconomicCenter
    """
    
    def __init__(self,
                 government_id: str,
                 initial_budget: float = 0.0,
                 tax_policy: TaxPolicy = None,
                 llm: Optional[LLM] = None,
                 economic_center: Optional[EconomicCenter] = None):
        """
        ## Initialize Government Agent
        Creates a new government agent with full tax management capabilities.
        
        ### Parameters
        - `government_id` (str): Unique identifier for the government
        - `initial_budget` (float): Starting budget allocation
        - `tax_policy` (TaxPolicy): Initial tax policy configuration
        - `llm` (LLM): Language model for policy recommendations
        - `economic_center` (EconomicCenter): Economic state manager
        
        ### Raises
        - ValueError: If government_id is empty or invalid
        """
        # Validate government ID
        if not government_id or not isinstance(government_id, str):
            raise ValueError("government_id must be a non-empty string")
        
        # Use default policy if none provided
        if tax_policy is None:
            tax_policy = TaxPolicy()
        
        # Store core state directly
        self.government_id = government_id
        self.tax_policy = tax_policy.model_copy()
        self.initial_budget = initial_budget
        # Store dependencies
        self.llm = llm
        self.economic_center = economic_center
        
        is_initialized = False
        # Initialize budget in economic center if provided
        # if economic_center and initial_budget:
        #     try:
        #         ray.get(self.economic_center.init_agent_ledger.remote(government_id, initial_budget))
        #         logger.info(f"Government {government_id} registered ledger with ${initial_budget:.2f}")
        #     except Exception as e:
        #         logger.warning(f"[Government Init] Failed to register ledger for {government_id}: {e}")

        # if economic_center and initial_budget > 0:
        #     # 直接使用ledger代替deposit_funds
        #     self.economic_center.ledger[government_id].amount += initial_budget # TODO(xiaxu):经济中心需要初始化函数
                
        # Cache management
        self._last_budget: Optional[float] = None
        self.logger = get_logger(__name__)
    async def initialize(self):
        """
        ## Initialize Government Agent
        Asynchronously initializes the government agent, including middleware registration.
        
        ### Implementation
        Calls internal _register_middleware method to set up tax collection processes
        """
        if self.economic_center:
            try:
                await asyncio.gather(self.economic_center.init_agent_ledger.remote(self.government_id, self.initial_budget),
                                     self.economic_center.register_id.remote(self.government_id, 'government'),
                                     self.economic_center.init_agent_product.remote(self.government_id)
                )
                self.logger.info(f"Government {self.government_id} registered in EconomicCenter")
            except Exception as e:
                self.logger.warning(f"[Government Init] Failed to register ledger for {self.government_id}: {e}")
        await self._register_middleware()

        self.initialized = True

    async def _register_middleware(self):
        """
        ## Register Tax Middleware
        Asynchronously registers tax collection middleware with the economic center.
        
        ### Registered Middleware
        1. **Consume Tax** - Tracks consumption-based taxation
        2. **Labor Tax** - Manages wage-based taxation
        3. **VAT Tax** - Handles value-added tax collection
        
        ### Implementation
        Uses centralized register_middleware interface for cleaner integration
        """
        if self.economic_center:
            try:
                results = await asyncio.gather(
                    # 消费税：家庭支付时收取8%消费税
                    self.economic_center.register_middleware.remote(
                        tx_type='consume_tax',
                        middleware_fn=consume_tax_middleware(self.tax_policy.vat_rate, self.government_id),
                        tag='consume_tax'
                    ),
                    # 个人所得税：工资发放时收取22.5%个人所得税
                    self.economic_center.register_middleware.remote(
                        tx_type='labor_tax',
                        middleware_fn=labor_tax_middleware(self.tax_policy.income_tax_rate, self.government_id),
                        tag='labor_tax'
                    ),
                    # 企业所得税：企业收入时收取21%企业所得税
                    self.economic_center.register_middleware.remote(
                        tx_type='corporate_tax',
                        middleware_fn=VAT_tax_middleware(self.tax_policy.corporate_tax_rate, self.government_id),
                        tag='corporate_tax'
                    ),
                    return_exceptions=True
                )
                # Log results
                self.logger.info(f"[middleware] Registered {len(results)} middleware functions for government {self.government_id}")
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"[middleware] Registration {i} failed: {result}")
            except Exception as e:
                self.logger.error(f"[middleware] Unexpected registration failure: {e}")

    async def update_tax_policy(self, new_policy: TaxPolicy) -> None: 
        """
        ## Update Tax Policy
        Applies new tax policy while ensuring validation and consistency.
        
        ### Parameters
        - `new_policy` (TaxPolicy): Validated tax policy to apply
        
        ### Validation
        - Ensures non-null input
        - Performs deep copy to prevent external state modification
        
        ### Raises
        - ValueError: If new_policy is None
        """
        # Validate new policy
        if not new_policy:
            raise ValueError("new_policy cannot be None")
        
        # Update internal state
        self.tax_policy = new_policy.model_copy()

    async def step(self): 
        """
        ## Government Step
        Executes a single step for the government agent, including policy updates and middleware re-registration.
        
        ### Implementation
        - Calls update_tax_policy if LLM is available
        - Re-registers middleware to apply any policy changes
        """
        if self.llm:
            # Use LLM to get new tax policy
            new_policy = await self.llm.get_tax_policy() #TODO: 需要补充税率更新函数
            await self.update_tax_policy(new_policy)
            await self._register_middleware()

        
        # Re-register middleware with updated tax policy
        
        self.logger.info(f"Government {self.government_id} step completed with policy: {self.tax_policy.model_dump_json()}")
