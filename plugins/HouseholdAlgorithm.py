from HouseholdDecisionRule import HouseholdDecisionRule, HouseholdState
from HouseholdConsumptionFunction import HouseholdConsumptionFunction
from typing import List
from abc import ABC, abstractmethod
from collections import deque
import random
from typing import Optional 
import math

"""
A Consumption Budget Decision Rule Based on AR1 (First-Order Autoregressive) Process
At each time step t, the operation mechanism of this class is as follows:
1.Invoke the delegated HouseholdDecisionRule object (implementation) to calculate the current consumption budget B(t).
2.Compute the smoothed budget using the formula B(t) * m + B(t - 1) * (1 - m), where m is a fixed, customizable memory parameter (adaptation rate).
3.Return the smoothed budget value, which shall not exceed the household's unreserved deposits.
"""
class AR1ConsumptionBudgetAlgorithm(HouseholdDecisionRule):
    def __init__(self, implementation: HouseholdDecisionRule, adaptation_rate: float):
        super().__init__(0.0)
        self.implementation = implementation
        self.adaptation_rate = adaptation_rate

    def compute_next(self, state: HouseholdState) -> float:
        self.implementation.compute_next(state)
        maximum_budget = state.get_unreserved_deposits()
        intended_consumption = self.implementation.get_last_value()
        smoothed_consumption = (
            self.adaptation_rate * intended_consumption +
            (1. - self.adaptation_rate) * super().get_last_value()
        )
        super().record_new_value(smoothed_consumption)
        return min(smoothed_consumption, maximum_budget)

    def get_adaptation_rate(self) -> float:
        return self.adaptation_rate

    def __str__(self):
        return f"{self.__class__.__name__} with adaptation rate: {self.get_adaptation_rate()}"
    

"""
Modeling Household Consumption Allocation Using CES (Constant Elasticity of Substitution) Formulation
The CES function is employed to simulate households' consumption allocation decisions among different goods. 
The key parameter in the CES framework is Rho (ρ), which determines the elasticity of substitution between commodities.
"""
class CESConsumptionAggregatorAlgorithm:
    def __init__(self, rho: float):
        if rho >= 1.0:
            raise ValueError("CESConsumptionAggregatorAlgorithm: rho must be less than 1.0")
        self.rho = rho

    def calculate_consumption(self, consumption_budget: float, goods_unit_prices: List[float]) -> List[float]:
        if not goods_unit_prices:
            return []
        P = 0.0
        rho_ex = self.rho / (self.rho - 1.0)
        rho_div = 1.0 / (1.0 - self.rho)
        for price in goods_unit_prices:
            P += price ** rho_ex
        P = P ** (1.0 / rho_ex)
        demand_list_result = []
        for price in goods_unit_prices:
            if price == 0.0:
                demand_list_result.append(float('inf'))
            else:
                demand_list_result.append(
                    consumption_budget / P * (P / price) ** rho_div
                )
        return demand_list_result

    def get_rho(self) -> float:
        return self.rho

    def __str__(self):
        return f"CES Consumption Aggregator algorithm, rho = {self.rho}"


"""
A household decision rule that calculates the consumption budget based on a 
consumption propensity multiplier and a wealth exponent.
The consumption budget is calculated as:
Budget = consumption_propensity * (wealth ^ wealth_exponent)
where wealth is the unreserved deposits of the household. The budget is capped 
at the available wealth.
"""
class ConsumptionPropensityBudgetAlgorithm(HouseholdDecisionRule):

    def __init__(self, consumption_propensity: float = 0.0, wealth_exponent: float = 1.0):
        
        # Initialize the parent class with a default value of 0.0
        super().__init__(0.0)

        # Validate wealth_exponent
        if wealth_exponent < 0.0:
            raise ValueError(
                f"ConsumptionPropensityBudgetAlgorithm: wealth exponent (value {wealth_exponent}) "
                "has an illegal value. Expected 0 <= wealthExponent"
            )

        # Validate consumption_propensity
        if not (0.0 <= consumption_propensity <= 1.0):
            raise ValueError(
                "ConsumptionPropensityBudgetAlgorithm: consumption propensity must be in the range [0, 1] inclusive."
            )

        self.consumption_propensity = consumption_propensity
        self.wealth_exponent = wealth_exponent

    def compute_next(self, state: HouseholdState) -> float:
        """
        Compute the consumption budget based on the household's current state.

        The budget is calculated as consumption_propensity * (wealth ^ wealth_exponent),
        capped at the available wealth.

        Args:
            state (HouseholdState): The current state of the household.

        Returns:
            float: The computed consumption budget.
        """
        # Get the household's unreserved deposits (wealth)
        wealth = state.get_unreserved_deposits()
        # Ensure wealth is non-negative and compute the budget
        result = self.consumption_propensity * (max(wealth, 0.0) ** self.wealth_exponent)
        # Cap the budget at the available wealth
        result = min(result, wealth)
        # Record the computed value in the parent class
        super().record_new_value(result)
        return result

    def get_consumption_propensity(self) -> float:
        """
        Get the current consumption propensity multiplier.

        Returns:
            float: The consumption propensity value.
        """
        return self.consumption_propensity

    def get_wealth_exponent(self) -> float:
        """
        Get the current wealth exponent.

        Returns:
            float: The wealth exponent value.
        """
        return self.wealth_exponent

    def set_consumption_propensity(self, consumption_propensity: float):
        
        if not (0.0 <= consumption_propensity <= 1.0):
            raise ValueError(
                "ConsumptionPropensityBudgetAlgorithm: consumption propensity must be in the range [0, 1] inclusive."
            )
        self.consumption_propensity = consumption_propensity

    def __str__(self):
       
        return (
            f"Consumption Propensity Budget Algorithm: "
            f"propensity: {self.get_consumption_propensity()}, "
            f"wealth exponent: {self.get_wealth_exponent()}."
        )


"""
A household decision rule that calculates the consumption budget based on 
expected wages assuming full employment, adjusted by the actual employment rate.
The consumption budget is calculated as:
    Budget = expected_wage_full_employment * employment_rate
where employment_rate is the ratio of last total employment to last total labor supply.
"""
class ExpectedWageConsumptionBudgetAlgorithm(HouseholdDecisionRule):
    
    def __init__(self, expected_income_full_employment: float, labour_market):
        """
        Initialize the ExpectedWageConsumptionBudgetAlgorithm with expected income 
        assuming full employment and the labor market.

        Args:
            expected_income_full_employment (float): The expected income per cycle 
                assuming full employment. Must be >= 0.
            labour_market: The labor market on which the household offers its labor.

        Raises:
            ValueError: If expected_income_full_employment < 0.
            TypeError: If labour_market is None.
        """
        # Validate expected_income_full_employment to ensure it's non-negative
        if expected_income_full_employment < 0.0:
            raise ValueError("expected_income_full_employment must be >= 0.")

        # Validate labour_market to ensure it's not None
        if labour_market is None:
            raise TypeError("labour_market must not be None.")

        self.expected_wage_assuming_full_employment = expected_income_full_employment
        self.labour_market = labour_market

    def compute_next(self, state: HouseholdState) -> float:
        """
        Compute the consumption budget based on the household's current state 
        and labor market conditions.

        The budget is calculated as expected_wage_full_employment multiplied by 
        the employment rate (last total employment / last total labor supply).

        Args:
            state (HouseholdState): The current state of the household.

        Returns:
            float: The computed consumption budget.
        """
        # Retrieve last total labor supply and employment from the labor market
        last_total_labour_supply = self.labour_market.get_last_labour_total_supply()
        last_total_employment = self.labour_market.get_last_total_employed_labour()

        # Calculate employment rate (e), handling division by zero
        if last_total_labour_supply == 0.0:
            e = 0.0
        else:
            e = last_total_employment / last_total_labour_supply

        # Compute budget as expected wage adjusted by employment rate
        result = self.expected_wage_assuming_full_employment * e

        # Record the computed value in the parent class
        super().record_new_value(result)
        return result

    def get_expected_wage_assuming_full_employment(self) -> float:
        return self.expected_wage_assuming_full_employment

    def __str__(self):
        return (
            f"Expected Wage Consumption Budget Algorithm, "
            f"expected wage: {self.expected_wage_assuming_full_employment}"
        )


"""
A household decision rule that always expects a fixed wage per unit of labor.
This algorithm returns a constant wage expectation regardless of the household's state.
"""
class FixedWageExpectationAlgorithm(HouseholdDecisionRule):
    
    def __init__(self, expected_labour_wage: float):
        """
        Initialize the FixedWageExpectationAlgorithm with a fixed expected wage.

        Args:
            expected_labour_wage (float): The fixed wage per unit of labor. Must be >= 0.

        Raises:
            ValueError: If expected_labour_wage < 0.
        """
        # Initialize the parent class with the expected_labour_wage
        super().__init__(expected_labour_wage)

        # Validate that expected_labour_wage is non-negative, mirroring Java's Preconditions.checkArgument
        if expected_labour_wage < 0.0:
            raise ValueError("FixedWageExpectationAlgorithm: labour wage must be non-negative.")

        # Store the fixed wage as an instance variable
        self.expected_labour_wage = expected_labour_wage

    def compute_next(self, state: HouseholdState) -> float:
        """
        Compute the expected wage, which is always the fixed value.

        Args:
            state (HouseholdState): The current state of the household (unused in this algorithm).

        Returns:
            float: The fixed expected wage.
        """
        # Simply return the fixed wage, no computation needed based on state
        return self.expected_labour_wage

    def get_fixed_labour_wage(self) -> float:
        """
        Get the fixed labour wage (wage per unit labour).

        Returns:
            float: The fixed wage value.
        """
        # Return the stored fixed wage value
        return self.expected_labour_wage

    def __str__(self):
        """
        Provide a string representation of the algorithm.

        Returns:
            str: A description including the fixed wage value.
        """
        # Return a string that matches the Java toString() format
        return f"Fixed Wage Ask Price Household Decision Rule, fixed wage = {self.get_fixed_labour_wage()}"
    


"""
Compute the desired consumption (cash value) for a set of goods given their
unit prices and the maximum size of the consumption budget. The same amount
of cash is allocated to each type of goods.
"""
class HomogeneousConsumptionAlgorithm:
   

    def __init__(self):
        # No initialization needed; this class is stateless.
        pass

    def calculate_consumption(self, consumption_budget: float, goods_unit_prices: List[float]) -> List[float]:
        """
        Compute the desired consumption quantities for a set of goods given their
        unit prices and the maximum size of the consumption budget. The same amount
        of cash is allocated to each type of goods.

        Args:
            consumption_budget (float): The maximum cash value of the consumption budget.
                If negative, it is silently trimmed to zero.
            goods_unit_prices (List[float]): A list of goods unit prices (cost per unit good to buy).

        Returns:
            List[float]: A list of consumption quantities, one for each good.
        """
        # Handle edge case: if there are no goods, return an empty list
        if len(goods_unit_prices) == 0:
            return []

        # Trim negative budget to zero as per the Java comment
        effective_budget = max(consumption_budget, 0.0)

        # Calculate the number of goods
        num_goods = len(goods_unit_prices)

        # Allocate equal cash to each good
        homogeneous_cash = effective_budget / num_goods

        # Compute consumption quantities for each good
        result = []
        for price in goods_unit_prices:
            if price == 0:
                # Handle division by zero: set consumption to infinity
                result.append(float('inf'))
            else:
                # Consumption quantity = (allocated cash) / price
                result.append(homogeneous_cash / price)

        return result

    def __str__(self):
        # Return a string representation of the object
        return "Homogeneous Consumption Function."
    


"""
Compute the desired consumption (cash value) for a set of goods given their
unit prices and the maximum size of the consumption budget. The same amount
of cash is allocated to each type of goods
"""
class HomogeneousConsumptionAlgorithm(HouseholdConsumptionFunction):
    
    def calculate_consumption(self, consumption_budget: float, goods_unit_prices: List[float]) -> List[float]:
        # 若预算为负，则默认为0
        budget = max(consumption_budget, 0.0)
        # 若无商品，返回空列表
        if not goods_unit_prices:
            return []
        # 将预算均匀分配到每种商品上（现金值）
        homogeneous_consumption = budget / len(goods_unit_prices)
        # 根据各商品单价计算应购买的数量
        result: List[float] = []
        for price in goods_unit_prices:
            # 若单价为0，则无法购买，设为0以避免除零错误
            if price <= 0:
                result.append(0.0)
            else:
                result.append(homogeneous_consumption / price)
        return result

    def __str__(self) -> str:
        # 返回对象描述
        return "Homogeneous Consumption Function."


"""
An exponential function-based household consumption allocation algorithm that considers the impact of 
commodity prices and production levels
"""
class ParisHouseholdConsumptionAlgorithm(HouseholdConsumptionFunction):
    def __init__(self, beta: float):
        # Beta is an immutable parameter
        self.beta = beta

    def calculate_consumption(self, consumption_budget: float, goods_unit_prices: List[float]) -> List[float]:
        result = [0.0] * len(goods_unit_prices)
        
        # Compute P̄ (weighted average price)
        p_bar = self.get_p_bar(goods_unit_prices)
        z = 0.0
        
        # First pass: compute unnormalized consumption
        exp_factors = []
        for price in goods_unit_prices:
            expfac = math.exp(-self.beta * price / p_bar)
            exp_factors.append(expfac)
            z += expfac
        
        # Calculate consumption quantities
        for i, (price, expfac) in enumerate(zip(goods_unit_prices, exp_factors)):
            result[i] = consumption_budget * expfac / price

        # Normalize to ensure budget constraint
        for i in range(len(result)):
            result[i] /= z

        return result

    def get_p_bar(self, goods_unit_prices: List[float]) -> float:
        # Y would represent production levels in each sector
        y = self.get_production_all_sectors()
        if y is None or len(goods_unit_prices) != len(y):
            raise ValueError("ParisHouseholdConsumptionAlgorithm.get_p_bar: goods unit prices and sector production do not correspond.")
        
        # Calculate the weighted average price
        weighted_sum = sum(y_i * p_i for y_i, p_i in zip(y, goods_unit_prices))
        total_y = sum(y)

        return weighted_sum / total_y

    def get_production_all_sectors(self) -> Optional[List[float]]:
        # TODO: Link to actual production data
        return None

    def get_beta(self) -> float:
        return self.beta

    def __str__(self) -> str:
        return f"Paris household consumption algorithm, beta = {self.beta}"



"""
A trivial HouseholdDecisionRule that always returns 0.0.
Corresponds to the Java TrivialHouseholdDecisionRule class.
It is stateless and provides a constant decision value.
"""
class TrivialHouseholdDecisionRule(HouseholdDecisionRule):
    
    def __init__(self):
        """
        Create a TrivialHouseholdDecisionRule.
        This rule is stateless and requires no specific initialization parameters.
        """
        # Call the parent class constructor
        super().__init__()
        # The Java class has a default constructor and no instance variables,
        # so the Python __init__ is simple.

    def compute_next(self, state: HouseholdState) -> float:
        """
        Computes the next value for the household decision.
        This trivial implementation always returns 0.0, regardless of the state.

        Args:
            state: The current state of the household (not used by this rule).

        Returns:
            Always returns 0.0.
        """
        # Corresponds to super.recordNewValue(0.) in the Java code.
        # We call the parent's method to record the computed value.
        super().record_new_value(0.0)

        # Return the constant decision value.
        return 0.0

    def __str__(self) -> str:
        """
        Returns a brief description of this object.
        Corresponds to the Java toString() method.
        """
        return "Trivial Household Decision Rule."

    def __repr__(self) -> str:
        """
        Provides a representation of the object, useful for debugging.
        """
        return "TrivialHouseholdDecisionRule()"
    


# Define the abstract base class for the callback, similar to the Java interface
class EmploymentProportionCallback(ABC):
    @abstractmethod
    def getEmploymentProportion(self):
        """Get total employment / maximum labour supply."""
        pass

# Define the main class inheriting from HouseholdDecisionRule
class UnemploymentBasedWageAskAlgorithm(HouseholdDecisionRule):
    def __init__(self, initialLabourWage, wageAdaptationRate, minimumWage, employmentProportionCallback):
        # Check if initial labour wage is negative, similar to Java's initGuard
        if initialLabourWage < 0:
            raise ValueError("UnemploymentBasedWageAskAlgorithm: labour wage is negative.")
        # Initialize the parent class with the initial labour wage
        super().__init__(initialLabourWage)
        # Ensure the callback is not None, equivalent to Java's null check
        if employmentProportionCallback is None:
            raise ValueError("employmentProportionCallback cannot be None")
        # Set instance variables
        self.wagePerturbation = wageAdaptationRate * initialLabourWage
        self.minimumWage = minimumWage
        self.employmentPropCallback = employmentProportionCallback
        # Initialize a circular buffer with maximum length of 5, replacing Java's CircularFifoBuffer
        self.wageConfidence = deque(maxlen=5)
        self.wageConfidence.append(0.0)
        # Create a Random instance for generating random numbers, replacing Java's Random
        self.dice = random.Random()  # Note: You might want to seed this with a simulation-specific value

    def computeNext(self, state):
        """Compute the next wage proposition based on the household state."""
        # Calculate employment deficit
        employmentDeficit = state.getMaximumLabourOffered() - state.getLastLabourEmployed()
        lastWageProposition = super().getLastValue()
        # Generate a random factor between 0 and 1, equivalent to Java's nextDouble
        _factor = self.dice.random()
        # Determine if wage should be raised (full employment)
        doRaiseWageProposition = (employmentDeficit == 0.0)
        # Get employment proportion from the callback
        employmentProportion = self.employmentPropCallback.getEmploymentProportion()
        # Calculate balance factor based on employment proportion
        _aggregateBalanceFactor = (1. - employmentProportion) / employmentProportion
        # Compute confidence adjustment based on employment status
        nextWageConfidence = (_aggregateBalanceFactor if doRaiseWageProposition else -1.) * _factor
        # Add confidence value to the circular buffer
        self.wageConfidence.append(nextWageConfidence)
        # Get smoothed confidence value
        avConfidence = self.getSmoothedConfidence()
        # Calculate next wage proposition
        nextWageProposition = lastWageProposition + avConfidence * self.wagePerturbation
        # Ensure wage does not fall below minimum wage
        nextWageProposition = max(nextWageProposition, self.minimumWage)
        # Record the new value in the parent class
        super().recordNewValue(nextWageProposition)
        return nextWageProposition

    def getSmoothedConfidence(self):
        """Calculate the average confidence from the wageConfidence buffer."""
        return sum(self.wageConfidence) / len(self.wageConfidence)

    def getLastLabourWage(self):
        """Get the last labour wage from the parent class."""
        return super().getLastValue()

    def getWagePerturbation(self):
        """Get the characteristic wage perturbation."""
        return self.wagePerturbation

    def getMinimumWage(self):
        """Get the fixed minimum wage."""
        return self.minimumWage

    def __str__(self):
        """Return a string description of the object."""
        return (f"unemployment based labour wage ask algorithm, "
                f"adaptation rate = {self.getWagePerturbation()}, "
                f"minimum wage = {self.getMinimumWage()}")