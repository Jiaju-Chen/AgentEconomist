from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional    #一个值可以是类型T或者是None


class EmpiricalDistribution:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.values = []

    def add(self, value: float):
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values.pop(0)

    def get_last_added(self) -> float:
        return self.values[-1] if self.values else 0.0

    def size(self) -> int:
        return len(self.values)

    def maxSize(self) -> int:
        return self.max_size

 #简化类的定义
@dataclass(frozen=True)
class HouseholdState:
    last_labour_employed: float
    maximum_labour_offered: float
    last_labour_unit_wage: float
    most_recent_consumption_budget: float
    previous_total_consumption: float
    total_deposits: float
    fund_account_value: float
    previous_fund_account_value: float
    household_equity: float
    previous_dividend_received: float
    total_assets: float
    total_liabilities: float
    unreserved_deposits: float
    previous_mean_wage: float

    class Builder:
        def __init__(self):
            self.last_labour_employed = float('-inf')
            self.maximum_labour_offered = float('-inf')
            self.last_labour_unit_wage = float('-inf')
            self.most_recent_consumption_budget = float('-inf')
            self.previous_total_consumption = float('-inf')
            self.total_deposits = float('-inf')
            self.fund_account_value = float('-inf')
            self.previous_fund_account_value = float('-inf')
            self.household_equity = float('-inf')
            self.previous_dividend_received = float('-inf')
            self.total_assets = float('-inf')
            self.total_liabilities = float('-inf')
            self.unreserved_deposits = float('-inf')
            self.previous_mean_wage = float('-inf')

        def last_labour_employed(self, value: float):
            self.last_labour_employed = value
            return self

        def maximum_labour_offered(self, value: float):
            self.maximum_labour_offered = value
            return self

        def last_labour_unit_wage(self, value: float):
            self.last_labour_unit_wage = value
            return self

        def most_recent_consumption_budget(self, value: float):
            self.most_recent_consumption_budget = value
            return self

        def previous_total_consumption(self, value: float):
            self.previous_total_consumption = value
            return self

        def total_deposits(self, value: float):
            self.total_deposits = value
            return self

        def fund_account_value(self, value: float):
            self.fund_account_value = value
            return self

        def previous_fund_account_value(self, value: float):
            self.previous_fund_account_value = value
            return self

        def household_equity(self, value: float):
            self.household_equity = value
            return self

        def previous_dividend_received(self, value: float):
            self.previous_dividend_received = value
            return self

        def total_assets(self, value: float):
            self.total_assets = value
            return self

        def total_liabilities(self, value: float):
            self.total_liabilities = value
            return self

        def unreserved_deposits(self, value: float):
            self.unreserved_deposits = value
            return self

        def previous_mean_wage(self, value: float):
            self.previous_mean_wage = value
            return self

        def build(self):
            return HouseholdState(
                last_labour_employed=self.last_labour_employed,
                maximum_labour_offered=self.maximum_labour_offered,
                last_labour_unit_wage=self.last_labour_unit_wage,
                most_recent_consumption_budget=self.most_recent_consumption_budget,
                previous_total_consumption=self.previous_total_consumption,
                total_deposits=self.total_deposits,
                fund_account_value=self.fund_account_value,
                previous_fund_account_value=self.previous_fund_account_value,
                household_equity=self.household_equity,
                previous_dividend_received=self.previous_dividend_received,
                total_assets=self.total_assets,
                total_liabilities=self.total_liabilities,
                unreserved_deposits=self.unreserved_deposits,
                previous_mean_wage=self.previous_mean_wage
            )

    def get_last_labour_employed(self) -> float:
        return self.last_labour_employed

    def get_maximum_labour_offered(self) -> float:
        return self.maximum_labour_offered

    def get_last_labour_unit_wage(self) -> float:
        return self.last_labour_unit_wage

    def get_most_recent_consumption_budget(self) -> float:
        return self.most_recent_consumption_budget

    def get_previous_total_consumption(self) -> float:
        return self.previous_total_consumption

    def get_total_deposits(self) -> float:
        return self.total_deposits

    def get_fund_account_value(self) -> float:
        return self.fund_account_value

    def get_previous_fund_account_value(self) -> float:
        return self.previous_fund_account_value

    def get_previous_household_equity(self) -> float:
        return self.household_equity

    def get_previous_dividend_received(self) -> float:
        return self.previous_dividend_received

    def get_total_assets(self) -> float:
        return self.total_assets

    def get_total_liabilities(self) -> float:
        return self.total_liabilities

    def get_unreserved_deposits(self) -> float:
        return self.unreserved_deposits

    def get_previous_mean_wage(self) -> float:
        return self.previous_mean_wage

#创建抽象基类(不能被实例化，必须被继承)
class HouseholdDecisionRule(ABC):
    MEMORY_LENGTH = 100

    def __init__(self, initial_value: float = None):
        if initial_value is None:
            initial_value = 0.0
        self.past_values = EmpiricalDistribution(self.MEMORY_LENGTH)
        self.past_values.add(initial_value)

    #声明抽象方法(子类必须实现的方法)
    @abstractmethod
    def compute_next(self, current_state: HouseholdState) -> float:
        pass

    #Commit a new value to memory
    def record_new_value(self, value: float):
        self.past_values.add(value)

    #Get the newest stored value
    def get_last_value(self) -> float:
        return self.past_values.get_last_added()

    def memory_used(self) -> int:
        return self.past_values.size()

    def __str__(self):
        return (f"state machine with {self.past_values.maxSize()} "
                f"units of memory ({self.memory_used()} used)")

if HouseholdDecisionRule.MEMORY_LENGTH < 0:
    raise Exception("HouseholdDecisionRule.MEMORY_LENGTH < 0")