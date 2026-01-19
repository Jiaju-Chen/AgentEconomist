from abc import ABC, abstractmethod
from typing import List

class HouseholdConsumptionFunction(ABC):
    """
    抽象基类，定义消费函数接口。
    方法calculate_consumption应返回[数量]列表，而不是预算列表。
    """

    @abstractmethod
    def calculate_consumption(self, consumption_budget: float, goods_unit_prices: List[float]) -> List[float]:
        """
        计算在给定的消费预算(consumption_budget)和商品单价列表(goods_unit_prices)条件下，
        每种商品应购买的数量。返回数量列表。
        :param consumption_budget: 最大消费预算，若为负数则视为0
        :param goods_unit_prices: 各种商品的单位价格列表
        :return: 各种商品的购买数量列表
        """
        pass  # 由子类实现