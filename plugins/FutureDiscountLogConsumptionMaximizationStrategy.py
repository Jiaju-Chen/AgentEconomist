import math
import random



"""
The objective of the algorithm is to maximize the sum of 
expected discounted log consumption over all time. Deposit 
account interest rates, effective goods price (per unit) and
nominal income are external to the optimization, and must be 
specified by the caller each cycle.
"""
class FutureDiscountLogConsumptionMaximizationStrategy:
  
    def __init__(self, beta, gamma, fund_investment_min, deposits_min):
        """
        :param beta: Future discount
        :param gamma: Adaptation rate for expectations
        :param fund_investment_min: Minimum fund investment
        :param deposits_min: Minimum deposits
        """
        # 参数
        self.bet = beta
        self.gam = gamma
        self.mmin = fund_investment_min
        self.dmin = deposits_min
        # 状态变量
        self.last_solution = [0.0] * 6  # 初始化为零
        self.Rf = 0.0
        self.RfLast = 0.0
        self.exL = 0.0
        self.exLQ = 0.0
        self.q = 0.0
        self.p = 0.0
        self.y = 0.0

    def predictNextL(self):
        return self.gam * self.last_solution[3] + (1 - self.gam) * self.exL

    def predictNextLQ(self):
        return self.gam * self.last_solution[3] * self.q + (1 - self.gam) * self.exLQ

    def processBoundaryCaseTwo(self):
        """
        Constrained by fund investments
        """
        denominator_c = (self.bet * self.exL * self.p * self.Rf) - (self.bet * self.exL * self.gam * self.p * self.Rf)
        if denominator_c == 0:
            return None  
        c = (1 - self.bet * self.gam * self.Rf) / denominator_c
        m = self.mmin
  
        numerator_d = 1 - self.bet * self.Rf * (self.gam + self.exL * (-1 + self.gam) * (self.mmin - self.last_solution[1]) * self.q -
                                               self.exL * (-1 + self.gam) * self.last_solution[2] * self.RfLast +
                                               self.exL * self.y - self.exL * self.gam * self.y)
        denominator_d = self.bet * self.exL * (-1 + self.gam) * self.Rf
        if denominator_d == 0:
            return None
        d = numerator_d / denominator_d
      
        lam = (self.bet * self.exL * (-1 + self.gam) * self.Rf) / (-1 + self.bet * self.gam * self.Rf)
        mu1 = 0.0
        mu2 = (self.bet * (-1 + self.gam) * (-self.exLQ + (self.bet * self.exLQ * self.gam +
                                                          (self.exL - self.bet * self.exL * self.gam) * self.q) * self.Rf)) / (-1 + self.bet * self.gam * self.Rf)
        if self.is_valid_state(c, m, d, lam, mu1, mu2):
            return [c, m, d, lam, mu1, mu2]
        else:
            return None

    def processBoundaryCaseThree(self):
        """
        Constrained by deposits
        """
     
        denominator_c = (self.bet * self.exLQ * self.p) - (self.bet * self.exLQ * self.gam * self.p)
        if denominator_c == 0:
            return None
        c = (self.q - self.bet * self.gam * self.q) / denominator_c
       
        numerator_m = ((1 - self.bet * self.gam + self.bet * self.exLQ * (-1 + self.gam) * self.last_solution[1]) * self.q -
                       self.bet * self.exLQ * (-1 + self.gam) * (self.dmin - self.last_solution[2] * self.RfLast - self.y))
        denominator_m = self.bet * self.exLQ * (-1 + self.gam) * self.q
        if denominator_m == 0:
            return None
        m = numerator_m / denominator_m
        d = self.dmin
     
        lam = (self.bet * self.exLQ * (-1 + self.gam)) / ((-1 + self.bet * self.gam) * self.q)
        mu1 = (self.bet * (-1 + self.gam) * (self.exLQ - (self.bet * self.exLQ * self.gam +
                                                          (self.exL - self.bet * self.exL * self.gam) * self.q) * self.Rf)) / ((-1 + self.bet * self.gam) * self.q)
        mu2 = 0.0
        if self.is_valid_state(c, m, d, lam, mu1, mu2):
            return [c, m, d, lam, mu1, mu2]
        else:
            return None

    def processBoundaryCaseFour(self):
        """
        processBoundaryCaseFour
        """
        c = (-self.dmin + (-self.mmin + self.last_solution[1]) * self.q + self.last_solution[2] * self.RfLast + self.y) / self.p
        m = self.mmin
        d = self.dmin
    
        denominator_lam = (-self.dmin + (-self.mmin + self.last_solution[1]) * self.q + self.last_solution[2] * self.RfLast + self.y)
        if denominator_lam == 0:
            return None
        lam = 1.0 / denominator_lam
        numerator_mu1 = (-1 + self.bet * self.Rf * (-(self.dmin * self.exL) + self.gam + self.dmin * self.exL * self.gam +
                                                     self.exL * (-1 + self.gam) * (self.mmin - self.last_solution[1]) * self.q -
                                                     self.exL * (-1 + self.gam) * self.last_solution[2] * self.RfLast +
                                                     self.exL * self.y - self.exL * self.gam * self.y))
        denominator_mu1 = (self.dmin + (self.mmin - self.last_solution[1]) * self.q - self.last_solution[2] * self.RfLast - self.y)
        if denominator_mu1 == 0:
            mu1 = float('inf')  
        else:
            mu1 = numerator_mu1 / denominator_mu1
        numerator_mu2 = ((-1 + self.bet * (self.gam - self.exLQ * self.mmin + self.exLQ * self.gam * self.mmin) -
                          self.bet * self.exLQ * (-1 + self.gam) * self.last_solution[1]) * self.q +
                         self.bet * self.exLQ * (-1 + self.gam) * (self.dmin - self.last_solution[2] * self.RfLast - self.y))
        denominator_mu2 = denominator_mu1
        if denominator_mu2 == 0:
            mu2 = float('inf')
        else:
            mu2 = numerator_mu2 / denominator_mu2
        if self.is_valid_state(c, m, d, lam, mu1, mu2):
            return [c, m, d, lam, mu1, mu2]
        else:
            return None

    def isValidState(self, c, m, d, lam, mu1, mu2):

        result = (mu1 >= 0.0 and d >= self.dmin and
                  mu2 >= 0.0 and m >= self.mmin and
                  c >= 0.0)
        residue = (self.p * c + self.q * m + d - self.y -
                   self.q * self.last_solution[1] - self.RfLast * self.last_solution[2])
        print(f"{'*' if result else ' '} c: {c:.10g} m: {m:.10g} d: {d:.10g} lam: {lam:.10g} mu1: {mu1:.10g} mu2: {mu2:.10g}")
        if abs(residue) > 1e-8:
            return False
        return result

    def compute_next(self, Rf, q, p, y):
        """
        :param Rf: Interest rate (absolute) on deposits
        :param q: MutualFund price
        :param p: Effective goods price per unit
        :param y: Nominal income received
        :return: A 3-dimensional vector with the following format:
               { consumption budget, desired fund position, desired deposits }
        """
        # 更新状态变量
        self.RfLast = self.Rf
        self.Rf = Rf
        self.q = q
        self.p = p
        self.y = y
        
        # 按顺序尝试边界情况
        result = self.processBoundaryCaseTwo()
        if result is None:
            result = self.processBoundaryCaseThree()
        if result is None:
            result = self.processBoundaryCaseFour()
        
        # 如果找到有效状态，更新 last_solution 和期望
        if result is not None:
            self.last_solution = result.copy()
            self.exL = self.predictNextL()
            self.exLQ = self.predictNextLQ()
            print("- step complete")
            return result[:3]  # 仅返回前三个元素
        else:
            raise ValueError("未找到有效状态")

if __name__ == "__main__":
    algorithm = FutureDiscountLogConsumptionMaximizationStrategy(0.98, 0.7, 0.0, 0.0)
    dice = random.Random()
    print("New Session")
    for i in range(500):
        Rf = 1.0 + (dice.gauss(0, 5e-2) ** 2)
        q = 1.0 + dice.gauss(0, 5e-2)
        p = dice.gauss(0, 1) ** 2
        y = dice.random() + 0.5
        position = algorithm.compute_next(Rf, q, p, y)
        print(f"  d: {position[1]:.10g} m: {position[2]:.10g}")
    print("Session Ends")