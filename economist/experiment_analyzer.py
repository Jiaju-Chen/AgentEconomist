"""
实验数据分析器
通用的实验数据分析和对比工具

功能：
1. 统计消费者总支出和总属性（食品营养供给 + 非食品满意度）
2. 统计企业利润和生产效率（labor_productivity_factor）
3. 分析创新事件与指标提升的关联性
4. 分析创新与市场占有率的关联性
5. 对比两个实验的结果
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

MAX_REASONABLE_PROFIT = 1e12  # Cap unrealistic monthly profits (~1 trillion)


@dataclass
class AnalysisConfig:
    """分析配置"""
    experiment_dir: str
    experiment_name: str
    metrics_to_analyze: List[str]  # 要分析的指标列表
    innovation_types: List[str] = None  # 要分析的创新类型
    market_share_type: str = "quantity_share_pct"  # 市场占有率类型: quantity_share_pct 或 revenue_share_pct


@dataclass
class ConsumerMetrics:
    """消费者指标"""
    total_expenditure: float  # 总支出
    total_nutrition_value: float  # 食品总营养值
    total_satisfaction_value: float  # 非食品总满意度值
    total_attribute_value: float  # 总属性值（营养值 + 满意度值）


@dataclass
class FirmMetrics:
    """企业指标"""
    total_profit: float  # 总利润
    avg_profit: float  # 平均利润
    profit_growth: Optional[float] = None  # 利润增长率
    labor_productivity_factors: Dict[str, List[float]] = None  # 每个企业的生产效率因子历史
    productivity_growth: Optional[Dict[str, float]] = None  # 生产效率增长率


@dataclass
class InnovationAnalysis:
    """创新分析结果"""
    innovation_events_count: int  # 创新事件数量
    innovation_improvements: Dict[str, List[Dict]]  # 各类型创新的改进情况
    innovation_correlation_with_metrics: Dict[str, float]  # 创新与指标提升的相关性
    innovation_market_share_correlation: Dict[str, Any]  # 创新与市场占有率的相关性


@dataclass
class FirmProductMetrics:
    """企业商品指标（微观指标）"""
    # 按企业ID组织的商品产量和质量数据
    firm_products: Dict[str, Dict[str, Any]]  # firm_id -> {food_quantity, food_quality, nonfood_quantity, nonfood_quality, ...}
    # 汇总统计
    total_food_quantity: float  # 总食物产量
    total_nonfood_quantity: float  # 总非食物产量
    avg_food_quality: float  # 平均食物质量
    avg_nonfood_quality: float  # 平均非食物质量


@dataclass
class MacroMetrics:
    """宏观指标"""
    gdp: float  # 国内生产总值（GDP）
    total_revenue: float  # 总营收
    total_expenditure: float  # 总支出
    # GDP计算方法：GDP = 总消费（消费支出）+ 总投资 + 政府支出 + 净出口
    # 简化版：GDP ≈ 总营收（所有企业的总收入）


@dataclass
class ExperimentMetrics:
    """实验指标结果"""
    experiment_name: str
    consumer_metrics: Optional[ConsumerMetrics] = None
    firm_metrics: Optional[FirmMetrics] = None
    innovation_analysis: Optional[InnovationAnalysis] = None
    firm_product_metrics: Optional[FirmProductMetrics] = None  # 微观指标
    macro_metrics: Optional[MacroMetrics] = None  # 宏观指标
    monthly_metrics: Optional[Dict[int, Dict[str, Any]]] = None


class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self, experiment_dir: str):
        """
        初始化分析器
        
        Args:
            experiment_dir: 实验输出目录路径
        """
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.monthly_stats_dir = self.experiment_dir / "monthly_statistics"
        self.industry_competition_dir = self.experiment_dir / "industry_competition" / "json"
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载所有数据文件"""
        # 企业月度指标
        firm_metrics_path = self.data_dir / "firm_monthly_metrics.json"
        if firm_metrics_path.exists():
            with open(firm_metrics_path, 'r', encoding='utf-8') as f:
                self.firm_metrics = json.load(f)
        else:
            self.firm_metrics = []
        
        # 创新事件
        innovation_path = self.data_dir / "innovation_events.json"
        if innovation_path.exists():
            with open(innovation_path, 'r', encoding='utf-8') as f:
                self.innovation_events = json.load(f)
        else:
            self.innovation_events = []
        
        # 创新配置（包含企业的 labor_productivity_factor）
        innovation_config_path = self.data_dir / "innovation_configs.json"
        if innovation_config_path.exists():
            with open(innovation_config_path, 'r', encoding='utf-8') as f:
                self.innovation_configs = json.load(f)
        else:
            self.innovation_configs = {}
        
        # 家庭购买记录
        purchase_records_path = self.data_dir / "household_purchase_records.json"
        if purchase_records_path.exists():
            with open(purchase_records_path, 'r', encoding='utf-8') as f:
                self.purchase_records = json.load(f)
        else:
            self.purchase_records = {}
        
        # 生产统计
        production_path = self.data_dir / "production_statistics.json"
        if production_path.exists():
            with open(production_path, 'r', encoding='utf-8') as f:
                self.production_stats = json.load(f)
        else:
            self.production_stats = {}
        
        # 行业竞争数据（市场占有率）
        self.industry_competition = {}
        if self.industry_competition_dir.exists():
            for month_file in sorted(self.industry_competition_dir.glob("month_*.json")):
                month_num = int(month_file.stem.split('_')[1])
                with open(month_file, 'r', encoding='utf-8') as f:
                    self.industry_competition[month_num] = json.load(f)
    
    def _safe_corr(self, series_a: pd.Series, series_b: pd.Series) -> Optional[float]:
        """
        安全计算相关性，若数据方差不足则返回 None
        
        为什么无法计算相关系数：
        1. 如果序列中唯一值的数量少于2个（nunique() < 2），说明数据没有变化
           例如：所有企业的创新事件数量都是1，或者所有企业的市场占有率变化都是相同的值
        2. 相关系数需要两个变量都有变化才能计算，如果其中一个或两个都是常数，就无法计算
        
        常见原因：
        - 创新事件数量相同：所有数据点的创新数量都一样（例如都是1个）
        - 市场占有率变化相同：所有数据点的市场占有率变化都一样（例如都是0或都是某个固定值）
        - 数据点太少且值都相同：虽然有很多数据点，但取值都相同
        
        Returns:
            相关系数，如果无法计算则返回 None
        """
        # 检查序列中唯一值的数量
        unique_a = series_a.nunique()
        unique_b = series_b.nunique()
        
        if unique_a < 2:
            # 序列a没有变化，无法计算相关性
            return None
        if unique_b < 2:
            # 序列b没有变化，无法计算相关性
            return None
        
        try:
            corr = series_a.corr(series_b)
            # 检查是否为 NaN（可能因为标准差为0）
            if pd.isna(corr):
                return None
            return corr
        except Exception:
            # 计算过程中出现错误（例如除以0）
            return None
    
    def _sum_nutrition_supply(self, nutrition_supply: Dict, quantity: float = 1.0) -> float:
        """计算营养供给的总值，并乘以购买数量"""
        if not nutrition_supply:
            return 0.0
        carb = nutrition_supply.get('carbohydrate_g', 0.0)
        protein = nutrition_supply.get('protein_g', 0.0)
        fat = nutrition_supply.get('fat_g', 0.0)
        water = nutrition_supply.get('water_g', 0.0)
        vitamin = nutrition_supply.get('vitamin_index', 0.0)
        mineral = nutrition_supply.get('mineral_index', 0.0)
        
        calories = 4 * (carb + protein) + 9 * fat
        hydration_score = water * 0.25
        micronutrient_score = (vitamin + mineral) * 100
        return (calories + hydration_score + micronutrient_score) * quantity
    
    def _sum_satisfaction_attributes(self, satisfaction_attrs: Dict, quantity: float = 1.0) -> float:
        """计算满意度属性的总值，并乘以购买数量"""
        if not satisfaction_attrs:
            return 0.0

        def recurse(node: Any) -> float:
            total = 0.0
            if isinstance(node, dict):
                monthly_supply = node.get('monthly_supply')
                if isinstance(monthly_supply, (int, float)):
                    total += float(monthly_supply) * 100
                for key, value in node.items():
                    if key == 'monthly_supply':
                        continue
                    total += recurse(value)
            elif isinstance(node, list):
                for item in node:
                    total += recurse(item)
            elif isinstance(node, (int, float)):
                total += float(node)
            return total

        total_value = recurse(satisfaction_attrs)
        return total_value * quantity
    
    def _get_record_quantity(self, record: Dict, product: Dict) -> float:
        """获取购买数量，默认为 1"""
        quantity = record.get('quantity')
        if quantity is None or quantity <= 0:
            quantity = product.get('amount')
        if quantity is None or quantity <= 0:
            quantity = 1.0
        return float(quantity)
    
    def _sanitize_profit(self, record: Dict) -> Optional[float]:
        """过滤异常利润值"""
        profit = record.get('monthly_profit')
        revenue = record.get('monthly_revenue', 0.0)
        expenses = record.get('monthly_expenses', 0.0)
        
        if profit is None:
            profit = revenue - expenses
        if profit is None or not math.isfinite(profit):
            return None
        if abs(profit) > MAX_REASONABLE_PROFIT:
            return None
        return float(profit)
    
    def _build_productivity_history(self) -> Dict[str, Dict[int, float]]:
        """构建企业生产效率因子历史"""
        history: Dict[str, Dict[int, float]] = {}
        for firm_id, config in self.innovation_configs.items():
            initial_factor = config.get('labor_productivity_factor')
            if initial_factor is not None:
                history.setdefault(firm_id, {})[0] = initial_factor
        for event in self.innovation_events:
            if event.get('innovation_type') == 'labor_productivity_factor':
                firm_id = event.get('company_id')
                month = event.get('month')
                new_value = event.get('new_value')
                if firm_id and month is not None and new_value is not None:
                    history.setdefault(firm_id, {})[month] = new_value
        return history
    
    def _build_profit_by_month(self) -> Dict[str, Dict[int, float]]:
        """构建 {firm_id: {month: profit}}，并过滤异常利润"""
        profit_by_month: Dict[str, Dict[int, float]] = {}
        for record in self.firm_metrics:
            firm_id = record.get('company_id')
            month = record.get('month')
            profit = self._sanitize_profit(record)
            if firm_id is None or month is None or profit is None:
                continue
            profit_by_month.setdefault(firm_id, {})[month] = profit
        return profit_by_month
    
    def analyze_consumer_metrics(self) -> ConsumerMetrics:
        """
        分析消费者指标
        
        Returns:
            消费者指标结果
        """
        total_expenditure = 0.0
        total_nutrition_value = 0.0
        total_satisfaction_value = 0.0
        
        for month_str, records in self.purchase_records.items():
            for record in records:
                # 总支出
                total_expenditure += record.get('total_amount', 0.0)
                
                # 获取产品属性
                product = record.get('product', {})
                attributes = product.get('attributes', {})
                is_food = product.get('is_food', False) or attributes.get('is_food', False)
                quantity = self._get_record_quantity(record, product)
                
                # 食品：计算营养供给
                if is_food:
                    nutrition_supply = attributes.get('nutrition_supply') or product.get('nutrition_supply')
                    if nutrition_supply:
                        total_nutrition_value += self._sum_nutrition_supply(nutrition_supply, quantity)
                
                # 非食品：计算满意度属性
                else:
                    satisfaction_attrs = attributes.get('satisfaction_attributes') or product.get('satisfaction_attributes')
                    if satisfaction_attrs:
                        total_satisfaction_value += self._sum_satisfaction_attributes(satisfaction_attrs, quantity)
        
        return ConsumerMetrics(
            total_expenditure=total_expenditure,
            total_nutrition_value=total_nutrition_value,
            total_satisfaction_value=total_satisfaction_value,
            total_attribute_value=total_nutrition_value + total_satisfaction_value
        )
    
    def analyze_firm_metrics(self) -> FirmMetrics:
        """
        分析企业指标
        
        指标说明：
        - total_profit（总利润）：所有企业所有月份利润的累加总和
          计算方式：遍历所有企业的所有月份，累加每个企业每个月的利润
        
        - avg_profit（平均利润）：总利润除以记录总数
          计算方式：total_profit / (企业数 × 月数)
          含义：平均每个企业每个月的利润水平
        
        - profit_growth（利润增长率）：系统整体利润从第一个月到最后一个月的变化率
          计算方式：(最后一个月所有企业总利润 - 第一个月所有企业总利润) / |第一个月总利润| × 100%
          含义：反映整个仿真期间企业利润的整体变化趋势（正值表示增长，负值表示下降）
        
        Returns:
            企业指标结果
        """
        if not self.firm_metrics:
            return None
        
        profit_by_month = self._build_profit_by_month()
        
        # 计算总利润：累加所有企业所有月份的利润
        total_profit = 0.0
        record_count = 0
        monthly_totals: Dict[int, float] = {}  # 每个月份所有企业的利润总和
        
        for month_dict in profit_by_month.values():
            for month, profit in month_dict.items():
                total_profit += profit
                record_count += 1
                monthly_totals[month] = monthly_totals.get(month, 0.0) + profit
        
        # 平均利润 = 总利润 / 记录总数（企业数 × 月数）
        avg_profit = total_profit / record_count if record_count else 0.0
        
        # 利润增长率：系统整体利润从第一个月到最后一个月的变化率
        # 公式：(最后一个月总利润 - 第一个月总利润) / |第一个月总利润| × 100%
        profit_growth = None
        if len(monthly_totals) >= 2:
            first_month = min(monthly_totals)
            last_month = max(monthly_totals)
            first_value = monthly_totals[first_month]
            last_value = monthly_totals[last_month]
            if first_value != 0:
                profit_growth = (last_value - first_value) / abs(first_value) * 100
        
        # 获取每个企业的 labor_productivity_factor 历史
        productivity_history = self._build_productivity_history()
        
        labor_productivity_factors: Dict[str, List[float]] = {}
        productivity_growth: Dict[str, float] = {}
        for firm_id, history in productivity_history.items():
            ordered_months = sorted(history.keys())
            ordered_values = [history[m] for m in ordered_months]
            labor_productivity_factors[firm_id] = ordered_values
            if len(ordered_values) > 1 and ordered_values[0]:
                productivity_growth[firm_id] = (
                    (ordered_values[-1] - ordered_values[0]) / ordered_values[0] * 100
                )
        
        return FirmMetrics(
            total_profit=total_profit,
            avg_profit=avg_profit,
            profit_growth=profit_growth,
            labor_productivity_factors=labor_productivity_factors,
            productivity_growth=productivity_growth
        )
    
    def analyze_innovation_improvements(
        self, 
        innovation_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        分析创新改进情况
        
        Args:
            innovation_types: 要分析的创新类型列表（None 表示所有类型）
        
        Returns:
            各类型创新的改进情况
        """
        if innovation_types is None:
            innovation_types = ['labor_productivity_factor', 'price', 'profit_margin']
        
        improvements = {inv_type: [] for inv_type in innovation_types}
        
        for event in self.innovation_events:
            inv_type = event.get('innovation_type')
            if inv_type not in innovation_types:
                continue
            
            firm_id = event.get('company_id')
            month = event.get('month')
            old_value = event.get('old_value')
            new_value = event.get('new_value')
            price_change = event.get('price_change')
            
            improvement = {
                'firm_id': firm_id,
                'month': month,
                'old_value': old_value,
                'new_value': new_value,
                'improvement': None,
                'improvement_pct': None
            }
            
            # 计算改进值
            if old_value is not None and new_value is not None:
                if old_value > 0:
                    improvement['improvement'] = new_value - old_value
                    improvement['improvement_pct'] = (new_value - old_value) / old_value * 100
                else:
                    improvement['improvement'] = new_value - old_value
            
            # 对于 price 类型，使用 price_change
            if inv_type == 'price' and price_change is not None:
                improvement['price_change'] = price_change
                improvement['improvement'] = price_change - 1.0  # price_change 是倍数
            
            improvements[inv_type].append(improvement)
        
        return improvements
    
    def analyze_innovation_correlation_with_metrics(
        self,
        innovation_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        分析创新与指标提升的相关性
        
        Args:
            innovation_types: 要分析的创新类型列表
        
        Returns:
            创新与各指标的相关性
        """
        if innovation_types is None:
            innovation_types = ['labor_productivity_factor', 'price', 'profit_margin']
        
        firm_profit_by_month = self._build_profit_by_month()
        firm_productivity_by_month = self._build_productivity_history()
        
        correlations = {}
        
        # 分析每种创新类型与利润提升的相关性
        for inv_type in innovation_types:
            innovation_data = []
            
            for event in self.innovation_events:
                if event.get('innovation_type') != inv_type:
                    continue
                
                firm_id = event.get('company_id')
                month = event.get('month')
                old_value = event.get('old_value')
                new_value = event.get('new_value')
                
                if not (firm_id and old_value is not None and new_value is not None):
                    continue
                
                # 计算创新改进
                if old_value > 0:
                    improvement_pct = (new_value - old_value) / old_value * 100
                else:
                    improvement_pct = new_value - old_value
                
                # 获取下个月的利润变化
                if firm_id in firm_profit_by_month:
                    current_profit = firm_profit_by_month[firm_id].get(month, None)
                    next_profit = firm_profit_by_month[firm_id].get(month + 1, None)
                    
                    if current_profit is not None and next_profit is not None and current_profit > 0:
                        profit_growth = (next_profit - current_profit) / current_profit * 100
                        innovation_data.append({
                            'innovation_improvement': improvement_pct,
                            'profit_growth': profit_growth
                        })
            
            # 计算相关性
            if len(innovation_data) > 1:
                df = pd.DataFrame(innovation_data)
                correlations[f'{inv_type}_vs_profit'] = self._safe_corr(
                    df['innovation_improvement'],
                    df['profit_growth'],
                )
            else:
                correlations[f'{inv_type}_vs_profit'] = None
        
        # 分析 labor_productivity_factor 创新与生产效率提升的相关性
        if 'labor_productivity_factor' in innovation_types:
            productivity_data = []
            
            for firm_id, factors_by_month in firm_productivity_by_month.items():
                months = sorted(factors_by_month.keys())
                for i in range(len(months) - 1):
                    month1 = months[i]
                    month2 = months[i + 1]
                    factor1 = factors_by_month[month1]
                    factor2 = factors_by_month[month2]
                    
                    if factor1 > 0:
                        growth = (factor2 - factor1) / factor1 * 100
                        productivity_data.append({
                            'month': month1,
                            'productivity_growth': growth
                        })
            
            if len(productivity_data) > 1:
                correlations['productivity_self_correlation'] = 1.0
        
        return correlations
    
    def get_firm_market_share(self, firm_id: str, month: int, share_type: str = "quantity_share_pct") -> Optional[float]:
        """
        获取企业在指定月份的市场占有率
        
        Args:
            firm_id: 企业ID
            month: 月份
            share_type: 占有率类型 ("quantity_share_pct" 或 "revenue_share_pct")
        
        Returns:
            市场占有率百分比，如果不存在则返回 None
        """
        if month not in self.industry_competition:
            return None
        
        month_data = self.industry_competition[month]
        
        # 遍历所有行业
        for industry_name, industry_data in month_data.items():
            if 'firms' in industry_data:
                for firm in industry_data['firms']:
                    if firm.get('firm_id') == firm_id:
                        return firm.get(share_type)
        
        return None
    
    def analyze_innovation_market_share_correlation(
        self, 
        innovation_types: Optional[List[str]] = None,
        market_share_type: str = "quantity_share_pct",
        months_ahead: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        分析创新与市场占有率的关联性
        
        指标说明：
        - correlation（相关系数）：创新事件数量与市场占有率变化的皮尔逊相关系数
          取值范围：[-1, 1]，接近1表示正相关，接近-1表示负相关，接近0表示无相关
        
        - avg_share_delta（平均市场占有率变化量）：
          计算方式：对于每个创新事件，计算(未来月份市场占有率 - 当前月份市场占有率)，然后求平均
          含义：平均每次创新后，市场占有率提升了多少个百分点（正数表示提升，负数表示下降）
        
        - avg_ratio（平均每个创新事件的市场占有率变化率）：
          计算方式：(未来月份市场占有率 - 当前月份市场占有率) / 创新事件数量，然后求平均
          含义：平均每个创新事件能带来多少市场占有率提升（单位：百分点/创新事件）
          例如：avg_ratio = 2.0 表示平均每个创新事件能带来 2 个百分点的市场占有率提升
        
        Args:
            innovation_types: 要分析的创新类型列表（None 表示所有类型，默认只分析三种主要类型）
            market_share_type: 市场占有率类型（"quantity_share_pct" 或 "revenue_share_pct"）
            months_ahead: 向前看几个月（例如 [1, 2, 3] 表示分析未来1、2、3个月的市场占有率变化）
        
        Returns:
            包含相关性分析结果的字典
        """
        if not self.innovation_events or not self.industry_competition:
            return {
                "error": "缺少创新事件或行业竞争数据",
                "correlation": None,
                "firm_level_correlations": []
            }
        
        # 如果未指定创新类型，默认包含所有类型
        if innovation_types is None:
            # 从所有创新事件中提取所有唯一的类型
            innovation_types = list(set(event.get('innovation_type') for event in self.innovation_events if event.get('innovation_type')))
        
        # 如果传入空列表，也表示包含所有类型
        if innovation_types == []:
            innovation_types = list(set(event.get('innovation_type') for event in self.innovation_events if event.get('innovation_type')))
        
        if not months_ahead:
            months_ahead = [1]
        
        # 构建创新事件数据：每个创新事件作为一个样本点
        innovation_records = []
        for event in self.innovation_events:
            if event.get('innovation_type') in innovation_types:
                innovation_records.append({
                    'firm_id': event.get('company_id'),
                    'month': event.get('month'),
                    'innovation_type': event.get('innovation_type'),
                })
        
        if not innovation_records:
            return {
                "error": "没有找到符合条件的创新事件",
                "correlation": None,
                "firm_level_correlations": [],
                "total_innovation_events": 0,
                "boundary_events": 0,
                "correlation_events": 0,
            }
        
        total_innovation_count = len(innovation_records)
        
        per_horizon_summary = {}
        per_horizon_firm_corrs = {}
        
        for horizon in months_ahead:
            # 统计每个（企业、月份）的创新数量（只包括有创新的）
            firm_month_counts = {}  # (firm_id, month) -> 创新数量
            for event in innovation_records:
                firm_id = event['firm_id']
                month = event['month']
                key = (firm_id, month)
                firm_month_counts[key] = firm_month_counts.get(key, 0) + 1
            
            # 获取所有（企业、月份）组合（包括创新次数为0的）
            # 从行业竞争数据中获取所有企业和月份的组合
            all_firm_month_combos = set()  # 所有（企业、月份）组合
            for month_num, month_data in self.industry_competition.items():
                # 检查是否有未来月份的数据
                if month_num + horizon not in self.industry_competition:
                    continue  # 跳过没有未来数据的月份
                
                # 遍历所有行业中的所有企业
                for industry_name, industry_data in month_data.items():
                    if 'firms' in industry_data:
                        for firm in industry_data['firms']:
                            firm_id = firm.get('firm_id')
                            if firm_id:
                                all_firm_month_combos.add((firm_id, month_num))
            
            # 构建所有（企业、月份）组合的记录（包括创新次数为0的）
            all_records = []  # 所有（企业、月份）组合的记录
            boundary_count = 0  # 边界样本：没有未来数据的
            
            for firm_id, month in all_firm_month_combos:
                # 获取该企业该月的创新数量（如果没有创新则为0）
                innovation_count = firm_month_counts.get((firm_id, month), 0)
                
                current_share = self.get_firm_market_share(firm_id, month, market_share_type)
                future_share = self.get_firm_market_share(firm_id, month + horizon, market_share_type)
                
                # 如果当前月份或未来月份的数据缺失，记录为边界样本
                if current_share is None or future_share is None:
                    boundary_count += 1
                    continue
                
                # 计算市场占有率变化量：未来月份占有率 - 当前月份占有率
                share_delta = future_share - current_share
                
                # 每个（企业、月份）组合作为一个样本点
                all_records.append({
                    'firm_id': firm_id,
                    'month': month,
                    'innovation_count': innovation_count,  # 该企业该月的创新事件总数（可能为0）
                    'share_delta': share_delta,  # 该月后的市场占有率变化
                })
            
            # 计算参与相关性计算的创新事件数量（只针对有创新的样本）
            # 统计有创新事件的（企业、月份）组合中有未来数据的创新事件数
            correlation_events_count = 0
            innovation_boundary_count = 0  # 有创新但无未来数据的创新事件数
            
            for (firm_id, month), count in firm_month_counts.items():
                current_share = self.get_firm_market_share(firm_id, month, market_share_type)
                future_share = self.get_firm_market_share(firm_id, month + horizon, market_share_type)
                if current_share is not None and future_share is not None:
                    correlation_events_count += count  # 累加创新事件数
                else:
                    innovation_boundary_count += count  # 有创新但无未来数据的创新事件数
            
            # 对于没有创新的（企业、月份）组合，它们也参与相关性计算，但创新次数为0
            
            if not all_records:
                per_horizon_summary[horizon] = {
                    "data_points": 0,
                    "correlation": None,
                    "avg_share_delta": None,
                    "avg_ratio": None,
                    "total_innovation_events": total_innovation_count,
                    "boundary_events": boundary_count,
                    "correlation_events": correlation_events_count,
                    "zero_innovation_count": 0,  # 创新次数为0的样本数
                }
                per_horizon_firm_corrs[horizon] = []
                continue
            
            # 创建DataFrame：每个（企业、月份）组合作为一个样本点
            # 包括创新次数为0的组合
            corr_df = pd.DataFrame(all_records)
            
            # 统计创新次数为0的样本数
            zero_innovation_count = len(corr_df[corr_df['innovation_count'] == 0])
            
            # 诊断信息：检查数据是否有足够的变化
            innovation_unique = corr_df['innovation_count'].nunique()
            delta_unique = corr_df['share_delta'].nunique()
            innovation_min = int(corr_df['innovation_count'].min())
            innovation_max = int(corr_df['innovation_count'].max())
            delta_min = corr_df['share_delta'].min()
            delta_max = corr_df['share_delta'].max()
            
            # 计算创新事件数量与市场占有率变化的相关系数
            # 每个创新事件作为一个样本点，但同一个（企业、月份）的多个创新事件会有相同的值
            # 这样计算的是：创新事件数量 vs 市场占有率变化的相关性
            corr_value = self._safe_corr(
                corr_df['innovation_count'], 
                corr_df['share_delta']
            )
            
            # 如果无法计算相关系数，提供诊断信息
            corr_error_reason = None
            if corr_value is None:
                if innovation_unique < 2:
                    corr_error_reason = f"创新事件数量缺乏变化（唯一值数={innovation_unique}，范围=[{innovation_min}, {innovation_max}]）"
                elif delta_unique < 2:
                    corr_error_reason = f"市场占有率变化缺乏变化（唯一值数={delta_unique}，范围=[{delta_min:.4f}, {delta_max:.4f}]）"
                else:
                    corr_error_reason = "数据方差不足或存在计算错误"
            
            # 计算平均市场占有率变化量：对所有（企业、月份）组合的市场占有率变化求平均
            # 包括创新次数为0的组合
            avg_share_delta = corr_df['share_delta'].mean()
            
            # 计算平均每个创新事件带来的市场占有率变化
            # 只对有创新的样本计算：市场占有率变化 / 创新次数，然后求平均
            # 或者：只计算有创新事件的样本的平均市场占有率变化
            with_innovation = corr_df[corr_df['innovation_count'] > 0]
            if len(with_innovation) > 0:
                # 对于有创新的样本，计算每个创新事件的平均效果
                # 方法：对每个（企业、月份）组合，share_delta / innovation_count，然后求平均
                avg_ratio = (with_innovation['share_delta'] / with_innovation['innovation_count']).mean()
            else:
                avg_ratio = None  # 没有创新事件，无法计算
            
            # 验证：创新事件总数 = 参与correlation的样本 + 边界样本（只针对有创新的样本）
            # 注意：这里只验证有创新的样本，不包括创新次数为0的样本
            # boundary_count 包括所有（企业、月份）组合中没有未来数据的数量（包括创新次数为0的）
            # 但验证时应该只使用有创新但无未来数据的数量
            total_should_be = correlation_events_count + innovation_boundary_count
            validation_passed = (total_innovation_count == total_should_be)
            
            # 获取唯一的企业-月份组合数（用于数据点统计）
            # 包括创新次数为0的组合
            unique_firm_month = corr_df[['firm_id', 'month']].drop_duplicates()
            data_points_count = len(unique_firm_month)
            
            per_horizon_summary[horizon] = {
                "data_points": data_points_count,  # 数据点数量（企业-月份组合数，包括创新次数为0的）
                "correlation": corr_value,  # 相关系数：创新数量与市场占有率变化的关联性（包括0创新）
                "correlation_error": corr_error_reason,  # 如果无法计算相关系数，说明原因
                "avg_share_delta": avg_share_delta,  # 平均市场占有率变化量（百分点）- 所有样本的平均（包括0创新）
                "avg_ratio": avg_ratio,  # 平均每个创新事件带来的市场占有率变化（只计算有创新的样本）
                "total_innovation_events": total_innovation_count,  # 总创新事件数（符合类型）
                "boundary_events": boundary_count,  # 边界样本：所有（企业、月份）组合中没有未来数据的数量（包括0创新）
                "innovation_boundary_events": innovation_boundary_count,  # 有创新但无未来数据的创新事件数
                "correlation_events": correlation_events_count,  # 参与相关性计算的创新事件数（有创新的）
                "zero_innovation_count": zero_innovation_count,  # 创新次数为0的样本数
                "validation": {
                    "total_should_be": total_should_be,
                    "validation_passed": validation_passed,  # 验证：总数 = correlation样本 + 创新边界样本
                },
                # 诊断信息
                "diagnostics": {
                    "innovation_unique_values": innovation_unique,
                    "innovation_range": [innovation_min, innovation_max],
                    "delta_unique_values": delta_unique,
                    "delta_range": [delta_min, delta_max],
                }
            }
            
            # 按企业级别的相关性（每个企业在不同月份可能有不同的创新数量和市场占有率变化）
            # 使用唯一的企业-月份组合（因为同一企业同一月的多个创新事件有相同的值）
            firm_month_unique = corr_df[['firm_id', 'month', 'innovation_count', 'share_delta']].drop_duplicates()
            firm_corrs = []
            for firm_id in firm_month_unique['firm_id'].unique():
                firm_data = firm_month_unique[firm_month_unique['firm_id'] == firm_id]
                if len(firm_data) > 1:
                    firm_corr = self._safe_corr(firm_data['innovation_count'], firm_data['share_delta'])
                    # 该企业在所有月份的总创新事件数
                    total_innovations = firm_data['innovation_count'].sum()
                    # 该企业在所有月份的市场占有率变化总和（注意：如果有多个创新事件在同一月，会重复计算）
                    # 但由于我们已经去重了，这里实际上是对不同月份求和
                    total_delta = firm_data['share_delta'].sum()
                    avg_delta_per_innovation = (total_delta / total_innovations) if total_innovations > 0 else None
                    firm_corrs.append({
                        'firm_id': firm_id,
                        'correlation': firm_corr,
                        'data_points': len(firm_data),  # 该企业有数据的月份数
                        'avg_ratio': avg_delta_per_innovation,
                    })
            per_horizon_firm_corrs[horizon] = firm_corrs
        
        return {
            "per_horizon": per_horizon_summary,
            "firm_level_correlations": per_horizon_firm_corrs,
            "total_innovation_events": total_innovation_count,  # 总创新事件数（所有符合条件的）
        }
    
    def analyze_firm_product_metrics(self) -> FirmProductMetrics:
        """
        分析企业商品指标（微观指标）
        包括：每个企业商品的产量、质量（分别考虑食物和非食物）
        
        Returns:
            企业商品指标结果
        """
        # 存储每个企业的商品数据
        firm_products: Dict[str, Dict[str, Any]] = {}
        
        # 总产量和质量（用于汇总）
        total_food_quantity = 0.0
        total_nonfood_quantity = 0.0
        total_food_quality = 0.0
        total_nonfood_quality = 0.0
        food_count = 0
        nonfood_count = 0
        
        # 提前构建product_id到is_food和quality的映射（只遍历一次购买记录）
        product_food_map = {}
        product_quality_map = {}  # product_id -> (quality, is_food)
        
        for month_str, records in self.purchase_records.items():
            for record in records:
                product = record.get('product', {})
                product_id = product.get('product_id')
                if product_id and product_id not in product_food_map:
                    is_food = product.get('is_food', False)
                    product_food_map[product_id] = is_food
                    
                    # 计算质量
                    if is_food:
                        quality = self._calculate_food_quality_from_product(product)
                    else:
                        quality = self._calculate_nonfood_quality_from_product(product)
                    
                    if quality is not None:
                        product_quality_map[product_id] = (quality, is_food)
        
        # 从行业竞争数据中获取企业商品信息
        for month_num, month_data in self.industry_competition.items():
            for industry_name, industry_data in month_data.items():
                if 'firms' not in industry_data:
                    continue
                
                for firm_data in industry_data['firms']:
                    firm_id = firm_data.get('firm_id')
                    if not firm_id:
                        continue
                    
                    # 初始化企业数据
                    if firm_id not in firm_products:
                        firm_products[firm_id] = {
                            'firm_name': firm_data.get('firm_name', ''),
                            'food_quantity': 0.0,
                            'nonfood_quantity': 0.0,
                            'food_quality_scores': [],
                            'nonfood_quality_scores': [],
                        }
                    
                    # 获取商品详情，区分食物和非食物
                    product_details = firm_data.get('product_details', [])
                    
                    for product_detail in product_details:
                        product_id = product_detail.get('product_id')
                        quantity_sold = product_detail.get('quantity_sold', 0.0)
                        
                        # 判断是否为食物
                        is_food = product_food_map.get(product_id, False)
                        
                        if is_food:
                            firm_products[firm_id]['food_quantity'] += quantity_sold
                            total_food_quantity += quantity_sold
                            
                            # 获取质量分数
                            quality_info = product_quality_map.get(product_id)
                            if quality_info:
                                quality_score, _ = quality_info
                                firm_products[firm_id]['food_quality_scores'].append(quality_score)
                                total_food_quality += quality_score
                                food_count += 1
                        else:
                            firm_products[firm_id]['nonfood_quantity'] += quantity_sold
                            total_nonfood_quantity += quantity_sold
                            
                            # 获取质量分数
                            quality_info = product_quality_map.get(product_id)
                            if quality_info:
                                quality_score, _ = quality_info
                                firm_products[firm_id]['nonfood_quality_scores'].append(quality_score)
                                total_nonfood_quality += quality_score
                                nonfood_count += 1
        
        # 计算平均质量
        avg_food_quality = total_food_quality / food_count if food_count > 0 else 0.0
        avg_nonfood_quality = total_nonfood_quality / nonfood_count if nonfood_count > 0 else 0.0
        
        # 计算每个企业的平均质量
        for firm_id, data in firm_products.items():
            food_scores = data['food_quality_scores']
            nonfood_scores = data['nonfood_quality_scores']
            data['avg_food_quality'] = sum(food_scores) / len(food_scores) if food_scores else 0.0
            data['avg_nonfood_quality'] = sum(nonfood_scores) / len(nonfood_scores) if nonfood_scores else 0.0
        
        return FirmProductMetrics(
            firm_products=firm_products,
            total_food_quantity=total_food_quantity,
            total_nonfood_quantity=total_nonfood_quantity,
            avg_food_quality=avg_food_quality,
            avg_nonfood_quality=avg_nonfood_quality
        )
    
    def _calculate_food_quality_from_product(self, product: Dict) -> Optional[float]:
        """从产品信息计算食物质量分数（直接相加，无系数）"""
        nutrition_supply = product.get('nutrition_supply') or product.get('attributes', {}).get('nutrition_supply')
        if nutrition_supply:
            carb = nutrition_supply.get('carbohydrate_g', 0.0)
            protein = nutrition_supply.get('protein_g', 0.0)
            fat = nutrition_supply.get('fat_g', 0.0)
            water = nutrition_supply.get('water_g', 0.0)
            vitamin = nutrition_supply.get('vitamin_index', 0.0)
            mineral = nutrition_supply.get('mineral_index', 0.0)
            
            # 直接相加，无系数
            quality = carb + protein + fat + water + vitamin + mineral
            return quality
        return None
    
    def _calculate_nonfood_quality_from_product(self, product: Dict) -> Optional[float]:
        """从产品信息计算非食物质量分数（直接相加，无系数）"""
        satisfaction_attrs = product.get('satisfaction_attributes') or product.get('attributes', {}).get('satisfaction_attributes')
        if satisfaction_attrs:
            # 直接递归求和，不乘100也不除100
            def recurse_sum(node: Any) -> float:
                total = 0.0
                if isinstance(node, dict):
                    for key, value in node.items():
                        total += recurse_sum(value)
                elif isinstance(node, list):
                    for item in node:
                        total += recurse_sum(item)
                elif isinstance(node, (int, float)):
                    total += float(node)
                return total
            
            quality = recurse_sum(satisfaction_attrs)
            return quality
        return None
    
    def analyze_firm_monthly_products(self, firm_id: str) -> Dict[int, Dict[str, Any]]:
        """
        分析特定企业每个月的产量和质量（按月份统计）
        
        Args:
            firm_id: 企业ID
        
        Returns:
            按月份组织的数据：{month: {food_quantity, nonfood_quantity, food_quality, nonfood_quality, ...}}
        """
        monthly_data: Dict[int, Dict[str, Any]] = {}
        
        # 提前构建product_id到is_food和quality的映射
        product_food_map = {}
        product_quality_map = {}  # product_id -> (quality, is_food)
        
        for month_str, records in self.purchase_records.items():
            for record in records:
                product = record.get('product', {})
                product_id = product.get('product_id')
                if product_id and product_id not in product_food_map:
                    is_food = product.get('is_food', False)
                    product_food_map[product_id] = is_food
                    
                    # 计算质量
                    if is_food:
                        quality = self._calculate_food_quality_from_product(product)
                    else:
                        quality = self._calculate_nonfood_quality_from_product(product)
                    
                    if quality is not None:
                        product_quality_map[product_id] = (quality, is_food)
        
        # 从行业竞争数据中获取该企业每个月的商品信息
        for month_num, month_data in self.industry_competition.items():
            for industry_name, industry_data in month_data.items():
                if 'firms' not in industry_data:
                    continue
                
                # 查找该企业
                firm_data = None
                for f in industry_data['firms']:
                    if f.get('firm_id') == firm_id:
                        firm_data = f
                        break
                
                if not firm_data:
                    continue
                
                # 初始化该月份的数据
                if month_num not in monthly_data:
                    monthly_data[month_num] = {
                        'firm_name': firm_data.get('firm_name', ''),
                        'food_quantity': 0.0,
                        'nonfood_quantity': 0.0,
                        'food_quality_scores': [],
                        'nonfood_quality_scores': [],
                        'product_count': 0,
                    }
                
                # 获取商品详情
                product_details = firm_data.get('product_details', [])
                monthly_data[month_num]['product_count'] = len(product_details)
                
                for product_detail in product_details:
                    product_id = product_detail.get('product_id')
                    quantity_sold = product_detail.get('quantity_sold', 0.0)
                    
                    # 判断是否为食物
                    is_food = product_food_map.get(product_id, False)
                    
                    if is_food:
                        monthly_data[month_num]['food_quantity'] += quantity_sold
                        quality_info = product_quality_map.get(product_id)
                        if quality_info:
                            quality_score, _ = quality_info
                            monthly_data[month_num]['food_quality_scores'].append(quality_score)
                    else:
                        monthly_data[month_num]['nonfood_quantity'] += quantity_sold
                        quality_info = product_quality_map.get(product_id)
                        if quality_info:
                            quality_score, _ = quality_info
                            monthly_data[month_num]['nonfood_quality_scores'].append(quality_score)
                
                # 计算该月份的平均质量
                food_scores = monthly_data[month_num]['food_quality_scores']
                nonfood_scores = monthly_data[month_num]['nonfood_quality_scores']
                monthly_data[month_num]['avg_food_quality'] = sum(food_scores) / len(food_scores) if food_scores else 0.0
                monthly_data[month_num]['avg_nonfood_quality'] = sum(nonfood_scores) / len(nonfood_scores) if nonfood_scores else 0.0
                monthly_data[month_num]['total_quantity'] = monthly_data[month_num]['food_quantity'] + monthly_data[month_num]['nonfood_quantity']
        
        return monthly_data
    
    def analyze_macro_metrics(self) -> MacroMetrics:
        """
        分析宏观指标
        包括：GDP（国内生产总值）
        
        Returns:
            宏观指标结果
        """
        # 计算总营收（所有企业的总收入）
        total_revenue = 0.0
        for firm_record in self.firm_metrics:
            revenue = firm_record.get('monthly_revenue', 0.0)
            if revenue and math.isfinite(revenue):
                total_revenue += revenue
        
        # 计算总支出（消费者总支出）
        total_expenditure = 0.0
        for month_str, records in self.purchase_records.items():
            for record in records:
                total_expenditure += record.get('total_amount', 0.0)
        
        # GDP计算：简化版，GDP ≈ 总营收（所有企业的总收入）
        # 更准确的计算应该是：GDP = 总消费 + 总投资 + 政府支出 + 净出口
        # 但在这个模型中，我们使用总营收作为GDP的近似值
        gdp = total_revenue
        
        return MacroMetrics(
            gdp=gdp,
            total_revenue=total_revenue,
            total_expenditure=total_expenditure
        )
    
    def analyze_all_metrics(self, config: AnalysisConfig) -> ExperimentMetrics:
        """
        分析所有指标
        
        Args:
            config: 分析配置
        
        Returns:
            实验指标结果
        """
        metrics = ExperimentMetrics(experiment_name=config.experiment_name)
        
        # 分析消费者指标
        if 'consumer_metrics' in config.metrics_to_analyze:
            metrics.consumer_metrics = self.analyze_consumer_metrics()
        
        # 分析企业指标
        if 'firm_metrics' in config.metrics_to_analyze:
            metrics.firm_metrics = self.analyze_firm_metrics()
        
        # 分析创新
        if 'innovation_analysis' in config.metrics_to_analyze:
            innovation_improvements = self.analyze_innovation_improvements(config.innovation_types)
            innovation_correlation = self.analyze_innovation_correlation_with_metrics(config.innovation_types)
            innovation_market_share = self.analyze_innovation_market_share_correlation(
                innovation_types=config.innovation_types,
                market_share_type=config.market_share_type,
                months_ahead=[1, 2, 3],
            )
            
            metrics.innovation_analysis = InnovationAnalysis(
                innovation_events_count=len(self.innovation_events),
                innovation_improvements=innovation_improvements,
                innovation_correlation_with_metrics=innovation_correlation,
                innovation_market_share_correlation=innovation_market_share
            )
        
        # 分析微观指标（企业商品产量和质量）
        if 'firm_product_metrics' in config.metrics_to_analyze:
            metrics.firm_product_metrics = self.analyze_firm_product_metrics()
        
        # 分析宏观指标（GDP）
        if 'macro_metrics' in config.metrics_to_analyze:
            metrics.macro_metrics = self.analyze_macro_metrics()
        
        return metrics


def compare_experiments(
    experiment1_dir: str,
    experiment2_dir: str,
    experiment1_name: str = "Experiment 1",
    experiment2_name: str = "Experiment 2",
    metrics_to_compare: List[str] = None,
    innovation_types: List[str] = None,
    market_share_type: str = "quantity_share_pct"
) -> Dict[str, Any]:
    """
    对比两个实验的结果
    
    Args:
        experiment1_dir: 第一个实验目录
        experiment2_dir: 第二个实验目录
        experiment1_name: 第一个实验名称
        experiment2_name: 第二个实验名称
        metrics_to_compare: 要对比的指标列表
        innovation_types: 要分析的创新类型
        market_share_type: 市场占有率类型
    
    Returns:
        对比结果字典
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['consumer_metrics', 'firm_metrics', 'innovation_analysis', 'firm_product_metrics', 'macro_metrics']
    
    # 创建分析器
    analyzer1 = ExperimentAnalyzer(experiment1_dir)
    analyzer2 = ExperimentAnalyzer(experiment2_dir)
    
    # 创建配置
    config1 = AnalysisConfig(
        experiment_dir=experiment1_dir,
        experiment_name=experiment1_name,
        metrics_to_analyze=metrics_to_compare,
        innovation_types=innovation_types,
        market_share_type=market_share_type
    )
    
    config2 = AnalysisConfig(
        experiment_dir=experiment2_dir,
        experiment_name=experiment2_name,
        metrics_to_analyze=metrics_to_compare,
        innovation_types=innovation_types,
        market_share_type=market_share_type
    )
    
    # 分析两个实验
    metrics1 = analyzer1.analyze_all_metrics(config1)
    metrics2 = analyzer2.analyze_all_metrics(config2)
    
    # 构建对比结果
    comparison = {
        "experiment1": {
            "name": experiment1_name,
            "consumer_metrics": metrics1.consumer_metrics.__dict__ if metrics1.consumer_metrics else None,
            "firm_metrics": metrics1.firm_metrics.__dict__ if metrics1.firm_metrics else None,
            "innovation_analysis": metrics1.innovation_analysis.__dict__ if metrics1.innovation_analysis else None,
            "firm_product_metrics": metrics1.firm_product_metrics.__dict__ if metrics1.firm_product_metrics else None,
            "macro_metrics": metrics1.macro_metrics.__dict__ if metrics1.macro_metrics else None
        },
        "experiment2": {
            "name": experiment2_name,
            "consumer_metrics": metrics2.consumer_metrics.__dict__ if metrics2.consumer_metrics else None,
            "firm_metrics": metrics2.firm_metrics.__dict__ if metrics2.firm_metrics else None,
            "innovation_analysis": metrics2.innovation_analysis.__dict__ if metrics2.innovation_analysis else None,
            "firm_product_metrics": metrics2.firm_product_metrics.__dict__ if metrics2.firm_product_metrics else None,
            "macro_metrics": metrics2.macro_metrics.__dict__ if metrics2.macro_metrics else None
        },
        "differences": {}
    }
    
    # 计算两个实验之间的差异
    # 
    # 差异指标说明：
    # - absolute（绝对差值）：实验2的值 - 实验1的值
    #   正值表示实验2高于实验1，负值表示实验2低于实验1
    # 
    # - percent（百分比变动）：(实验2的值 - 实验1的值) / 实验1的值 × 100%
    #   正值表示实验2相对于实验1的增长率，负值表示下降率
    
    if metrics1.consumer_metrics and metrics2.consumer_metrics:
        comp = metrics1.consumer_metrics
        comp2 = metrics2.consumer_metrics
        comparison["differences"]["consumer_metrics"] = {
            "total_expenditure": {
                "absolute": comp2.total_expenditure - comp.total_expenditure,  # 总支出差值
                "percent": ((comp2.total_expenditure - comp.total_expenditure) / comp.total_expenditure * 100) if comp.total_expenditure > 0 else 0  # 总支出百分比变动
            },
            "total_attribute_value": {
                "absolute": comp2.total_attribute_value - comp.total_attribute_value,  # 总属性值差值（营养值+满意度值）
                "percent": ((comp2.total_attribute_value - comp.total_attribute_value) / comp.total_attribute_value * 100) if comp.total_attribute_value > 0 else 0  # 总属性值百分比变动
            }
        }
    
    if metrics1.firm_metrics and metrics2.firm_metrics:
        comp = metrics1.firm_metrics
        comp2 = metrics2.firm_metrics
        comparison["differences"]["firm_metrics"] = {
            "total_profit": {
                "absolute": comp2.total_profit - comp.total_profit,  # 总利润差值
                "percent": ((comp2.total_profit - comp.total_profit) / comp.total_profit * 100) if comp.total_profit > 0 else 0  # 总利润百分比变动
            },
            "profit_growth": {
                "absolute": (comp2.profit_growth or 0) - (comp.profit_growth or 0),  # 利润增长率差值（因为增长率本身就是百分比，所以不再计算百分比变动）
                "percent": None  # 利润增长率的差值本身就是百分比，不需要再计算百分比变动
            }
        }
    
    return comparison


if __name__ == "__main__":
    # 示例使用
    exp1_dir = "/root/project/agentsociety-ecosim/output/exp_100h_12m_20251120_063248"
    exp2_dir = "/root/project/agentsociety-ecosim/output/suppressed"
    
    result = compare_experiments(
        exp1_dir, exp2_dir,
        experiment1_name="Policy Enabled",
        experiment2_name="Policy Disabled",
        metrics_to_compare=['consumer_metrics', 'firm_metrics', 'innovation_analysis'],
        innovation_types=['labor_productivity_factor', 'price', 'profit_margin']
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
