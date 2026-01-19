"""
涌现行为分析器 - 研究个体行为如何导致宏观模式

核心功能：
1. 微观行为追踪：记录每个智能体的决策和行为
2. 宏观模式识别：检测系统层面的涌现模式
3. 微观-宏观映射：分析个体行为如何聚合形成宏观模式
4. 涌现机制量化：计算涌现强度、临界点等指标
5. 自组织检测：识别自发形成的结构和模式

输出包括：
- 涌现模式报告
- 微观-宏观连接分析
- 相变检测结果
- 自组织结构识别
- 可视化图表
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import networkx as nx

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


@dataclass
class MicroBehavior:
    """微观行为记录"""
    agent_id: str
    agent_type: str  # 'household' or 'firm'
    timestamp: int  # month
    behavior_type: str  # 'consume', 'produce', 'hire', 'invest', etc.
    behavior_data: Dict[str, Any]  # 具体行为数据
    context: Dict[str, Any]  # 行为发生的上下文


@dataclass
class MacroPattern:
    """宏观模式识别结果"""
    pattern_id: str
    pattern_type: str  # 'market_concentration', 'wealth_inequality', 'price_correlation', etc.
    emergence_month: int  # 模式出现的月份
    strength: float  # 模式强度 (0-1)
    stability: float  # 模式稳定性 (0-1)
    micro_contributors: List[str]  # 贡献该模式的个体ID列表
    macro_metrics: Dict[str, float]  # 宏观指标
    description: str  # 模式描述


@dataclass
class EmergenceMetrics:
    """涌现指标"""
    emergence_strength: float  # 涌现强度
    critical_point: Optional[int]  # 临界点（相变发生的月份）
    order_parameter: float  # 序参量
    correlation_length: float  # 关联长度
    self_organization_index: float  # 自组织指数


class EmergentBehaviorAnalyzer:
    """
    涌现行为分析器
    
    分析个体行为如何导致宏观模式，识别涌现现象和自组织行为
    """
    
    def __init__(self, output_dir: str = "output/emergent_behavior"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.micro_behaviors: List[MicroBehavior] = []
        self.macro_patterns: List[MacroPattern] = []
        self.monthly_aggregates: Dict[int, Dict[str, Any]] = {}
        
        # 分析结果
        self.emergence_metrics: Dict[int, EmergenceMetrics] = {}
        self.phase_transitions: List[Dict[str, Any]] = []
        self.self_organizing_structures: List[Dict[str, Any]] = []
        
    def record_micro_behavior(
        self,
        agent_id: str,
        agent_type: str,
        month: int,
        behavior_type: str,
        behavior_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """记录微观行为"""
        behavior = MicroBehavior(
            agent_id=agent_id,
            agent_type=agent_type,
            timestamp=month,
            behavior_type=behavior_type,
            behavior_data=behavior_data,
            context=context or {}
        )
        self.micro_behaviors.append(behavior)
    
    def analyze_emergence(self, current_month: int) -> Dict[str, Any]:
        """
        分析涌现行为
        
        输出结构：
        {
            "emergence_report": {
                "patterns": [...],  # 识别的宏观模式
                "metrics": {...},   # 涌现指标
                "phase_transitions": [...],  # 相变检测
                "self_organization": [...]   # 自组织结构
            },
            "micro_macro_mapping": {
                "pattern_id": {
                    "micro_contributors": [...],
                    "contribution_weights": {...},
                    "emergence_mechanism": "..."
                }
            },
            "visualizations": {
                "pattern_evolution": "path/to/chart.png",
                "phase_diagram": "path/to/chart.png",
                ...
            }
        }
        """
        logger.info(f"🔬 开始分析第 {current_month} 月的涌现行为...")
        
        # 1. 聚合微观行为到宏观层面
        monthly_aggregate = self._aggregate_micro_to_macro(current_month)
        self.monthly_aggregates[current_month] = monthly_aggregate
        
        # 2. 识别宏观模式
        patterns = self._detect_macro_patterns(current_month)
        self.macro_patterns.extend(patterns)
        
        # 3. 计算涌现指标
        metrics = self._calculate_emergence_metrics(current_month)
        self.emergence_metrics[current_month] = metrics
        
        # 4. 检测相变
        phase_transitions = self._detect_phase_transitions(current_month)
        self.phase_transitions.extend(phase_transitions)
        
        # 5. 识别自组织结构
        self_org_structures = self._detect_self_organization(current_month)
        self.self_organizing_structures.extend(self_org_structures)
        
        # 6. 分析微观-宏观映射
        micro_macro_mapping = self._analyze_micro_macro_mapping(patterns)
        
        # 7. 生成可视化
        visualizations = self._generate_visualizations(current_month)
        
        # 8. 生成报告
        report = {
            "month": current_month,
            "emergence_report": {
                "patterns": [self._pattern_to_dict(p) for p in patterns],
                "metrics": self._metrics_to_dict(metrics),
                "phase_transitions": phase_transitions,
                "self_organization": self_org_structures
            },
            "micro_macro_mapping": micro_macro_mapping,
            "visualizations": visualizations,
            "summary": self._generate_summary(patterns, metrics, phase_transitions)
        }
        
        # 保存报告
        self._save_report(report, current_month)
        
        logger.info(f"✅ 涌现行为分析完成，识别了 {len(patterns)} 个宏观模式")
        
        return report
    
    def _aggregate_micro_to_macro(self, month: int) -> Dict[str, Any]:
        """将微观行为聚合到宏观层面"""
        # 筛选当月的微观行为
        month_behaviors = [b for b in self.micro_behaviors if b.timestamp == month]
        
        aggregate = {
            "total_behaviors": len(month_behaviors),
            "behavior_distribution": defaultdict(int),
            "household_behaviors": [],
            "firm_behaviors": [],
            "aggregated_metrics": {}
        }
        
        # 按类型分类
        for behavior in month_behaviors:
            aggregate["behavior_distribution"][behavior.behavior_type] += 1
            
            if behavior.agent_type == "household":
                aggregate["household_behaviors"].append(behavior)
            elif behavior.agent_type == "firm":
                aggregate["firm_behaviors"].append(behavior)
        
        # 计算聚合指标
        # 例如：消费总额、生产总额、平均价格等
        if aggregate["household_behaviors"]:
            total_consumption = sum(
                b.behavior_data.get("amount", 0) 
                for b in aggregate["household_behaviors"]
                if b.behavior_type == "consume"
            )
            aggregate["aggregated_metrics"]["total_consumption"] = total_consumption
        
        if aggregate["firm_behaviors"]:
            total_production = sum(
                b.behavior_data.get("quantity", 0)
                for b in aggregate["firm_behaviors"]
                if b.behavior_type == "produce"
            )
            aggregate["aggregated_metrics"]["total_production"] = total_production
        
        return aggregate
    
    def _detect_macro_patterns(self, month: int) -> List[MacroPattern]:
        """检测宏观模式"""
        patterns = []
        
        # 获取历史数据（至少需要3个月的数据才能检测模式）
        if month < 3:
            return patterns
        
        # 1. 市场集中度模式
        concentration_pattern = self._detect_market_concentration(month)
        if concentration_pattern:
            patterns.append(concentration_pattern)
        
        # 2. 财富不平等模式
        inequality_pattern = self._detect_wealth_inequality(month)
        if inequality_pattern:
            patterns.append(inequality_pattern)
        
        # 3. 价格相关性模式
        price_correlation_pattern = self._detect_price_correlation(month)
        if price_correlation_pattern:
            patterns.append(price_correlation_pattern)
        
        # 4. 消费集群模式
        consumption_cluster_pattern = self._detect_consumption_clusters(month)
        if consumption_cluster_pattern:
            patterns.append(consumption_cluster_pattern)
        
        # 5. 创新扩散模式
        innovation_diffusion_pattern = self._detect_innovation_diffusion(month)
        if innovation_diffusion_pattern:
            patterns.append(innovation_diffusion_pattern)
        
        return patterns
    
    def _detect_market_concentration(self, month: int) -> Optional[MacroPattern]:
        """检测市场集中度模式（寡头垄断、完全竞争等）"""
        # 获取企业市场份额数据
        firm_behaviors = [b for b in self.micro_behaviors 
                         if b.timestamp == month and b.agent_type == "firm"]
        
        if not firm_behaviors:
            return None
        
        # 计算市场份额
        revenues = {}
        for behavior in firm_behaviors:
            if behavior.behavior_type == "sell":
                firm_id = behavior.agent_id
                revenue = behavior.behavior_data.get("revenue", 0)
                revenues[firm_id] = revenues.get(firm_id, 0) + revenue
        
        if not revenues:
            return None
        
        total_revenue = sum(revenues.values())
        if total_revenue == 0:
            return None
        
        # 计算HHI指数（Herfindahl-Hirschman Index）
        market_shares = {firm_id: rev / total_revenue for firm_id, rev in revenues.items()}
        hhi = sum(share ** 2 for share in market_shares.values())
        
        # 判断市场结构
        if hhi > 0.25:  # 高度集中
            pattern_type = "high_market_concentration"
            strength = min(1.0, hhi / 0.5)  # 归一化到0-1
        elif hhi < 0.15:  # 竞争市场
            pattern_type = "competitive_market"
            strength = 1.0 - (hhi / 0.15)
        else:
            return None  # 中等集中度，不算明显模式
        
        # 找出主要贡献者（市场份额最大的几家企业）
        top_firms = sorted(market_shares.items(), key=lambda x: x[1], reverse=True)[:5]
        contributors = [firm_id for firm_id, _ in top_firms]
        
        return MacroPattern(
            pattern_id=f"market_concentration_{month}",
            pattern_type=pattern_type,
            emergence_month=month,
            strength=strength,
            stability=self._calculate_stability("market_concentration", month),
            micro_contributors=contributors,
            macro_metrics={"hhi": hhi, "top_firm_share": top_firms[0][1] if top_firms else 0},
            description=f"市场集中度模式：HHI={hhi:.3f}, {'高度集中' if hhi > 0.25 else '竞争市场'}"
        )
    
    def _detect_wealth_inequality(self, month: int) -> Optional[MacroPattern]:
        """检测财富不平等模式"""
        household_behaviors = [b for b in self.micro_behaviors
                              if b.timestamp == month and b.agent_type == "household"]
        
        if not household_behaviors:
            return None
        
        # 获取财富数据
        wealths = []
        for behavior in household_behaviors:
            if behavior.behavior_type == "wealth_update":
                wealth = behavior.behavior_data.get("wealth", 0)
                wealths.append(wealth)
        
        if len(wealths) < 10:  # 需要足够样本
            return None
        
        # 计算基尼系数
        wealths_sorted = sorted(wealths)
        n = len(wealths_sorted)
        cumsum = np.cumsum(wealths_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1)) * wealths_sorted)) / (n * np.sum(wealths_sorted)) - (n + 1) / n
        
        # 判断是否出现明显的不平等模式
        if gini > 0.4:  # 高度不平等
            pattern_type = "high_wealth_inequality"
            strength = min(1.0, (gini - 0.4) / 0.3)  # 归一化
        elif gini < 0.2:  # 高度平等
            pattern_type = "wealth_equality"
            strength = 1.0 - (gini / 0.2)
        else:
            return None
        
        # 找出极端值（最富和最穷的家庭）
        top_10_percent = int(n * 0.1)
        bottom_10_percent = int(n * 0.1)
        # 这里简化处理，实际需要记录对应的agent_id
        
        return MacroPattern(
            pattern_id=f"wealth_inequality_{month}",
            pattern_type=pattern_type,
            emergence_month=month,
            strength=strength,
            stability=self._calculate_stability("wealth_inequality", month),
            micro_contributors=[],  # 需要从behavior中提取
            macro_metrics={"gini": gini, "top_10_share": 0, "bottom_10_share": 0},
            description=f"财富不平等模式：基尼系数={gini:.3f}"
        )
    
    def _detect_price_correlation(self, month: int) -> Optional[MacroPattern]:
        """检测价格相关性模式（价格联动、价格泡沫等）"""
        # 获取价格数据
        prices_by_product = defaultdict(list)
        
        for behavior in self.micro_behaviors:
            if behavior.timestamp == month and behavior.behavior_type == "price_change":
                product_id = behavior.behavior_data.get("product_id")
                price = behavior.behavior_data.get("price")
                if product_id and price:
                    prices_by_product[product_id].append(price)
        
        if len(prices_by_product) < 5:
            return None
        
        # 计算价格变化的相关性矩阵
        price_changes = {}
        for product_id, prices in prices_by_product.items():
            if len(prices) > 1:
                changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                price_changes[product_id] = np.mean(changes) if changes else 0
        
        if len(price_changes) < 5:
            return None
        
        # 计算平均相关性
        changes_list = list(price_changes.values())
        if len(changes_list) < 2:
            return None
        
        # 简化的相关性检测：如果大部分价格同向变化，说明有相关性
        positive_changes = sum(1 for c in changes_list if c > 0)
        negative_changes = sum(1 for c in changes_list if c < 0)
        correlation_strength = max(positive_changes, negative_changes) / len(changes_list)
        
        if correlation_strength > 0.7:  # 70%以上同向变化
            pattern_type = "price_correlation"
            strength = correlation_strength
            
            return MacroPattern(
                pattern_id=f"price_correlation_{month}",
                pattern_type=pattern_type,
                emergence_month=month,
                strength=strength,
                stability=self._calculate_stability("price_correlation", month),
                micro_contributors=list(price_changes.keys())[:10],
                macro_metrics={"correlation_strength": correlation_strength},
                description=f"价格相关性模式：{correlation_strength:.1%}的商品价格同向变化"
            )
        
        return None
    
    def _detect_consumption_clusters(self, month: int) -> Optional[MacroPattern]:
        """检测消费集群模式（消费偏好分组）"""
        # 获取消费数据
        consumption_vectors = {}
        
        for behavior in self.micro_behaviors:
            if behavior.timestamp == month and behavior.agent_type == "household":
                if behavior.behavior_type == "consume":
                    household_id = behavior.agent_id
                    category = behavior.behavior_data.get("category")
                    amount = behavior.behavior_data.get("amount", 0)
                    
                    if household_id not in consumption_vectors:
                        consumption_vectors[household_id] = defaultdict(float)
                    consumption_vectors[household_id][category] += amount
        
        if len(consumption_vectors) < 10:
            return None
        
        # 使用聚类分析识别消费模式
        # 简化实现：使用PCA降维后聚类
        categories = set()
        for vec in consumption_vectors.values():
            categories.update(vec.keys())
        categories = sorted(list(categories))
        
        if len(categories) < 3:
            return None
        
        # 构建特征矩阵
        X = []
        household_ids = []
        for hh_id, vec in consumption_vectors.items():
            row = [vec.get(cat, 0) for cat in categories]
            X.append(row)
            household_ids.append(hh_id)
        
        X = np.array(X)
        
        # 标准化
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # 聚类
        clustering = DBSCAN(eps=0.5, min_samples=3)
        labels = clustering.fit_predict(X_normalized)
        
        # 检查是否有明显的集群
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 移除噪声点
        
        if len(unique_labels) >= 2:  # 至少2个集群
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            max_cluster_size = max(cluster_sizes)
            strength = max_cluster_size / len(consumption_vectors)
            
            if strength > 0.3:  # 最大集群占比超过30%
                return MacroPattern(
                    pattern_id=f"consumption_clusters_{month}",
                    pattern_type="consumption_clustering",
                    emergence_month=month,
                    strength=strength,
                    stability=self._calculate_stability("consumption_clusters", month),
                    micro_contributors=household_ids[:20],
                    macro_metrics={
                        "num_clusters": len(unique_labels),
                        "max_cluster_size": max_cluster_size,
                        "cluster_sizes": cluster_sizes
                    },
                    description=f"消费集群模式：识别出{len(unique_labels)}个消费偏好集群"
                )
        
        return None
    
    def _detect_innovation_diffusion(self, month: int) -> Optional[MacroPattern]:
        """检测创新扩散模式（创新如何传播）"""
        # 获取创新事件
        innovation_events = []
        for behavior in self.micro_behaviors:
            if behavior.timestamp == month and behavior.behavior_type == "innovate":
                innovation_events.append(behavior)
        
        if len(innovation_events) < 3:
            return None
        
        # 分析创新的空间/时间分布
        # 简化实现：检查创新是否集中在某些企业
        innovating_firms = [b.agent_id for b in innovation_events]
        firm_counts = defaultdict(int)
        for firm_id in innovating_firms:
            firm_counts[firm_id] += 1
        
        # 如果创新集中在少数企业，说明有扩散模式
        if len(firm_counts) < len(innovating_firms) * 0.5:  # 少于50%的企业有创新
            concentration = len(firm_counts) / len(innovating_firms) if innovating_firms else 0
            strength = 1.0 - concentration
            
            return MacroPattern(
                pattern_id=f"innovation_diffusion_{month}",
                pattern_type="innovation_clustering",
                emergence_month=month,
                strength=strength,
                stability=self._calculate_stability("innovation_diffusion", month),
                micro_contributors=list(firm_counts.keys()),
                macro_metrics={
                    "num_innovating_firms": len(firm_counts),
                    "total_innovations": len(innovation_events),
                    "concentration": concentration
                },
                description=f"创新扩散模式：创新集中在{len(firm_counts)}家企业"
            )
        
        return None
    
    def _calculate_stability(self, pattern_type: str, month: int, window: int = 3) -> float:
        """计算模式稳定性（基于历史数据）"""
        if month < window:
            return 0.5  # 数据不足，返回中等稳定性
        
        # 检查前几个月是否也有类似模式
        recent_patterns = [p for p in self.macro_patterns 
                          if p.pattern_type == pattern_type 
                          and month - window <= p.emergence_month < month]
        
        if not recent_patterns:
            return 0.3  # 新出现的模式，稳定性较低
        
        # 稳定性 = 连续出现的月份数 / 窗口大小
        stability = len(recent_patterns) / window
        return min(1.0, stability)
    
    def _calculate_emergence_metrics(self, month: int) -> EmergenceMetrics:
        """计算涌现指标"""
        # 1. 涌现强度：基于模式数量和强度
        recent_patterns = [p for p in self.macro_patterns if p.emergence_month == month]
        if recent_patterns:
            emergence_strength = np.mean([p.strength for p in recent_patterns])
        else:
            emergence_strength = 0.0
        
        # 2. 临界点检测（相变）
        critical_point = self._detect_critical_point(month)
        
        # 3. 序参量：系统有序程度的度量
        order_parameter = self._calculate_order_parameter(month)
        
        # 4. 关联长度：系统各部分的相关性范围
        correlation_length = self._calculate_correlation_length(month)
        
        # 5. 自组织指数
        self_org_index = self._calculate_self_organization_index(month)
        
        return EmergenceMetrics(
            emergence_strength=emergence_strength,
            critical_point=critical_point,
            order_parameter=order_parameter,
            correlation_length=correlation_length,
            self_organization_index=self_org_index
        )
    
    def _detect_critical_point(self, month: int, window: int = 5) -> Optional[int]:
        """检测临界点（相变发生的月份）"""
        if month < window * 2:
            return None
        
        # 检查是否有指标发生突变
        # 例如：基尼系数、市场集中度等的突然变化
        
        # 简化实现：检查模式数量的突变
        pattern_counts = []
        for m in range(max(1, month - window), month + 1):
            count = len([p for p in self.macro_patterns if p.emergence_month == m])
            pattern_counts.append(count)
        
        if len(pattern_counts) < window:
            return None
        
        # 检测突变点
        for i in range(1, len(pattern_counts)):
            if pattern_counts[i] > pattern_counts[i-1] * 2:  # 模式数量翻倍
                return month - window + i
        
        return None
    
    def _calculate_order_parameter(self, month: int) -> float:
        """计算序参量（系统有序程度）"""
        # 序参量可以通过多种方式计算
        # 例如：市场集中度、价格相关性、消费集群度等
        
        recent_patterns = [p for p in self.macro_patterns if p.emergence_month == month]
        if not recent_patterns:
            return 0.0
        
        # 简化：使用模式强度的加权平均
        order_parameter = np.mean([p.strength * p.stability for p in recent_patterns])
        return order_parameter
    
    def _calculate_correlation_length(self, month: int) -> float:
        """计算关联长度"""
        # 关联长度：系统各部分的相关性范围
        # 可以通过分析智能体之间的相关性来计算
        
        # 简化实现：基于行为的相关性
        behaviors = [b for b in self.micro_behaviors if b.timestamp == month]
        if len(behaviors) < 10:
            return 0.0
        
        # 计算行为类型的多样性
        behavior_types = set(b.behavior_type for b in behaviors)
        diversity = len(behavior_types) / max(1, len(behaviors))
        
        # 关联长度与多样性成反比
        correlation_length = 1.0 - diversity
        return max(0.0, min(1.0, correlation_length))
    
    def _calculate_self_organization_index(self, month: int) -> float:
        """计算自组织指数"""
        # 自组织指数：系统自发形成结构的程度
        # 可以通过分析结构的复杂性和有序性来计算
        
        # 简化实现：基于模式数量和稳定性
        recent_patterns = [p for p in self.macro_patterns if p.emergence_month == month]
        if not recent_patterns:
            return 0.0
        
        # 自组织指数 = 模式数量 × 平均稳定性 × 平均强度
        avg_stability = np.mean([p.stability for p in recent_patterns])
        avg_strength = np.mean([p.strength for p in recent_patterns])
        num_patterns = len(recent_patterns)
        
        # 归一化
        self_org_index = (num_patterns / 10.0) * avg_stability * avg_strength
        return min(1.0, self_org_index)
    
    def _detect_phase_transitions(self, month: int) -> List[Dict[str, Any]]:
        """检测相变"""
        transitions = []
        
        # 检查是否有临界点
        critical_point = self._detect_critical_point(month)
        if critical_point and critical_point == month:
            transitions.append({
                "type": "critical_transition",
                "month": month,
                "description": "检测到系统临界点，可能发生相变",
                "indicators": {
                    "order_parameter_change": 0,  # 需要计算
                    "pattern_count_change": 0
                }
            })
        
        return transitions
    
    def _detect_self_organization(self, month: int) -> List[Dict[str, Any]]:
        """检测自组织结构"""
        structures = []
        
        # 检测消费集群
        cluster_patterns = [p for p in self.macro_patterns 
                           if p.pattern_type == "consumption_clustering" 
                           and p.emergence_month == month]
        
        for pattern in cluster_patterns:
            structures.append({
                "type": "consumption_cluster",
                "month": month,
                "pattern_id": pattern.pattern_id,
                "description": "自发形成的消费偏好集群",
                "metrics": pattern.macro_metrics
            })
        
        # 检测市场结构
        market_patterns = [p for p in self.macro_patterns
                          if "market" in p.pattern_type
                          and p.emergence_month == month]
        
        for pattern in market_patterns:
            structures.append({
                "type": "market_structure",
                "month": month,
                "pattern_id": pattern.pattern_id,
                "description": "自发形成的市场结构",
                "metrics": pattern.macro_metrics
            })
        
        return structures
    
    def _analyze_micro_macro_mapping(self, patterns: List[MacroPattern]) -> Dict[str, Dict[str, Any]]:
        """分析微观-宏观映射关系"""
        mapping = {}
        
        for pattern in patterns:
            # 分析每个模式由哪些个体行为贡献
            contributors = pattern.micro_contributors
            
            # 计算贡献权重
            contribution_weights = {}
            if contributors:
                # 简化：平均分配权重
                weight = 1.0 / len(contributors)
                for contributor in contributors:
                    contribution_weights[contributor] = weight
            
            # 分析涌现机制
            mechanism = self._identify_emergence_mechanism(pattern)
            
            mapping[pattern.pattern_id] = {
                "micro_contributors": contributors,
                "contribution_weights": contribution_weights,
                "emergence_mechanism": mechanism,
                "contribution_analysis": self._analyze_contributions(pattern)
            }
        
        return mapping
    
    def _identify_emergence_mechanism(self, pattern: MacroPattern) -> str:
        """识别涌现机制"""
        # 根据模式类型和特征识别机制
        
        if "concentration" in pattern.pattern_type:
            return "正反馈机制：成功者获得更多资源，导致市场集中"
        elif "inequality" in pattern.pattern_type:
            return "累积优势：初始差异通过复利效应放大"
        elif "correlation" in pattern.pattern_type:
            return "信息传播：价格信息通过市场网络传播"
        elif "cluster" in pattern.pattern_type:
            return "同质性偏好：相似个体形成集群"
        elif "diffusion" in pattern.pattern_type:
            return "网络效应：创新通过社会网络扩散"
        else:
            return "复杂交互：多个因素共同作用"
    
    def _analyze_contributions(self, pattern: MacroPattern) -> Dict[str, Any]:
        """分析个体贡献"""
        # 分析哪些个体对模式形成贡献最大
        
        contributors = pattern.micro_contributors
        if not contributors:
            return {"top_contributors": [], "contribution_distribution": "uniform"}
        
        # 简化分析
        return {
            "top_contributors": contributors[:5],
            "contribution_distribution": "power_law" if len(contributors) > 10 else "uniform",
            "contribution_inequality": 0.5  # 需要实际计算
        }
    
    def _generate_visualizations(self, month: int) -> Dict[str, str]:
        """生成可视化图表"""
        visualizations = {}
        
        # 1. 模式演化图
        pattern_evolution_path = self._plot_pattern_evolution(month)
        visualizations["pattern_evolution"] = pattern_evolution_path
        
        # 2. 相图
        phase_diagram_path = self._plot_phase_diagram(month)
        visualizations["phase_diagram"] = phase_diagram_path
        
        # 3. 微观-宏观映射图
        micro_macro_path = self._plot_micro_macro_mapping(month)
        visualizations["micro_macro_mapping"] = micro_macro_path
        
        return visualizations
    
    def _plot_pattern_evolution(self, month: int) -> str:
        """绘制模式演化图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 收集历史数据
        months = sorted(set(p.emergence_month for p in self.macro_patterns))
        pattern_counts = [len([p for p in self.macro_patterns if p.emergence_month == m]) 
                         for m in months]
        
        ax.plot(months, pattern_counts, marker='o', linewidth=2)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Patterns')
        ax.set_title('Macro Pattern Evolution Over Time')
        ax.grid(True, alpha=0.3)
        
        path = f"{self.output_dir}/pattern_evolution_month_{month}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def _plot_phase_diagram(self, month: int) -> str:
        """绘制相图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 收集数据
        months = sorted(self.emergence_metrics.keys())
        order_params = [self.emergence_metrics[m].order_parameter for m in months]
        self_org_indices = [self.emergence_metrics[m].self_organization_index for m in months]
        
        scatter = ax.scatter(order_params, self_org_indices, c=months, cmap='viridis', s=100)
        ax.set_xlabel('Order Parameter')
        ax.set_ylabel('Self-Organization Index')
        ax.set_title('Phase Diagram: System State Evolution')
        plt.colorbar(scatter, label='Month')
        ax.grid(True, alpha=0.3)
        
        path = f"{self.output_dir}/phase_diagram_month_{month}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def _plot_micro_macro_mapping(self, month: int) -> str:
        """绘制微观-宏观映射图"""
        # 简化实现：展示模式与贡献者的关系
        fig, ax = plt.subplots(figsize=(12, 8))
        
        recent_patterns = [p for p in self.macro_patterns if p.emergence_month == month]
        
        if not recent_patterns:
            ax.text(0.5, 0.5, 'No patterns detected', ha='center', va='center')
        else:
            # 绘制模式-贡献者网络
            # 简化：条形图显示每个模式的贡献者数量
            pattern_names = [p.pattern_id[:20] for p in recent_patterns]
            contributor_counts = [len(p.micro_contributors) for p in recent_patterns]
            
            ax.barh(pattern_names, contributor_counts)
            ax.set_xlabel('Number of Micro Contributors')
            ax.set_title('Micro-Macro Mapping: Contributors per Pattern')
        
        path = f"{self.output_dir}/micro_macro_mapping_month_{month}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def _generate_summary(self, patterns: List[MacroPattern], 
                         metrics: EmergenceMetrics,
                         phase_transitions: List[Dict[str, Any]]) -> str:
        """生成摘要"""
        summary = f"""
涌现行为分析摘要（第 {metrics.emergence_strength:.2%} 涌现强度）

识别的宏观模式：{len(patterns)} 个
- {'; '.join([p.pattern_type for p in patterns[:3]])}

涌现指标：
- 序参量：{metrics.order_parameter:.3f}
- 自组织指数：{metrics.self_organization_index:.3f}
- 关联长度：{metrics.correlation_length:.3f}
- 临界点：{'是' if metrics.critical_point else '否'}

相变检测：{len(phase_transitions)} 个
"""
        return summary.strip()
    
    def _pattern_to_dict(self, pattern: MacroPattern) -> Dict[str, Any]:
        """将模式对象转换为字典"""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "emergence_month": pattern.emergence_month,
            "strength": pattern.strength,
            "stability": pattern.stability,
            "micro_contributors": pattern.micro_contributors,
            "macro_metrics": pattern.macro_metrics,
            "description": pattern.description
        }
    
    def _metrics_to_dict(self, metrics: EmergenceMetrics) -> Dict[str, Any]:
        """将指标对象转换为字典"""
        return {
            "emergence_strength": metrics.emergence_strength,
            "critical_point": metrics.critical_point,
            "order_parameter": metrics.order_parameter,
            "correlation_length": metrics.correlation_length,
            "self_organization_index": metrics.self_organization_index
        }
    
    def _save_report(self, report: Dict[str, Any], month: int):
        """保存报告"""
        report_path = f"{self.output_dir}/emergence_report_month_{month}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 涌现行为报告已保存: {report_path}")


# 使用示例
if __name__ == "__main__":
    analyzer = EmergentBehaviorAnalyzer()
    
    # 记录一些示例行为
    analyzer.record_micro_behavior(
        agent_id="household_1",
        agent_type="household",
        month=1,
        behavior_type="consume",
        behavior_data={"category": "food", "amount": 100}
    )
    
    # 分析涌现行为
    report = analyzer.analyze_emergence(month=1)
    print(json.dumps(report, indent=2, ensure_ascii=False))

