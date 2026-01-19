#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异质性专项分析工具
只运行异质性计算部分，跳过耗时的预算一致性和商品画像一致性评估
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import math
from collections import defaultdict, Counter
import sys
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeterogeneityOnlyAnalyzer:
    """专门用于异质性分析的类"""
    
    def __init__(self):
        """初始化分析器"""
        self.output_dir = "onsumer_modeling/output"
        self.family_data_path = "consumer_modeling/household_data/PSID/extracted_data/processed_data/integrated_psid_families_data.json"
        
        # 大类支出映射
        self.category_keys = [
            "food_expenditure",    
            "clothing_expenditure",   
            "education_expenditure",   
            "childcare_expenditure",   
            "electronics_expenditure",     
            "home_furnishing_equipment",     
            "other_recreation_expenditure",   
            "housing_expenditure",   
            "utilities_expenditure",   
            "transportation_expenditure",   
            "healthcare_expenditure",   
            "travel_expenditure",   
            "phone_internet_expenditure"
        ]
        
        self.category_names = {
            "food_expenditure": "食品总支出",
            "clothing_expenditure": "服装支出",
            "education_expenditure": "教育支出",
            "childcare_expenditure": "儿童保育支出",
            "electronics_expenditure": "电子产品支出",
            "other_recreation_expenditure": "其他娱乐支出",
            "housing_expenditure": "住房总支出",
            "utilities_expenditure": "公用事业总支出",
            "transportation_expenditure": "交通总支出",
            "healthcare_expenditure": "医疗保健支出",
            "travel_expenditure": "旅行支出",
            "phone_internet_expenditure": "电话/互联网支出"
        }
        
        # 加载数据
        self.family_history_data = self._load_family_history_data()
        self.family_output_data = self._load_family_output_data()
        
    def _load_family_history_data(self) -> Dict:
        """加载家庭历史数据"""
        try:
            with open(self.family_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 提取families部分的数据
            families_data = raw_data.get('families', {})
            logger.info(f"成功加载{len(families_data)}个家庭的历史数据")
            return families_data
        except Exception as e:
            logger.error(f"加载家庭历史数据失败: {e}")
            return {}
    
    def _load_family_output_data(self) -> Dict:
        """加载LLM输出的家庭分配数据"""
        output_data = {}
        
        if not os.path.exists(self.output_dir):
            logger.warning(f"输出目录不存在: {self.output_dir}")
            return output_data
            
        for folder_name in os.listdir(self.output_dir):
            if folder_name.startswith('family'):
                family_id = folder_name.replace('family', '')
                family_path = os.path.join(self.output_dir, folder_name)
                
                if not os.path.isdir(family_path):
                    continue
                
                try:
                    # 加载年度预算分配
                    annual_file = os.path.join(family_path, f'annual_budget_allocation_family_{family_id}.json')
                    annual_allocation = {}
                    if os.path.exists(annual_file):
                        with open(annual_file, 'r', encoding='utf-8') as f:
                            annual_data = json.load(f)
                            # 提取category_allocations部分
                            annual_allocation = annual_data.get('category_allocations', {})
                    
                    # 加载月度购物计划
                    shopping_file = os.path.join(family_path, f'monthly_shopping_plan_family_{family_id}.json')
                    shopping_plan = {}
                    if os.path.exists(shopping_file):
                        with open(shopping_file, 'r', encoding='utf-8') as f:
                            shopping_plan = json.load(f)
                    
                    output_data[family_id] = {
                        'annual_allocation': annual_allocation,
                        'shopping_plan': shopping_plan
                    }
                    
                except Exception as e:
                    logger.error(f"加载家庭{family_id}数据失败: {e}")
                    continue
        
        logger.info(f"成功加载{len(output_data)}个家庭的输出数据")
        return output_data
    
    def calculate_product_diversity_score(self, family_id: str) -> Dict[str, float]:
        """
        计算商品多样性得分，包含类别多样性、价格区间覆盖度、数量多样性
        """
        if family_id not in self.family_output_data:
            return {'category_diversity': 0.0, 'price_coverage': 0.0, 'quantity_diversity': 0.0, 'overall_diversity': 0.0}
        
        shopping_plan = self.family_output_data[family_id].get('shopping_plan', {})
        all_products = self._extract_representative_products(shopping_plan, max_products=1000)  # 获取所有商品
        
        if not all_products:
            return {'category_diversity': 0.0, 'price_coverage': 0.0, 'quantity_diversity': 0.0, 'overall_diversity': 0.0}
        
        # 1. 类别多样性 - 香农多样性指数
        category_counts = defaultdict(int)
        for month, category_data in shopping_plan.items():
            if not isinstance(category_data, dict):
                continue
            for category, subcat_data in category_data.items():
                if not isinstance(subcat_data, dict):
                    continue
                for subcat, product_list in subcat_data.items():
                    if isinstance(product_list, list) and len(product_list) > 0:
                        category_counts[subcat] += len(product_list)
        
        total_products = sum(category_counts.values())
        if total_products == 0:
            shannon_index = 0.0
        else:
            shannon_index = -sum((count/total_products) * math.log(count/total_products) 
                               for count in category_counts.values() if count > 0)
        
        # 归一化香农指数 (最大值为log(类别数))
        max_shannon = math.log(len(category_counts)) if len(category_counts) > 1 else 1
        category_diversity = shannon_index / max_shannon if max_shannon > 0 else 0
        
        # 2. 价格区间覆盖度
        prices = [product['price'] for product in all_products if product['price'] > 0]
        if not prices:
            price_coverage = 0.0
        else:
            # 使用对数价格区间以更好地反映实际购买模式
            log_prices = [math.log10(max(price, 0.01)) for price in prices]
            min_log_price = min(log_prices)
            max_log_price = max(log_prices)
            
            if max_log_price > min_log_price:
                # 使用10个价格区间获得更好的分辨率
                num_bins = 10
                bin_width = (max_log_price - min_log_price) / num_bins
                covered_bins = set()
                
                for log_price in log_prices:
                    bin_index = min(num_bins-1, int((log_price - min_log_price) / bin_width))
                    covered_bins.add(bin_index)
                
                # 基础覆盖度
                base_coverage = len(covered_bins) / num_bins
                
                # 添加价格范围奖励：价格跨度越大，得分越高
                price_range_ratio = (max(prices) - min(prices)) / max(prices) if max(prices) > 0 else 0
                range_bonus = min(0.3, price_range_ratio * 0.5)  # 最多30%奖励
                
                price_coverage = min(1.0, base_coverage + range_bonus)
            else:
                # 所有商品同价时，根据商品数量给予基础分
                price_coverage = min(0.4, 0.1 + len(prices) * 0.02)
        
        # 3. 数量多样性
        quantities = [product['quantity'] for product in all_products if product['quantity'] > 0]
        if len(quantities) <= 1:
            quantity_diversity = 0.0
        else:
            # 基础变异系数
            quantity_mean = np.mean(quantities)
            quantity_std = np.std(quantities)
            coefficient_of_variation = quantity_std / quantity_mean if quantity_mean > 0 else 0
            
            # 数量范围多样性：考虑数量分布的跨度
            min_qty = min(quantities)
            max_qty = max(quantities)
            range_diversity = (max_qty - min_qty) / max_qty if max_qty > 0 else 0
            
            # 数量分布熵：考虑不同数量值的分布均匀性
            from collections import Counter
            qty_counts = Counter(quantities)
            total_items = len(quantities)
            qty_entropy = -sum((count/total_items) * math.log(count/total_items) 
                              for count in qty_counts.values() if count > 0)
            max_entropy = math.log(len(qty_counts)) if len(qty_counts) > 1 else 1
            normalized_entropy = qty_entropy / max_entropy if max_entropy > 0 else 0
            
            # 综合数量多样性：变异系数(40%) + 范围多样性(30%) + 分布熵(30%)
            quantity_diversity = (
                min(1.0, coefficient_of_variation) * 0.4 + 
                min(1.0, range_diversity) * 0.3 + 
                normalized_entropy * 0.3
            )
        
        # 综合多样性得分 (加权平均)
        overall_diversity = (category_diversity * 0.5 + price_coverage * 0.3 + quantity_diversity * 0.2)
        
        return {
            'category_diversity': category_diversity,
            'price_coverage': price_coverage,
            'quantity_diversity': quantity_diversity,
            'overall_diversity': overall_diversity
        }
    
    def _extract_representative_products(self, shopping_plan: Dict, max_products: int = 20) -> List[Dict]:
        """从购物计划中提取代表性商品"""
        all_products = {}
        
        # 遍历所有月份
        for month, category_data in shopping_plan.items():
            if not isinstance(category_data, dict):
                continue
                
            # 遍历所有类别
            for category, subcat_data in category_data.items():
                if not isinstance(subcat_data, dict):
                    continue
                    
                # 遍历所有子类别
                for subcat, product_list in subcat_data.items():
                    if not isinstance(product_list, list):
                        continue
                        
                    for product in product_list:
                        if isinstance(product, dict) and 'name' in product:
                            product_name = product['name']
                            quantity = product.get('quantity', 1)
                            price = product.get('price', 0)
                            
                            if product_name in all_products:
                                all_products[product_name]['quantity'] += quantity
                            else:
                                all_products[product_name] = {
                                    'name': product_name,
                                    'quantity': quantity,
                                    'price': price
                                }
        
        # 按总购买数量排序，取前N个
        product_list = list(all_products.values())
        product_list.sort(key=lambda x: x['quantity'], reverse=True)
        
        return product_list[:max_products]
    
    def calculate_family_heterogeneity_score(self) -> Dict[str, float]:
        """
        计算不同家庭间的异质性得分
        使用改进的方法：基于画像-行为一致性的异质性评估
        """
        if len(self.family_output_data) < 2:
            logger.warning("家庭数量不足，无法计算异质性")
            return {
                'comprehensive_heterogeneity': 0.0,
                'detailed_analysis': {}
            }
        
        # 导入改进的异质性分析器
        try:
            from improved_heterogeneity import ImprovedHeterogeneityAnalyzer
            improved_analyzer = ImprovedHeterogeneityAnalyzer(self)
            
            # 使用新的异质性分析方法
            comprehensive_report = improved_analyzer.generate_comprehensive_heterogeneity_report()
            
            result = {
                'comprehensive_heterogeneity': comprehensive_report.get('综合异质性得分', 0.0),
                'detailed_analysis': comprehensive_report
            }
            
            logger.info(f"综合异质性得分: {result['comprehensive_heterogeneity']:.3f}")
            
            return result
            
        except ImportError as e:
            logger.error(f"无法导入改进的异质性分析器: {e}")
            return {'comprehensive_heterogeneity': 0.0, 'detailed_analysis': {}}
        except Exception as e:
            logger.error(f"异质性分析失败: {e}")
            return {'comprehensive_heterogeneity': 0.0, 'detailed_analysis': {}}
    
    def _calculate_average_product_price(self, shopping_plan: Dict) -> float:
        """计算购物计划中的平均商品价格"""
        total_price = 0
        product_count = 0
        
        for month, category_data in shopping_plan.items():
            if not isinstance(category_data, dict):
                continue
            for category, subcat_data in category_data.items():
                if not isinstance(subcat_data, dict):
                    continue
                for subcat, product_list in subcat_data.items():
                    if not isinstance(product_list, list):
                        continue
                    for product in product_list:
                        if isinstance(product, dict) and 'price' in product:
                            price = product['price']
                            if price > 0:
                                total_price += price
                                product_count += 1
        
        return total_price / product_count if product_count > 0 else 0
    
    def _calculate_total_purchase_quantity(self, shopping_plan: Dict) -> int:
        """计算购物计划中的总购买数量"""
        total_quantity = 0
        
        for month, category_data in shopping_plan.items():
            if not isinstance(category_data, dict):
                continue
            for category, subcat_data in category_data.items():
                if not isinstance(subcat_data, dict):
                    continue
                for subcat, product_list in subcat_data.items():
                    if not isinstance(product_list, list):
                        continue
                    for product in product_list:
                        if isinstance(product, dict) and 'quantity' in product:
                            quantity = product['quantity']
                            total_quantity += quantity
        
        return total_quantity
    
    def generate_heterogeneity_only_report(self, output_file_path: str = None) -> Dict[str, Any]:
        """生成仅包含异质性分析的报告"""
        logger.info("开始异质性专项分析...")
        start_time = time.time()
        
        if not output_file_path:
            # 使用固定的文件名，保存在assessment目录下
            output_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),  # assessment目录
                f"heterogeneity_only_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # 计算家庭异质性
        logger.info("计算家庭异质性...")
        heterogeneity_results = self.calculate_family_heterogeneity_score()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 构建报告
        report = {
            '异质性分析摘要': {
                '分析日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '家庭总数': len(self.family_output_data),
                '分析耗时(秒)': round(total_time, 2),
                '综合异质性得分': round(heterogeneity_results.get('comprehensive_heterogeneity', 0), 3),
                '异质性指标': {
                    '商品选择异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('商品异质性', {}).get('异质性得分', 0), 3),
                    '预算分配异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('预算异质性', {}).get('异质性得分', 0), 3),
                    '购买模式异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('模式异质性', {}).get('异质性得分', 0), 3),
                    '聚类质量得分': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('综合指标', {}).get('聚类质量', 0), 3)
                }
            },
            '异质性详细分析': heterogeneity_results.get('detailed_analysis', {}),
            '分析说明': {
                '综合异质性得分': '基于画像-行为一致性、消费模式正确性和家庭差异化的综合异质性评分',
                '异质性指标说明': {
                    '商品选择异质性': '基于TF-IDF商品类别向量的组内/组间相似度差异，衡量相似家庭是否选择相似商品',
                    '预算分配异质性': '基于13类支出比例向量的组内/组间相似度差异，衡量相似家庭是否有相似预算分配',
                    '购买模式异质性': '基于价格偏好、购买数量等行为特征的组内/组间相似度差异',
                    '聚类质量得分': '家庭画像聚类的轮廓系数，反映聚类的合理性和区分度'
                }
            }
        }
        
        # 保存报告
        try:
            # 转换NumPy类型为Python原生类型以确保JSON序列化兼容
            def convert_numpy_types(obj):
                """递归转换NumPy类型为Python原生类型"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            # 转换报告中的NumPy类型
            serializable_report = convert_numpy_types(report)
            
            # 保存报告
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, ensure_ascii=False, indent=2)
            logger.info(f"异质性分析报告已保存到: {output_file_path}")
            
        except Exception as e:
            logger.error(f"保存异质性分析报告失败: {e}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("异质性专项分析报告摘要")
        print("="*80)
        print(f"分析日期: {report['异质性分析摘要']['分析日期']}")
        print(f"家庭总数: {report['异质性分析摘要']['家庭总数']}")
        print(f"分析耗时: {report['异质性分析摘要']['分析耗时(秒)']}秒")
        print(f"综合异质性得分: {report['异质性分析摘要']['综合异质性得分']}")
        
        # 显示异质性指标
        het_metrics = report['异质性分析摘要'].get('异质性指标', {})
        if het_metrics:
            print("\n--- 异质性指标 (组内相似 vs 组间差异) ---")
            print(f"商品选择异质性: {het_metrics.get('商品选择异质性', 0)}")
            print(f"预算分配异质性: {het_metrics.get('预算分配异质性', 0)}")
            print(f"购买模式异质性: {het_metrics.get('购买模式异质性', 0)}")
            print(f"聚类质量得分: {het_metrics.get('聚类质量得分', 0)}")
            
            # 显示聚类信息
            detailed_analysis = report.get('异质性详细分析', {})
            new_analysis = detailed_analysis.get('新异质性分析', {})
            comprehensive_metrics = new_analysis.get('综合指标', {})
            
            if comprehensive_metrics.get('聚类名称'):
                print(f"\n--- 家庭聚类信息 ---")
                cluster_names = comprehensive_metrics['聚类名称']
                cluster_sizes = comprehensive_metrics.get('聚类规模', {})
                print(f"识别出{len(cluster_names)}个家庭类型:")
                for i, name in enumerate(cluster_names):
                    size = cluster_sizes.get(name, 0)
                    print(f"  {i+1}. {name}: {size}个家庭")
        
        print("="*80)
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='异质性专项分析工具')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径')
    
    args = parser.parse_args()
    
    print("启动异质性专项分析工具...")
    print("注意：此工具跳过耗时的预算一致性和商品画像一致性评估，专注于异质性计算")
    print("-" * 60)
    
    # 创建异质性分析器实例
    analyzer = HeterogeneityOnlyAnalyzer()
    
    # 生成异质性分析报告
    report = analyzer.generate_heterogeneity_only_report(args.output)
    
    return report


if __name__ == "__main__":
    main()
