#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消费者模型评估系统
用于评估LLM分配预算和商品选择的准确性、多样性和异质性
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
import concurrent.futures
import threading
from functools import partial
import time

# 添加父目录到路径以导入llm_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入LLM工具函数，如果失败则直接报错停止运行
from llm_utils import call_llm_chat_completion

# 线程安全的LLM调用函数
_llm_lock = threading.Lock()

def safe_llm_call(prompt, system_content, max_retries=3, timeout_seconds=30):
    """线程安全的LLM调用函数，带重试机制（不使用signal，因为在多线程中不可用）"""
    import threading
    
    for attempt in range(max_retries):
        try:
            with _llm_lock:
                time.sleep(0.02)  # 减少延迟时间
                
                # 使用线程来实现超时控制，而不是signal
                result_container = [None]
                exception_container = [None]
                
                def llm_call_thread():
                    try:
                        result_container[0] = call_llm_chat_completion(prompt, system_content)
                    except Exception as e:
                        exception_container[0] = e
                
                # 启动LLM调用线程
                thread = threading.Thread(target=llm_call_thread)
                thread.daemon = True
                thread.start()
                
                # 等待结果或超时
                thread.join(timeout=timeout_seconds)
                
                if thread.is_alive():
                    # 线程仍在运行，说明超时了
                    logger.warning(f"LLM调用超时 (尝试 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"LLM调用最终超时")
                        return "75"  # 返回默认评分
                
                # 检查是否有异常
                if exception_container[0]:
                    raise exception_container[0]
                
                # 返回结果
                if result_container[0] is not None:
                    return result_container[0]
                else:
                    raise Exception("LLM调用返回了None")
                
        except Exception as e:
            logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                logger.error(f"LLM调用最终失败: {e}")
                return "75"  # 返回默认评分

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsumerEvaluationSystem:
    """消费者模型评估系统主类"""
    
    def __init__(self, max_workers=4, batch_size=10):
        """初始化评估系统"""
        self.output_dir = "consumer_modeling/output"
        self.family_data_path = "consumer_modeling/household_data/PSID/extracted_data/processed_data/integrated_psid_families_data.json"
        self.max_workers = max_workers  # 最大并发线程数
        self.batch_size = batch_size  # 每批处理的家庭数量
        
        # 暂存文件路径
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_results")
        self.progress_file = os.path.join(self.temp_dir, "evaluation_progress.json")
        self.results_cache_file = os.path.join(self.temp_dir, "family_results_cache.json")
        
        # 确保暂存目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
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
    
    def calculate_budget_prediction_metrics(self, family_id: str) -> Dict[str, float]:
        """
        计算预算预测的多个指标，包括MAPE、结构相似度等
        基于前5年数据预测2021年，与实际2021年数据对比
        """
        if family_id not in self.family_output_data:
            logger.warning(f"家庭{family_id}缺少输出数据")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        # 如果没有历史数据，直接提示并返回默认值
        if family_id not in self.family_history_data:
            logger.warning(f"家庭{family_id}没有历史数据，无法计算预算预测指标")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        history_data = self.family_history_data[family_id]
        output_data = self.family_output_data[family_id]
        
        # 从expenditure_categories中提取历史支出数据
        expenditure_categories = history_data.get('expenditure_categories', {})
        if not expenditure_categories:
            logger.warning(f"家庭{family_id}没有历史支出类别数据")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        # 获取第一个类别的数据长度来确定有多少年的数据
        first_category_key = None
        for key in self.category_keys:
            if key in expenditure_categories:
                first_category_key = key
                break
        
        if first_category_key is None:
            logger.warning(f"家庭{family_id}没有匹配的支出类别数据")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        first_category = expenditure_categories[first_category_key]
        num_years = len(first_category) if isinstance(first_category, list) else 0
        
        if num_years < 6:  # 至少需要6年数据（前5年+2021年实际值）
            logger.warning(f"家庭{family_id}数据年份不足，需要至少6年数据")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        # 提取2021年实际支出（假设最后一年是2021年）
        actual_2021_expenses = []
        for category in self.category_keys:
            expenses_list = expenditure_categories.get(category, [])
            if isinstance(expenses_list, list) and len(expenses_list) >= num_years:
                # 取最后一年作为2021年实际值
                expense = expenses_list[-1]
                expense = float(expense) if expense is not None else 0.0
            else:
                expense = 0.0
            actual_2021_expenses.append(expense)
        
        actual_2021_total = sum(actual_2021_expenses)
        if actual_2021_total == 0:
            logger.warning(f"家庭{family_id}的2021年实际总支出为0")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        # 获取LLM预测的2021年预算分配
        annual_allocation = output_data.get('annual_allocation', {})
        if not annual_allocation:
            logger.warning(f"家庭{family_id}没有LLM年度分配数据")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        predicted_2021_expenses = []
        for category in self.category_keys:
            allocation = annual_allocation.get(category, 0)
            predicted_2021_expenses.append(float(allocation) if allocation else 0.0)
        
        predicted_2021_total = sum(predicted_2021_expenses)
        if predicted_2021_total == 0:
            logger.warning(f"家庭{family_id}的LLM预测总预算为0")
            return {'mape': 1.0, 'structure_similarity': 0.0, 'legacy_consistency': 0.0}
        
        # 1. 计算MAPE (Mean Absolute Percentage Error)
        mape_values = []
        for actual, predicted in zip(actual_2021_expenses, predicted_2021_expenses):
            if actual > 0:  # 避免除零错误
                mape_value = abs(actual - predicted) / actual
                mape_values.append(mape_value)
        
        mape = np.mean(mape_values) if mape_values else 1.0
        mape = min(1.0, mape)  # 限制MAPE最大值为1.0
        
        # 2. 计算结构相似度 (基于支出比例的皮尔逊相关系数)
        actual_2021_structure = [exp/actual_2021_total for exp in actual_2021_expenses]
        predicted_2021_structure = [exp/predicted_2021_total for exp in predicted_2021_expenses]
        
        try:
            structure_correlation, _ = pearsonr(actual_2021_structure, predicted_2021_structure)
            structure_similarity = max(0, structure_correlation)  # 负相关设为0
        except Exception as e:
            logger.error(f"计算家庭{family_id}结构相似度失败: {e}")
            structure_similarity = 0.0
        
        # 3. 计算传统的一致性得分（基于历史数据的加权平均）
        legacy_consistency = self._calculate_legacy_consistency(
            expenditure_categories, predicted_2021_structure, num_years
        )
        
        logger.info(f"家庭{family_id}预测指标 - MAPE: {mape:.3f}, 结构相似度: {structure_similarity:.3f}, 传统一致性: {legacy_consistency:.3f}")
        
        return {
            'mape': mape,
            'structure_similarity': structure_similarity,
            'legacy_consistency': legacy_consistency
        }
    
    def _calculate_legacy_consistency(self, expenditure_categories: Dict, predicted_structure: List[float], num_years: int) -> float:
        """
        计算基于历史数据的传统一致性得分（原有方法）
        """
        try:
            # 构建历史支出矩阵（年份 x 类别）
            historical_expenses = []
            
            # 只取前5年的数据用于计算历史趋势（排除最后一年的2021年实际值）
            years_to_use = min(5, num_years - 1)  # 排除最后一年
            start_index = max(0, num_years - 1 - years_to_use)  # 从倒数第6年开始
            
            for year_idx in range(start_index, num_years - 1):  # 不包括最后一年
                year_expenses = []
                for category in self.category_keys:
                    expenses_list = expenditure_categories.get(category, [])
                    if isinstance(expenses_list, list) and len(expenses_list) > year_idx:
                        expense = expenses_list[year_idx]
                        expense = float(expense) if expense is not None else 0.0
                    else:
                        expense = 0.0
                    year_expenses.append(expense)
                
                # 计算该年度的总支出
                total_expense = sum(year_expenses)
                if total_expense > 0:
                    # 转换为比例
                    year_ratios = [exp/total_expense for exp in year_expenses]
                    historical_expenses.append(year_ratios)
            
            if len(historical_expenses) == 0:
                return 0.0
            
            # 计算加权平均历史结构（越近年份权重越大）
            weights = [0.6**(i) for i in range(len(historical_expenses))]  # 指数衰减权重
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]  # 归一化
            
            historical_structure = np.zeros(len(self.category_keys))
            for i, ratios in enumerate(historical_expenses):
                historical_structure += np.array(ratios) * weights[i]
            
            # 计算皮尔逊相关系数
            correlation, _ = pearsonr(historical_structure, predicted_structure)
            return max(0, correlation)  # 负相关设为0
            
        except Exception as e:
            logger.error(f"计算传统一致性得分失败: {e}")
            return 0.0

    def calculate_budget_consistency_score(self, family_id: str) -> float:
        """
        保持向后兼容性的方法，返回结构相似度作为主要得分
        """
        metrics = self.calculate_budget_prediction_metrics(family_id)
        return metrics['structure_similarity']
    
    def calculate_product_profile_consistency_score(self, family_id: str) -> float:
        """
        使用LLM评估商品选择与家庭画像的一致性
        """
        if family_id not in self.family_history_data or family_id not in self.family_output_data:
            return 0.0
        
        # 获取家庭画像
        history_data = self.family_history_data[family_id]
        family_profile = self._extract_family_profile(history_data)
        
        # 提取代表性商品清单
        shopping_plan = self.family_output_data[family_id].get('shopping_plan', {})
        representative_products = self._extract_representative_products(shopping_plan, max_products=20)
        
        if not representative_products:
            logger.warning(f"家庭{family_id}没有有效的商品数据")
            return 0.0
        
        # 构建LLM评估提示 - 简化版本以提高速度
        prompt = f"""评估任务：商品选择与家庭画像匹配度

家庭信息：{family_profile}

购买商品（前{len(representative_products)}项）：
"""
        for i, product in enumerate(representative_products, 1):
            product_name = product.get('name', product.get('名称', '未知商品'))
            product_quantity = product.get('quantity', product.get('数量', 1))
            prompt += f"{i}. {product_name} x{product_quantity}\n"
        
        prompt += """
评估维度：
1. 收入水平匹配 (商品价格与家庭收入的符合度)
2. 家庭规模适配 (商品数量与家庭人数的合理性)
3. 生活方式契合 (商品类型与家庭特征的一致性)
4. 整体协调性 (商品组合的逻辑性)

请根据家庭特征仔细评估，给出0-100分的评分。
注意：不同家庭差异很大，请避免给出相似分数，要体现个性化差异。

格式：评分：XX

评分："""
        
        try:
            system_prompt = """你是消费行为分析师。请仔细分析家庭画像和商品清单的匹配度，给出0-100分的评分。

要求：
1. 分析收入、家庭规模、生活方式、整体协调性四个维度
2. 评分要有区分度，避免集中在某几个数值
3. 严格按照"评分：XX"格式输出数字

不同家庭情况差异很大，评分应体现这种差异性。"""
            response = safe_llm_call(prompt, system_prompt)
            
            # 从回复中提取数字评分
            import re
            # 使用多种模式来提取评分，优先匹配我们要求的格式
            score_patterns = [
                r'评分[：:]\s*(\d+(?:\.\d+)?)',  # 首选：匹配 "评分：85" 或 "评分:85"
                r'得分[：:]\s*(\d+(?:\.\d+)?)',  # 匹配 "得分：85"
                r'(\d+(?:\.\d+)?)分',           # 匹配 "85分"
                r'[：:](\d+(?:\.\d+)?)(?:\s|$)', # 匹配结尾的 ":85"
                r'(\d+(?:\.\d+)?)/100',         # 匹配 "85/100"
                r'\b(\d+(?:\.\d+)?)\b'          # 最后：匹配独立数字
            ]
            
            score = None
            for pattern in score_patterns:
                score_match = re.search(pattern, response)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            if score is not None:
                # 添加小随机扰动以增加评分多样性 (±0.5分以内，最小化对画像一致性的干扰)
                import random
                random_factor = random.uniform(-0.5, 0.5)
                score = max(0, min(100, score + random_factor))
                
                score = score / 100  # 归一化到0-1
                logger.info(f"家庭{family_id}商品画像一致性得分: {score:.3f}")
                return score
            else:
                logger.warning(f"无法从LLM回复中提取评分: {response}")
                return 0.5  # 默认中等评分
                
        except Exception as e:
            logger.error(f"LLM评估家庭{family_id}商品一致性失败: {e}")
            return 0.5
    
    def _extract_family_profile(self, history_data: Dict) -> str:
        """从历史数据中提取家庭画像描述"""
        
        # 检查是否已经有现成的family_profile
        if 'family_profile' in history_data:
            return history_data['family_profile']
        
        # 从basic_family_info中提取信息
        basic_info = history_data.get('basic_family_info', {})
        wealth_info = history_data.get('family_wealth_situation', {})
        
        if not basic_info:
            return "普通家庭"
        
        # 提取关键信息
        family_size = basic_info.get('family_size', 1)
        head_age = basic_info.get('head_age', 0)
        head_gender = basic_info.get('head_gender', '')
        marital_status = basic_info.get('head_marital_status', '')
        num_children = basic_info.get('num_children', 0)
        num_vehicles = basic_info.get('num_vehicles', 0)
        life_satisfaction = basic_info.get('life_satisfaction', '')
        
        # 从wealth_analysis中提取收入信息
        wealth_analysis = wealth_info.get('wealth_analysis', '')
        
        # 构建画像描述
        profile_parts = []
        
        # 基本人口统计信息
        if head_age and head_gender:
            age_desc = "年轻" if head_age < 35 else "中年" if head_age < 60 else "老年"
            profile_parts.append(f"{age_desc}{head_gender}户主，年龄{head_age}岁")
        
        if family_size:
            profile_parts.append(f"家庭规模{family_size}人")
            
        if marital_status:
            profile_parts.append(f"婚姻状况：{marital_status}")
            
        if num_children > 0:
            profile_parts.append(f"有{num_children}个孩子")
        else:
            profile_parts.append("无子女")
            
        if num_vehicles > 0:
            profile_parts.append(f"拥有{num_vehicles}辆车")
        else:
            profile_parts.append("无车家庭")
            
        if life_satisfaction:
            profile_parts.append(f"生活满意度：{life_satisfaction}")
        
        # 财务状况描述
        if wealth_analysis:
            # 从财务分析中提取简要信息
            if "增长" in wealth_analysis or "increase" in wealth_analysis.lower():
                profile_parts.append("财务状况呈上升趋势")
            elif "下降" in wealth_analysis or "decrease" in wealth_analysis.lower():
                profile_parts.append("财务状况面临挑战")
            else:
                profile_parts.append("财务状况稳定")
        
        if profile_parts:
            return "，".join(profile_parts) + "。"
        else:
            return "普通家庭"
    
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
        
        # 2. 价格区间覆盖度 - 改进版：使用更细粒度的价格分析
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
        
        # 3. 数量多样性 - 改进版：结合多个维度
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
    
    def calculate_family_heterogeneity_score(self) -> Dict[str, float]:
        """
        计算不同家庭间的异质性得分
        使用改进的方法：基于画像-行为一致性的异质性评估
        """
        if len(self.family_output_data) < 2:
            logger.warning("家庭数量不足，无法计算异质性")
            return {
                'comprehensive_heterogeneity': 0.0,
                'detailed_analysis': {'error': '家庭数量不足，至少需要2个家庭'}
            }
        
        logger.info("开始计算家庭异质性得分...")
        
        # 导入改进的异质性分析器
        try:
            logger.info("尝试导入改进的异质性分析器...")
            from improved_heterogeneity import ImprovedHeterogeneityAnalyzer
            logger.info("成功导入异质性分析器")
            
            improved_analyzer = ImprovedHeterogeneityAnalyzer(self)
            logger.info("异质性分析器初始化完成")
            
            # 使用新的异质性分析方法，添加超时机制
            logger.info("开始生成综合异质性报告...")
            import threading
            
            # 使用线程来实现超时控制，而不是signal（因为在多线程中不可用）
            timeout_seconds = 600  # 10分钟超时
            result_container = [None]
            exception_container = [None]
            
            def heterogeneity_analysis_thread():
                try:
                    result_container[0] = improved_analyzer.generate_comprehensive_heterogeneity_report()
                except Exception as e:
                    exception_container[0] = e
            
            # 启动异质性分析线程
            thread = threading.Thread(target=heterogeneity_analysis_thread)
            thread.daemon = True
            thread.start()
            
            # 等待结果或超时
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                # 线程仍在运行，说明超时了
                logger.error("异质性分析超时，返回默认值")
                return {
                    'comprehensive_heterogeneity': 0.0, 
                    'detailed_analysis': {'error': '异质性分析超时（超过10分钟）'}
                }
            
            # 检查是否有异常
            if exception_container[0]:
                logger.error(f"异质性分析过程中发生错误: {exception_container[0]}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return {
                    'comprehensive_heterogeneity': 0.0, 
                    'detailed_analysis': {'error': f'异质性分析执行失败: {str(exception_container[0])}'}
                }
            
            # 检查结果
            if result_container[0] is None:
                logger.error("异质性分析返回了None")
                return {
                    'comprehensive_heterogeneity': 0.0, 
                    'detailed_analysis': {'error': '异质性分析返回了空结果'}
                }
            
            comprehensive_report = result_container[0]
            logger.info("异质性报告生成完成")
            
            result = {
                'comprehensive_heterogeneity': comprehensive_report.get('综合异质性得分', 0.0),
                'detailed_analysis': comprehensive_report
            }
            
            logger.info(f"综合异质性得分: {result['comprehensive_heterogeneity']:.3f}")
            
            return result
            
        except ImportError as e:
            logger.error(f"无法导入改进的异质性分析器: {e}")
            logger.warning("跳过异质性分析，继续生成报告...")
            return {
                'comprehensive_heterogeneity': 0.0,
                'detailed_analysis': {'error': f'无法导入improved_heterogeneity模块: {str(e)}'}
            }
        except Exception as e:
            logger.error(f"异质性分析初始化失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                'comprehensive_heterogeneity': 0.0,
                'detailed_analysis': {'error': f'异质性分析初始化失败: {str(e)}'}
            }
    
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
    
    def _evaluate_single_family(self, family_id: str) -> Dict[str, Any]:
        """
        评估单个家庭的所有指标（用于并发处理）
        """
        try:
            logger.info(f"开始评估家庭{family_id}...")
            start_time = time.time()
            
            # 计算各项指标
            logger.debug(f"家庭{family_id}: 计算预算一致性...")
            budget_consistency = self.calculate_budget_consistency_score(family_id)
            
            logger.debug(f"家庭{family_id}: 计算商品画像一致性...")
            product_consistency = self.calculate_product_profile_consistency_score(family_id)
            
            logger.debug(f"家庭{family_id}: 计算商品多样性...")
            diversity_scores = self.calculate_product_diversity_score(family_id)
            
            # 计算新的预测指标（MAPE和结构相似度）
            logger.debug(f"家庭{family_id}: 计算预测指标...")
            prediction_metrics = self.calculate_budget_prediction_metrics(family_id)
            
            elapsed_time = time.time() - start_time
            
            result = {
                '家庭ID': family_id,
                '预算一致性得分': round(budget_consistency, 3),
                '商品画像一致性得分': round(product_consistency, 3),
                '类别多样性得分': round(diversity_scores['category_diversity'], 3),
                '价格区间覆盖度': round(diversity_scores['price_coverage'], 3),
                '数量多样性得分': round(diversity_scores['quantity_diversity'], 3),
                '综合商品多样性得分': round(diversity_scores['overall_diversity'], 3),
                # 新增指标
                '绝对预测误差MAPE': round(prediction_metrics['mape'], 3),
                '预测结构相似度': round(prediction_metrics['structure_similarity'], 3),
                '传统一致性得分': round(prediction_metrics['legacy_consistency'], 3)
            }
            
            logger.info(f"家庭{family_id}评估完成，耗时{elapsed_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"评估家庭{family_id}时发生错误: {e}")
            import traceback
            logger.error(f"家庭{family_id}详细错误: {traceback.format_exc()}")
            # 返回默认值
            return {
                '家庭ID': family_id,
                '预算一致性得分': 0.0,
                '商品画像一致性得分': 0.0,
                '类别多样性得分': 0.0,
                '价格区间覆盖度': 0.0,
                '数量多样性得分': 0.0,
                '综合商品多样性得分': 0.0,
                # 新增指标默认值
                '绝对预测误差MAPE': 1.0,
                '预测结构相似度': 0.0,
                '传统一致性得分': 0.0
            }
    
    def _save_progress(self, completed_families: set, family_results: dict, batch_num: int):
        """保存当前进度和结果到暂存文件"""
        try:
            # 保存进度信息
            progress_data = {
                'completed_families': list(completed_families),
                'total_completed': len(completed_families),
                'current_batch': batch_num,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            # 保存家庭评估结果（按ID排序）
            sorted_results = dict(sorted(family_results.items(), key=lambda x: int(x[0])))
            with open(self.results_cache_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"进度已保存: 已完成{len(completed_families)}个家庭，当前批次{batch_num}")
            
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def _load_progress(self) -> tuple:
        """从暂存文件加载之前的进度"""
        completed_families = set()
        family_results = {}
        last_batch = 0
        
        try:
            # 加载进度信息
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    completed_families = set(progress_data.get('completed_families', []))
                    last_batch = progress_data.get('current_batch', 0)
                    logger.info(f"发现暂存进度: 已完成{len(completed_families)}个家庭，上次批次{last_batch}")
            
            # 加载家庭评估结果
            if os.path.exists(self.results_cache_file):
                with open(self.results_cache_file, 'r', encoding='utf-8') as f:
                    family_results = json.load(f)
                    logger.info(f"加载了{len(family_results)}个家庭的评估结果")
            
        except Exception as e:
            logger.error(f"加载暂存进度失败: {e}")
            completed_families = set()
            family_results = {}
            last_batch = 0
        
        return completed_families, family_results, last_batch
    
    def _clean_temp_files(self):
        """清理暂存文件"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            if os.path.exists(self.results_cache_file):
                os.remove(self.results_cache_file)
            logger.info("暂存文件已清理")
        except Exception as e:
            logger.warning(f"清理暂存文件失败: {e}")
    
    def generate_evaluation_report(self, output_file_path: str = None, resume_from_cache: bool = True) -> Dict[str, Any]:
        """生成完整的评估报告（并发版本，支持暂存和恢复）"""
        logger.info("开始生成评估报告...")
        start_time = time.time()
        
        if not output_file_path:
            # 使用固定的文件名，保存在assessment目录下
            output_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),  # assessment目录
                "evaluation_report.json"
            )
        
        family_ids = list(self.family_output_data.keys())
        total_families = len(family_ids)
        
        # 按ID排序家庭列表
        family_ids.sort(key=lambda x: int(x))
        
        # 尝试从暂存恢复进度
        completed_families = set()
        family_results = {}
        last_batch = 0
        
        if resume_from_cache:
            completed_families, family_results, last_batch = self._load_progress()
        
        # 筛选出还需要处理的家庭
        remaining_families = [fid for fid in family_ids if fid not in completed_families]
        
        if completed_families:
            logger.info(f"从暂存恢复: 已完成{len(completed_families)}个家庭，还需处理{len(remaining_families)}个家庭")
        else:
            logger.info(f"开始新的评估: 总共需要处理{total_families}个家庭")
        
        logger.info(f"将并发评估剩余{len(remaining_families)}个家庭，使用{self.max_workers}个线程，批量大小{self.batch_size}")
        
        # 分批处理剩余家庭
        batch_num = last_batch
        for i in range(0, len(remaining_families), self.batch_size):
            batch_families = remaining_families[i:i + self.batch_size]
            batch_num += 1
            
            logger.info(f"开始处理第{batch_num}批，包含{len(batch_families)}个家庭 (总进度: {len(completed_families) + i}/{total_families})")
            
            # 使用线程池并发处理当前批次
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交当前批次的家庭评估任务
                future_to_family = {
                    executor.submit(self._evaluate_single_family, family_id): family_id 
                    for family_id in batch_families
                }
                
                # 收集当前批次的结果
                batch_results = {}
                for future in concurrent.futures.as_completed(future_to_family):
                    family_id = future_to_family[future]
                    try:
                        result = future.result()
                        batch_results[family_id] = result
                        completed_families.add(family_id)
                        logger.debug(f"家庭{family_id}评估完成")
                    except Exception as e:
                        logger.error(f"获取家庭{family_id}评估结果失败: {e}")
                        # 添加默认结果以确保进度记录
                        default_result = {
                            '家庭ID': family_id,
                            '预算一致性得分': 0.0,
                            '商品画像一致性得分': 0.0,
                            '类别多样性得分': 0.0,
                            '价格区间覆盖度': 0.0,
                            '数量多样性得分': 0.0,
                            '综合商品多样性得分': 0.0
                        }
                        batch_results[family_id] = default_result
                        completed_families.add(family_id)
                
                # 将当前批次结果合并到总结果中
                family_results.update(batch_results)
                
                # 保存进度和结果到暂存文件
                self._save_progress(completed_families, family_results, batch_num)
                
                logger.info(f"第{batch_num}批处理完成，当前总进度: {len(completed_families)}/{total_families}")
        
        # 收集得分用于计算平均值
        budget_consistency_scores = []
        product_consistency_scores = []
        product_diversity_scores = []
        mape_scores = []
        structure_similarity_scores = []
        legacy_consistency_scores = []
        
        for result in family_results.values():
            budget_consistency_scores.append(result['预算一致性得分'])
            product_consistency_scores.append(result['商品画像一致性得分'])
            product_diversity_scores.append(result['综合商品多样性得分'])
            # 新增指标
            mape_scores.append(result.get('绝对预测误差MAPE', 1.0))
            structure_similarity_scores.append(result.get('预测结构相似度', 0.0))
            legacy_consistency_scores.append(result.get('传统一致性得分', 0.0))
        
        # 计算家庭异质性（这个不能并发，因为需要所有家庭的数据）
        logger.info(f"个体家庭评估已完成，开始计算家庭异质性...（当前已完成{len(completed_families)}个家庭）")
        try:
            heterogeneity_results = self.calculate_family_heterogeneity_score()
            logger.info("家庭异质性计算完成")
        except Exception as het_error:
            logger.error(f"家庭异质性计算失败: {het_error}")
            import traceback
            logger.error(f"异质性计算详细错误: {traceback.format_exc()}")
            # 使用默认值继续
            heterogeneity_results = {
                'comprehensive_heterogeneity': 0.0,
                'detailed_analysis': {'error': f'异质性计算失败: {str(het_error)}'}
            }
        
        # 计算平均得分
        avg_budget_consistency = np.mean(budget_consistency_scores) if budget_consistency_scores else 0
        avg_product_consistency = np.mean(product_consistency_scores) if product_consistency_scores else 0
        avg_product_diversity = np.mean(product_diversity_scores) if product_diversity_scores else 0
        # 新增指标的平均值
        avg_mape = np.mean(mape_scores) if mape_scores else 1.0
        avg_structure_similarity = np.mean(structure_similarity_scores) if structure_similarity_scores else 0.0
        avg_legacy_consistency = np.mean(legacy_consistency_scores) if legacy_consistency_scores else 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 按家庭ID排序家庭详细评估（确保最终输出有序）
        sorted_family_results = {}
        for family_id in sorted(family_results.keys(), key=lambda x: int(x)):
            sorted_family_results[family_id] = family_results[family_id]
        
        # 构建评估报告
        evaluation_report = {
            '评估摘要': {
                '评估日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '家庭总数': len(sorted_family_results),
                '评估耗时(秒)': round(total_time, 2),
                '并发线程数': self.max_workers,
                '批量大小': self.batch_size,
                '平均预算一致性得分': round(avg_budget_consistency, 3),
                '平均商品画像一致性得分': round(avg_product_consistency, 3),
                '平均商品多样性得分': round(avg_product_diversity, 3),
                # 新增预测指标
                '平均绝对预测误差MAPE': round(avg_mape, 3),
                '平均预测结构相似度': round(avg_structure_similarity, 3),
                '平均传统一致性得分': round(avg_legacy_consistency, 3),
                '综合异质性得分': round(heterogeneity_results.get('comprehensive_heterogeneity', 0), 3),
                '异质性指标': {
                    '商品选择异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('商品异质性', {}).get('异质性得分', 0), 3),
                    '预算分配异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('预算异质性', {}).get('异质性得分', 0), 3),
                    '购买模式异质性': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('模式异质性', {}).get('异质性得分', 0), 3),
                    '聚类质量得分': round(heterogeneity_results.get('detailed_analysis', {}).get('新异质性分析', {}).get('综合指标', {}).get('聚类质量', 0), 3)
                }
            },
            '家庭详细评估': sorted_family_results,
            '异质性详细分析': heterogeneity_results.get('detailed_analysis', {}),
            '评估说明': {
                '预算一致性得分': '基于LLM预算分配与家庭历史消费模式的皮尔逊相关系数',
                '商品画像一致性得分': '基于LLM对商品选择与家庭画像匹配度的评估',
                '商品多样性得分': '综合类别多样性、价格区间覆盖度和数量多样性的得分',
                # 新增指标说明
                '绝对预测误差MAPE': 'LLM预测2021年各类支出与实际2021年支出的平均绝对百分比误差，值越小表示预测越准确',
                '预测结构相似度': 'LLM预测2021年支出结构与实际2021年支出结构的皮尔逊相关系数，值越高表示结构预测越相似',
                '传统一致性得分': '基于前5年历史数据的加权平均结构与LLM预测结构的相关系数，反映历史趋势一致性',
                '综合异质性得分': '基于画像-行为一致性、消费模式正确性和家庭差异化的综合异质性评分',
                '新异质性指标说明': {
                    '商品选择异质性': '基于TF-IDF商品类别向量的组内/组间相似度差异，衡量相似家庭是否选择相似商品',
                    '预算分配异质性': '基于13类支出比例向量的组内/组间相似度差异，衡量相似家庭是否有相似预算分配',
                    '购买模式异质性': '基于价格偏好、购买数量等行为特征的组内/组间相似度差异',
                    '聚类质量得分': '家庭画像聚类的轮廓系数，反映聚类的合理性和区分度'
                },
                '暂存机制说明': f'支持分批处理(批量大小:{self.batch_size})和断点恢复，暂存文件保存在temp_results目录'
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
            logger.info("开始转换NumPy类型...")
            serializable_report = convert_numpy_types(evaluation_report)
            logger.info("NumPy类型转换完成")
            
            # 尝试JSON序列化以提前发现问题
            logger.info("尝试JSON序列化测试...")
            try:
                json_test = json.dumps(serializable_report, ensure_ascii=False, indent=2)
                logger.info("JSON序列化测试成功")
            except Exception as json_error:
                logger.error(f"JSON序列化测试失败: {json_error}")
                
                # 逐个检查主要部分以定位问题
                logger.info("开始逐个检查主要部分...")
                
                # 检查评估摘要
                try:
                    json.dumps(serializable_report.get('评估摘要', {}), ensure_ascii=False)
                    logger.info("✅ 评估摘要部分JSON序列化正常")
                except Exception as e:
                    logger.error(f"❌ 评估摘要部分有问题: {e}")
                
                # 检查家庭详细评估（取前几个样本）
                try:
                    family_sample = dict(list(serializable_report.get('家庭详细评估', {}).items())[:3])
                    json.dumps(family_sample, ensure_ascii=False)
                    logger.info("✅ 家庭详细评估部分JSON序列化正常")
                except Exception as e:
                    logger.error(f"❌ 家庭详细评估部分有问题: {e}")
                
                # 检查异质性详细分析
                try:
                    json.dumps(serializable_report.get('异质性详细分析', {}), ensure_ascii=False)
                    logger.info("✅ 异质性详细分析部分JSON序列化正常")
                except Exception as e:
                    logger.error(f"❌ 异质性详细分析部分有问题: {e}")
                    
                    # 进一步检查异质性分析的子部分
                    heterogeneity_data = serializable_report.get('异质性详细分析', {})
                    for key, value in heterogeneity_data.items():
                        try:
                            json.dumps(value, ensure_ascii=False)
                            logger.info(f"✅ 异质性分析子部分 '{key}' 正常")
                        except Exception as sub_e:
                            logger.error(f"❌ 异质性分析子部分 '{key}' 有问题: {sub_e}")
                            
                            # 如果是字典，进一步检查
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    try:
                                        json.dumps(sub_value, ensure_ascii=False)
                                    except Exception as subsub_e:
                                        logger.error(f"❌❌ 异质性分析 '{key}.{sub_key}' 有问题: {subsub_e}, 类型: {type(sub_value)}")
                
                # 检查评估说明
                try:
                    json.dumps(serializable_report.get('评估说明', {}), ensure_ascii=False)
                    logger.info("✅ 评估说明部分JSON序列化正常")
                except Exception as e:
                    logger.error(f"❌ 评估说明部分有问题: {e}")
                
                # 如果测试失败，尝试保存一个简化版本
                logger.warning("由于JSON序列化问题，尝试保存简化版本...")
                simplified_report = {
                    '评估摘要': serializable_report.get('评估摘要', {}),
                    '评估说明': serializable_report.get('评估说明', {}),
                    '错误信息': f"完整报告保存失败: {json_error}"
                }
                
                backup_path = output_file_path.replace('.json', '_simplified.json')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_report, f, ensure_ascii=False, indent=2)
                logger.warning(f"简化版本已保存到: {backup_path}")
                
                raise json_error
            
            # 如果测试通过，正常保存
            logger.info("开始保存完整报告...")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, ensure_ascii=False, indent=2)
            logger.info(f"评估报告已保存到: {output_file_path}")
            
            # 成功生成最终报告后，清理暂存文件
            self._clean_temp_files()
            
        except Exception as e:
            logger.error(f"保存评估报告失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("消费者模型评估报告摘要")
        print("="*80)
        print(f"评估日期: {evaluation_report['评估摘要']['评估日期']}")
        print(f"家庭总数: {evaluation_report['评估摘要']['家庭总数']}")
        print(f"评估耗时: {evaluation_report['评估摘要']['评估耗时(秒)']}秒")
        print(f"并发线程数: {evaluation_report['评估摘要']['并发线程数']}")
        print(f"批量大小: {evaluation_report['评估摘要']['批量大小']}")
        print("\n--- 基础评估结果 ---")
        print(f"平均预算一致性得分: {evaluation_report['评估摘要']['平均预算一致性得分']}")
        print(f"平均商品画像一致性得分: {evaluation_report['评估摘要']['平均商品画像一致性得分']}")
        print(f"平均商品多样性得分: {evaluation_report['评估摘要']['平均商品多样性得分']}")
        print("\n--- 新增预测验证指标 ---")
        print(f"平均绝对预测误差MAPE: {evaluation_report['评估摘要']['平均绝对预测误差MAPE']} (越小越好)")
        print(f"平均预测结构相似度: {evaluation_report['评估摘要']['平均预测结构相似度']} (越高越好)")
        print(f"平均传统一致性得分: {evaluation_report['评估摘要']['平均传统一致性得分']}")
        print(f"综合异质性得分: {evaluation_report['评估摘要']['综合异质性得分']}")
        
        # 显示新的异质性指标
        new_het_metrics = evaluation_report['评估摘要'].get('异质性指标', {})
        if new_het_metrics:
            print("\n--- 异质性指标 (组内相似 vs 组间差异) ---")
            print(f"商品选择异质性: {new_het_metrics.get('商品选择异质性', 0)}")
            print(f"预算分配异质性: {new_het_metrics.get('预算分配异质性', 0)}")
            print(f"购买模式异质性: {new_het_metrics.get('购买模式异质性', 0)}")
            print(f"聚类质量得分: {new_het_metrics.get('聚类质量得分', 0)}")
            
            # 显示聚类信息
            detailed_analysis = evaluation_report.get('异质性详细分析', {})
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
        
        return evaluation_report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='消费者模型评估系统')
    parser.add_argument('--max-workers', type=int, default=32,
                       help='最大并发线程数 (默认: 32)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='每批处理的家庭数量 (默认: 32)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径')
    parser.add_argument('--no-resume', action='store_true',
                       help='不从暂存恢复，重新开始评估')
    
    args = parser.parse_args()
    
    # 创建评估系统实例
    evaluation_system = ConsumerEvaluationSystem(
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # 生成评估报告
    resume_from_cache = not args.no_resume
    if not resume_from_cache:
        logger.info("指定不从暂存恢复，将重新开始评估")
        # 清理可能存在的暂存文件
        evaluation_system._clean_temp_files()
    
    report = evaluation_system.generate_evaluation_report(
        output_file_path=args.output,
        resume_from_cache=resume_from_cache
    )
    
    return report


if __name__ == "__main__":
    main()
