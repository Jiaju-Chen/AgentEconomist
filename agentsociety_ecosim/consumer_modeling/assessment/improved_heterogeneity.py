#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的异质性分析器 - 精简版
基于"组内相似，组间差异"的异质性评估方法
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict, Counter
import math

# 机器学习相关导入
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedHeterogeneityAnalyzer:
    """改进的异质性分析器"""
    
    def __init__(self, eval_system):
        """初始化异质性分析器"""
        self.eval_system = eval_system
        self.logger = logger
    
    def create_product_category_vectors(self) -> Dict[str, np.ndarray]:
        """
        Step 1: 为每个家庭创建商品类别TF-IDF向量
        基于购买的商品类别，构建向量表示
        """
        logger.info("创建商品类别TF-IDF向量...")
        
        # 收集所有家庭的商品类别文档
        family_documents = {}
        all_categories = set()
        
        for family_id in self.eval_system.family_output_data.keys():
            shopping_plan = self.eval_system.family_output_data[family_id].get('shopping_plan', {})
            
            # 提取该家庭的所有商品子类别
            family_categories = []
            for month, category_data in shopping_plan.items():
                if not isinstance(category_data, dict):
                    continue
                for category, subcat_data in category_data.items():
                    if not isinstance(subcat_data, dict):
                        continue
                    for subcat, product_list in subcat_data.items():
                        if isinstance(product_list, list) and len(product_list) > 0:
                            # 根据购买数量加权
                            quantity = sum(p.get('quantity', 1) for p in product_list if isinstance(p, dict))
                            # 重复类别名称以反映购买强度
                            family_categories.extend([subcat] * min(quantity, 10))  # 限制最大重复次数
                            all_categories.add(subcat)
            
            if family_categories:
                # 创建该家庭的"商品类别文档"
                family_documents[family_id] = ' '.join(family_categories)
            else:
                family_documents[family_id] = ''
        
        if not family_documents or len(all_categories) < 2:
            logger.warning("商品类别数据不足，使用零向量")
            return {fid: np.zeros(1) for fid in family_documents.keys()}
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(
            max_features=min(100, len(all_categories)),  # 限制特征数量
            min_df=1,  # 至少在1个文档中出现
            max_df=0.95,  # 最多在95%的文档中出现
            ngram_range=(1, 1)
        )
        
        # 准备文档列表
        family_ids = list(family_documents.keys())
        documents = [family_documents[fid] for fid in family_ids]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            # 转换为字典格式
            vectors = {}
            for i, family_id in enumerate(family_ids):
                vectors[family_id] = tfidf_matrix[i].toarray().flatten()
            
            logger.info(f"成功创建{len(vectors)}个家庭的商品类别TF-IDF向量，维度: {tfidf_matrix.shape[1]}")
            return vectors
            
        except Exception as e:
            logger.error(f"TF-IDF向量化失败: {e}")
            # 返回简单的类别计数向量
            category_list = sorted(all_categories)
            vectors = {}
            for family_id in family_ids:
                vector = np.zeros(len(category_list))
                doc = family_documents[family_id]
                for i, cat in enumerate(category_list):
                    vector[i] = doc.count(cat)
                # 归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                vectors[family_id] = vector
            
            return vectors
    
    def create_budget_allocation_vectors(self) -> Dict[str, np.ndarray]:
        """
        Step 2: 为每个家庭创建预算分配向量
        基于13个支出类别的比例分配
        """
        logger.info("创建预算分配向量...")
        
        vectors = {}
        category_keys = self.eval_system.category_keys
        
        for family_id in self.eval_system.family_output_data.keys():
            annual_allocation = self.eval_system.family_output_data[family_id].get('annual_allocation', {})
            
            if not annual_allocation:
                vectors[family_id] = np.zeros(len(category_keys))
                continue
            
            # 提取各类别的预算分配
            budget_vector = []
            total_budget = sum(annual_allocation.get(category, 0) for category in category_keys)
            
            if total_budget == 0:
                vectors[family_id] = np.zeros(len(category_keys))
                continue
            
            # 转换为比例向量
            for category in category_keys:
                ratio = annual_allocation.get(category, 0) / total_budget
                budget_vector.append(ratio)
            
            vectors[family_id] = np.array(budget_vector)
        
        logger.info(f"成功创建{len(vectors)}个家庭的预算分配向量，维度: {len(category_keys)}")
        return vectors
    
    def create_behavior_pattern_vectors(self) -> Dict[str, np.ndarray]:
        """
        Step 3: 为每个家庭创建行为模式向量
        基于价格偏好、购买数量、购买频率等行为特征
        """
        logger.info("创建行为模式向量...")
        
        vectors = {}
        
        for family_id in self.eval_system.family_output_data.keys():
            shopping_plan = self.eval_system.family_output_data[family_id].get('shopping_plan', {})
            
            # 计算行为特征
            total_items = 0
            total_value = 0
            price_list = []
            quantity_list = []
            category_counts = defaultdict(int)
            
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
                            if isinstance(product, dict):
                                price = product.get('price', 0)
                                quantity = product.get('quantity', 1)
                                
                                total_items += 1
                                total_value += price * quantity
                                price_list.append(price)
                                quantity_list.append(quantity)
                                category_counts[subcat] += 1
            
            # 构建行为特征向量
            if total_items == 0:
                vectors[family_id] = np.zeros(11)  # 更新为11维
                continue
            
            avg_price = np.mean(price_list) if price_list else 0
            std_price = np.std(price_list) if price_list else 0
            avg_quantity = np.mean(quantity_list) if quantity_list else 0
            std_quantity = np.std(quantity_list) if quantity_list else 0
            
            # 价格偏好：高价位 vs 低价位商品比例
            if price_list:
                price_75th = np.percentile(price_list, 75)
                high_price_ratio = sum(1 for p in price_list if p > price_75th) / len(price_list)
            else:
                high_price_ratio = 0
            
            # 购买多样性：不同类别数量 / 总购买次数
            diversity = len(category_counts) / total_items if total_items > 0 else 0
            
            # 批量购买倾向：平均每类别购买数量
            avg_category_quantity = total_items / len(category_counts) if category_counts else 0
            
            # 消费强度：总价值 / 总商品数
            spending_intensity = total_value / total_items if total_items > 0 else 0
            
            behavior_vector = np.array([
                avg_price / 1000,  # 归一化平均价格
                std_price / 1000,  # 归一化价格标准差
                avg_quantity / 10,  # 归一化平均数量
                std_quantity / 10,  # 归一化数量标准差
                high_price_ratio,  # 高价商品偏好
                diversity,  # 购买多样性
                avg_category_quantity / 10,  # 归一化类别购买强度
                spending_intensity / 10000,  # 归一化消费强度
                # 新增特征：为大数据集提供更细致的区分度
                len(price_list) / 100 if price_list else 0,  # 购买频率
                np.median(price_list) / 1000 if price_list else 0,  # 中位价格偏好
                (np.percentile(price_list, 90) - np.percentile(price_list, 10)) / 1000 if len(price_list) > 5 else 0  # 价格范围
            ])
            
            vectors[family_id] = behavior_vector
        
        logger.info(f"成功创建{len(vectors)}个家庭的行为模式向量，维度: 11")
        return vectors
    
    def adaptive_family_clustering(self, vectors: Dict[str, np.ndarray]) -> Tuple[Dict[str, int], List[str], float]:
        """
        Step 4: 自适应家庭聚类
        基于轮廓系数选择最优聚类数量，支持大规模数据集
        """
        logger.info("执行自适应家庭聚类...")
        
        if len(vectors) < 3:
            logger.warning("家庭数量不足，无法进行有效聚类")
            return {}, [], 0.0
        
        # 准备聚类数据
        family_ids = list(vectors.keys())
        feature_matrix = np.array([vectors[fid] for fid in family_ids])
        n_families = len(family_ids)
        
        # 标准化特征
        feature_std = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-6)
        
        # 根据数据规模动态确定聚类数量范围
        if n_families <= 100:
            # 小规模数据：适中聚类数，为大数据集做准备
            min_k = 3
            max_k = min(15, max(8, n_families // 4))  # 至少每个聚类有4个家庭
        elif n_families <= 500:
            # 中小规模数据
            min_k = 8
            max_k = min(25, n_families // 8)  # 至少每个聚类有8个家庭
        elif n_families <= 2000:
            # 中等规模数据
            min_k = 15
            max_k = min(50, n_families // 15)  # 至少每个聚类有15个家庭
        elif n_families <= 5000:
            # 中大规模数据
            min_k = 25
            max_k = min(80, n_families // 20)  # 至少每个聚类有20个家庭
        else:
            # 大规模数据（8000+）
            min_k = 40
            max_k = min(150, n_families // 25)  # 至少每个聚类有25个家庭
        
        logger.info(f"数据规模: {n_families}个家庭，聚类数量范围: {min_k}-{max_k}")
        
        # 尝试不同的聚类数量
        best_k = min_k
        best_score = -1
        best_labels = None
        k_scores = []
        
        for k in range(min_k, max_k + 1):
            try:
                # 使用更多初始化次数提高聚类质量
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
                labels = kmeans.fit_predict(feature_std)
                
                # 检查聚类结果是否有效（避免空聚类）
                unique_labels = np.unique(labels)
                if len(unique_labels) < k:
                    logger.warning(f"聚类k={k}产生空聚类，跳过")
                    continue
                
                # 计算轮廓系数
                score = silhouette_score(feature_std, labels)
                k_scores.append((k, score))
                
                # 调整评分策略：在数据规模较小时，偏向更多聚类
                cluster_sizes = [np.sum(labels == i) for i in range(k)]
                min_size = min(cluster_sizes)
                max_size = max(cluster_sizes)
                
                # 聚类平衡性评分
                balance_score = 1.0
                if min_size > 0:
                    imbalance_ratio = max_size / min_size
                    # 对于大数据集，允许更大的不平衡
                    imbalance_threshold = 3 if n_families <= 100 else (5 if n_families <= 1000 else 8)
                    if imbalance_ratio > imbalance_threshold * 2:
                        balance_score = 0.7
                    elif imbalance_ratio > imbalance_threshold:
                        balance_score = 0.85
                
                # 聚类数量奖励策略：根据数据规模调整
                cluster_bonus = 0.0
                if n_families <= 100:
                    # 小数据集：轻微鼓励更多聚类
                    cluster_bonus = min(0.05, (k - min_k) * 0.01)
                elif n_families <= 1000:
                    # 中等数据集：适度鼓励更多聚类
                    cluster_bonus = min(0.08, (k - min_k) * 0.015)
                else:
                    # 大数据集：显著鼓励更多聚类，发现细分家庭类型
                    cluster_bonus = min(0.12, (k - min_k) * 0.02)
                
                # 综合评分：轮廓系数 + 平衡性 + 聚类数量奖励
                adjusted_score = score * balance_score + cluster_bonus
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_k = k
                    best_labels = labels
                    
                logger.info(f"k={k}: 轮廓系数={score:.3f}, 平衡度={balance_score:.3f}, 调整后={adjusted_score:.3f}, 聚类大小={cluster_sizes}")
                    
            except Exception as e:
                logger.warning(f"聚类k={k}失败: {e}")
                continue
        
        if best_labels is None:
            logger.error("所有聚类尝试都失败了")
            return {}, [], 0.0
        
        # 生成聚类标签映射
        cluster_labels = {family_ids[i]: int(best_labels[i]) for i in range(len(family_ids))}
        
        # 生成聚类名称（基于家庭画像特征）
        cluster_names = self._generate_cluster_names(cluster_labels, best_k)
        
        # 输出聚类统计信息
        cluster_sizes = [np.sum(best_labels == i) for i in range(best_k)]
        logger.info(f"最优聚类数: {best_k}, 轮廓系数: {best_score:.3f}")
        logger.info(f"聚类大小分布: {cluster_sizes}")
        
        return cluster_labels, cluster_names, float(best_score)
    
    def _generate_cluster_names(self, cluster_labels: Dict[str, int], n_clusters: int) -> List[str]:
        """生成更细致的聚类名称，支持多维度特征组合"""
        cluster_names = []
        
        # 分析每个聚类的家庭特征
        for cluster_id in range(n_clusters):
            cluster_families = [fid for fid, cid in cluster_labels.items() if cid == cluster_id]
            
            # 收集该聚类家庭的多维特征
            income_levels = []
            age_groups = []
            family_types = []
            education_levels = []
            employment_statuses = []
            spending_patterns = []
            
            for family_id in cluster_families:
                if family_id in self.eval_system.family_history_data:
                    basic_info = self.eval_system.family_history_data[family_id].get('basic_family_info', {})
                    
                    # 1. 分析收入水平（基于总支出）
                    annual_allocation = self.eval_system.family_output_data[family_id].get('annual_allocation', {})
                    total_spending = sum(annual_allocation.values()) if annual_allocation else 0
                    
                    if total_spending > 150000:
                        income_levels.append('超高收入')
                    elif total_spending > 120000:
                        income_levels.append('高收入')
                    elif total_spending > 90000:
                        income_levels.append('中高收入') 
                    elif total_spending > 60000:
                        income_levels.append('中等收入')
                    elif total_spending > 40000:
                        income_levels.append('中低收入')
                    elif total_spending > 25000:
                        income_levels.append('低收入')
                    else:
                        income_levels.append('极低收入')
                    
                    # 2. 分析年龄组（更细致）
                    head_age = basic_info.get('head_age', 40)
                    if head_age < 25:
                        age_groups.append('青年期')
                    elif head_age < 35:
                        age_groups.append('成年期')
                    elif head_age < 45:
                        age_groups.append('中年期')
                    elif head_age < 55:
                        age_groups.append('成熟期')
                    elif head_age < 65:
                        age_groups.append('中老年期')
                    elif head_age < 75:
                        age_groups.append('老年期')
                    else:
                        age_groups.append('高龄期')
                    
                    # 3. 分析家庭类型（更详细）
                    num_children = basic_info.get('num_children', 0)
                    family_size = basic_info.get('family_size', 1)
                    marital_status = basic_info.get('head_marital_status', '')
                    
                    if num_children >= 3:
                        family_types.append('多孩家庭')
                    elif num_children == 2:
                        family_types.append('二孩家庭')
                    elif num_children == 1:
                        family_types.append('独孩家庭')
                    elif family_size == 1:
                        family_types.append('单身')
                    elif family_size == 2 and ('已婚' in marital_status or 'married' in marital_status.lower()):
                        family_types.append('夫妻二人')
                    elif family_size > 2:
                        family_types.append('大家庭')
                    else:
                        family_types.append('小家庭')
                    
                    # 4. 分析教育水平
                    education = basic_info.get('head_education', '').lower()
                    if 'phd' in education or 'doctor' in education or '博士' in education:
                        education_levels.append('博士')
                    elif 'master' in education or 'graduate' in education or '硕士' in education or '研究生' in education:
                        education_levels.append('硕士')
                    elif 'bachelor' in education or 'college' in education or '本科' in education or '大学' in education:
                        education_levels.append('本科')
                    elif 'high' in education or '高中' in education:
                        education_levels.append('高中')
                    else:
                        education_levels.append('其他')
                    
                    # 5. 分析就业状态
                    employment = basic_info.get('head_employment_status', '').lower()
                    if 'employed' in employment or '在职' in employment or '工作' in employment:
                        employment_statuses.append('在职')
                    elif 'retired' in employment or '退休' in employment:
                        employment_statuses.append('退休')
                    elif 'unemployed' in employment or '失业' in employment:
                        employment_statuses.append('失业')
                    elif 'student' in employment or '学生' in employment:
                        employment_statuses.append('学生')
                    else:
                        employment_statuses.append('其他')
                    
                    # 6. 分析消费模式（更细致分类）
                    if annual_allocation:
                        total = sum(annual_allocation.values())
                        if total > 0:
                            food_ratio = annual_allocation.get('Food', 0) / total
                            housing_ratio = annual_allocation.get('Housing', 0) / total
                            transport_ratio = annual_allocation.get('Transportation', 0) / total
                            entertainment_ratio = annual_allocation.get('Entertainment', 0) / total
                            healthcare_ratio = annual_allocation.get('Healthcare', 0) / total
                            education_ratio = annual_allocation.get('Education', 0) / total
                            
                            # 根据主要支出类别进行细分
                            if food_ratio > 0.45:
                                spending_patterns.append('生存型')
                            elif food_ratio > 0.35:
                                spending_patterns.append('温饱型')
                            elif housing_ratio > 0.4:
                                spending_patterns.append('置业型')
                            elif housing_ratio > 0.25:
                                spending_patterns.append('居住型')
                            elif transport_ratio > 0.25:
                                spending_patterns.append('出行型')
                            elif entertainment_ratio > 0.15:
                                spending_patterns.append('享受型')
                            elif healthcare_ratio > 0.15:
                                spending_patterns.append('健康型')
                            elif education_ratio > 0.12:
                                spending_patterns.append('教育型')
                            else:
                                # 进一步细分均衡消费
                                if entertainment_ratio + healthcare_ratio > 0.2:
                                    spending_patterns.append('品质型')
                                elif transport_ratio + entertainment_ratio > 0.25:
                                    spending_patterns.append('活跃型')
                                else:
                                    spending_patterns.append('均衡型')
            
            # 生成聚类名称：使用最常见特征的组合
            def get_most_common(feature_list, default=''):
                if not feature_list:
                    return default
                counter = Counter(feature_list)
                return counter.most_common(1)[0][0]
            
            income = get_most_common(income_levels, '中等收入')
            age = get_most_common(age_groups, '中年期')
            family_type = get_most_common(family_types, '小家庭')
            education = get_most_common(education_levels, '')
            employment = get_most_common(employment_statuses, '')
            spending = get_most_common(spending_patterns, '')
            
            # 构建聚类名称：优先使用区分度高的特征
            name_parts = [income, age, family_type]
            
            # 如果教育或就业状态有明显特征，加入名称
            if education and education != '其他':
                if len([e for e in education_levels if e == education]) / len(cluster_families) > 0.6:
                    name_parts.append(education)
            
            if employment and employment != '其他':
                if len([e for e in employment_statuses if e == employment]) / len(cluster_families) > 0.6:
                    name_parts.append(employment)
            
            if spending and spending != '均衡型':
                if len([s for s in spending_patterns if s == spending]) / len(cluster_families) > 0.5:
                    name_parts.append(spending)
            
            # 组合名称，确保合理长度
            if len(name_parts) > 3:
                # 优先保留收入、年龄、家庭类型
                name_parts = name_parts[:3]
            
            cluster_name = ''.join(name_parts)
            
            # 如果名称重复，添加聚类ID作为后缀
            base_name = cluster_name
            counter = 1
            while cluster_name in cluster_names:
                cluster_name = f"{base_name}_{counter}"
                counter += 1
            
            cluster_names.append(cluster_name)
        
        return cluster_names
    
    def calculate_intra_inter_group_similarity(self) -> Dict[str, Any]:
        """
        Step 5: 计算组内/组间相似度差异
        这是核心的异质性评估方法
        """
        logger.info("计算组内/组间相似度...")
        
        # 创建各类向量
        product_vectors = self.create_product_category_vectors()
        budget_vectors = self.create_budget_allocation_vectors()
        behavior_vectors = self.create_behavior_pattern_vectors()
        
        if not product_vectors or not budget_vectors or not behavior_vectors:
            logger.error("向量创建失败")
            return self._create_empty_similarity_result()
        
        # 基于综合特征进行聚类
        combined_vectors = {}
        for family_id in product_vectors.keys():
            if family_id in budget_vectors and family_id in behavior_vectors:
                # 组合三类向量
                combined_vector = np.concatenate([
                    product_vectors[family_id] * 0.4,  # 商品向量权重
                    budget_vectors[family_id] * 0.4,   # 预算向量权重
                    behavior_vectors[family_id] * 0.2  # 行为向量权重
                ])
                combined_vectors[family_id] = combined_vector
        
        # 自适应聚类
        cluster_labels, cluster_names, clustering_quality = self.adaptive_family_clustering(combined_vectors)
        
        if not cluster_labels:
            logger.error("聚类失败")
            return self._create_empty_similarity_result()
        
        # 构建结果字典
        results = {}
        
        # 1. 商品选择异质性
        product_result = self._calculate_dimension_similarity(
            product_vectors, cluster_labels, "商品选择"
        )
        results['product_heterogeneity'] = product_result
        
        # 2. 预算分配异质性
        budget_result = self._calculate_dimension_similarity(
            budget_vectors, cluster_labels, "预算分配"
        )
        results['budget_heterogeneity'] = budget_result
        
        # 3. 购买模式异质性
        pattern_result = self._calculate_dimension_similarity(
            behavior_vectors, cluster_labels, "购买模式"
        )
        results['pattern_heterogeneity'] = pattern_result
        
        # 4. 综合异质性评分
        comprehensive_score = (
            product_result['heterogeneity_score'] * 0.4 + 
            budget_result['heterogeneity_score'] * 0.35 + 
            pattern_result['heterogeneity_score'] * 0.25
        )
        
        results['comprehensive'] = {
            'heterogeneity_score': float(comprehensive_score),
            'clustering_quality': float(clustering_quality),
            'cluster_names': cluster_names,
            'cluster_sizes': {name: sum(1 for c in cluster_labels.values() if c == i) 
                            for i, name in enumerate(cluster_names)}
        }
        
        # 5. 质量验证指标
        validation_results = self._validate_heterogeneity_quality(
            cluster_labels, product_vectors, budget_vectors, behavior_vectors
        )
        results['validation'] = validation_results
        
        logger.info(f"综合异质性评分: {comprehensive_score:.3f}")
        return results
    
    def _calculate_dimension_similarity(self, vectors: Dict[str, np.ndarray], 
                                      cluster_labels: Dict[str, int], 
                                      dimension_name: str) -> Dict[str, float]:
        """计算特定维度的组内/组间相似度"""
        
        intra_similarities = []  # 组内相似度
        inter_similarities = []  # 组间相似度
        
        # 按聚类分组
        clusters = defaultdict(list)
        for family_id, cluster_id in cluster_labels.items():
            if family_id in vectors:
                clusters[cluster_id].append((family_id, vectors[family_id]))
        
        # 计算组内相似度
        for cluster_id, family_vectors in clusters.items():
            if len(family_vectors) > 1:
                for i in range(len(family_vectors)):
                    for j in range(i + 1, len(family_vectors)):
                        vec1 = family_vectors[i][1]
                        vec2 = family_vectors[j][1]
                        
                        # 组合余弦相似度和欧氏距离
                        cosine_sim = self._safe_cosine_similarity(vec1, vec2)
                        euclidean_sim = 1 / (1 + np.linalg.norm(vec1 - vec2))
                        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
                        
                        intra_similarities.append(combined_sim)
        
        # 计算组间相似度
        cluster_pairs = [(i, j) for i in clusters.keys() for j in clusters.keys() if i < j]
        for cluster_i, cluster_j in cluster_pairs:
            for family_i, vec_i in clusters[cluster_i]:
                for family_j, vec_j in clusters[cluster_j]:
                    cosine_sim = self._safe_cosine_similarity(vec_i, vec_j)
                    euclidean_sim = 1 / (1 + np.linalg.norm(vec_i - vec_j))
                    combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
                    
                    inter_similarities.append(combined_sim)
        
        # 计算平均相似度
        avg_intra = float(np.mean(intra_similarities)) if intra_similarities else 0.0
        avg_inter = float(np.mean(inter_similarities)) if inter_similarities else 0.0
        
        # 计算异质性评分
        epsilon = 1e-6
        heterogeneity_score = (avg_intra - avg_inter) / (avg_intra + avg_inter + epsilon)
        
        # 确保异质性评分在0-1范围内
        heterogeneity_score = max(0, min(1, (heterogeneity_score + 1) / 2))
        
        result = {
            'avg_intra_similarity': avg_intra,
            'avg_inter_similarity': avg_inter,
            'heterogeneity_score': float(heterogeneity_score),
            'intra_count': len(intra_similarities),
            'inter_count': len(inter_similarities)
        }
        
        logger.info(f"{dimension_name}异质性 - 组内相似度: {avg_intra:.3f}, 组间相似度: {avg_inter:.3f}, 异质性评分: {heterogeneity_score:.3f}")
        return result
    
    def _safe_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """安全的余弦相似度计算，处理零向量"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _validate_heterogeneity_quality(self, cluster_labels: Dict[str, int], 
                                       product_vectors: Dict[str, np.ndarray],
                                       budget_vectors: Dict[str, np.ndarray], 
                                       behavior_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """验证异质性质量"""
        
        # 1. 聚类稳定性：多次聚类的一致性
        stability_scores = []
        feature_matrix = []
        family_ids = list(cluster_labels.keys())
        
        for fid in family_ids:
            if fid in product_vectors and fid in budget_vectors and fid in behavior_vectors:
                combined_vector = np.concatenate([
                    product_vectors[fid] * 0.4,
                    budget_vectors[fid] * 0.4,
                    behavior_vectors[fid] * 0.2
                ])
                feature_matrix.append(combined_vector)
        
        if len(feature_matrix) > 3:
            feature_matrix = np.array(feature_matrix)
            feature_std = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-6)
            
            n_clusters = len(set(cluster_labels.values()))
            for seed in [42, 123, 456]:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
                    temp_labels = kmeans.fit_predict(feature_std)
                    original_labels = [cluster_labels[fid] for fid in family_ids if fid in cluster_labels]
                    
                    if len(temp_labels) == len(original_labels):
                        from sklearn.metrics import adjusted_rand_score
                        stability = adjusted_rand_score(original_labels, temp_labels)
                        stability_scores.append(max(0, stability))
                except:
                    continue
        
        cluster_stability = float(np.mean(stability_scores)) if stability_scores else 0.0
        
        # 2. 语义一致性：聚类内家庭画像的相似性
        semantic_consistency = 0.0
        
        # 3. 期望符合度：是否符合常识期望
        expectation_conformity = self._check_expectation_conformity(cluster_labels)
        
        # 4. 统计显著性
        statistical_significance = 0.0
        
        return {
            'cluster_stability': cluster_stability,
            'semantic_consistency': semantic_consistency,
            'expectation_conformity': expectation_conformity,
            'statistical_significance': statistical_significance
        }
    
    def _check_expectation_conformity(self, cluster_labels: Dict[str, int]) -> float:
        """检查异质性是否符合常识期望"""
        conformity_checks = 0
        total_checks = 0
        
        # 按聚类分组
        clusters = defaultdict(list)
        for family_id, cluster_id in cluster_labels.items():
            clusters[cluster_id].append(family_id)
        
        for cluster_id, family_list in clusters.items():
            if len(family_list) < 2:
                continue
            
            # 检查同聚类内家庭是否有相似的消费模式
            spending_patterns = []
            for family_id in family_list:
                annual_allocation = self.eval_system.family_output_data[family_id].get('annual_allocation', {})
                total_spending = sum(annual_allocation.values()) if annual_allocation else 0
                
                if total_spending > 0:
                    # 计算各类别支出比例
                    pattern = {}
                    for category in self.eval_system.category_keys:
                        pattern[category] = annual_allocation.get(category, 0) / total_spending
                    spending_patterns.append(pattern)
            
            if len(spending_patterns) >= 2:
                # 计算组内支出模式的相似性
                similarities = []
                for i in range(len(spending_patterns)):
                    for j in range(i + 1, len(spending_patterns)):
                        pattern1 = np.array(list(spending_patterns[i].values()))
                        pattern2 = np.array(list(spending_patterns[j].values()))
                        
                        # 计算余弦相似度
                        cosine_sim = self._safe_cosine_similarity(pattern1, pattern2)
                        similarities.append(cosine_sim)
                
                avg_similarity = np.mean(similarities)
                # 如果组内相似度高于阈值，认为符合期望
                if avg_similarity > 0.6:
                    conformity_checks += 1
                total_checks += 1
        
        return conformity_checks / total_checks if total_checks > 0 else 0.5
    
    def _create_empty_similarity_result(self) -> Dict[str, Any]:
        """创建空的相似度结果"""
        empty_dimension = {
            'avg_intra_similarity': 0.0,
            'avg_inter_similarity': 0.0,
            'heterogeneity_score': 0.0,
            'intra_count': 0,
            'inter_count': 0
        }
        
        return {
            'product_heterogeneity': empty_dimension.copy(),
            'budget_heterogeneity': empty_dimension.copy(),
            'pattern_heterogeneity': empty_dimension.copy(),
            'comprehensive': {
                'heterogeneity_score': 0.0,
                'clustering_quality': 0.0,
                'cluster_names': [],
                'cluster_sizes': {}
            },
            'validation': {
                'cluster_stability': 0.0,
                'semantic_consistency': 0.0,
                'expectation_conformity': 0.0,
                'statistical_significance': 0.0
            }
        }
    
    def generate_comprehensive_heterogeneity_report(self) -> Dict[str, Any]:
        """生成综合异质性报告"""
        logger.info("生成综合异质性报告...")
        
        # 使用新的异质性计算方法
        new_heterogeneity_results = self.calculate_intra_inter_group_similarity()
        
        # 转换NumPy类型为Python原生类型
        def convert_numpy_types(obj):
            """递归转换NumPy类型为Python原生类型，确保JSON序列化兼容"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 构建最终报告（内部使用英文键名，输出时转换为中文）
        internal_report = {
            # 新的异质性指标（主要）
            'comprehensive_heterogeneity_score': new_heterogeneity_results['comprehensive']['heterogeneity_score'],
            'new_heterogeneity_analysis': {
                'product_heterogeneity': new_heterogeneity_results['product_heterogeneity'],
                'budget_heterogeneity': new_heterogeneity_results['budget_heterogeneity'],
                'pattern_heterogeneity': new_heterogeneity_results['pattern_heterogeneity'],
                'comprehensive_metrics': new_heterogeneity_results['comprehensive'],
                'quality_validation': new_heterogeneity_results['validation']
            },
            
            # 解释和诊断
            'interpretation': self._interpret_new_heterogeneity_results(new_heterogeneity_results),
            'diagnostic_summary': self._create_diagnostic_summary(new_heterogeneity_results)
        }
        
        # 转换为中文键名的报告
        report = self._convert_to_chinese_keys(internal_report)
        
        return convert_numpy_types(report)
    
    def _convert_to_chinese_keys(self, internal_report: Dict[str, Any]) -> Dict[str, Any]:
        """将内部英文键名转换为中文键名用于输出"""
        
        # 获取内部数据
        new_analysis = internal_report['new_heterogeneity_analysis']
        interpretation = internal_report['interpretation']
        diagnostic = internal_report['diagnostic_summary']
        
        # 构建中文键名的报告
        chinese_report = {
            '综合异质性得分': internal_report['comprehensive_heterogeneity_score'],
            '新异质性分析': {
                '商品异质性': {
                    '组内平均相似度': new_analysis['product_heterogeneity']['avg_intra_similarity'],
                    '组间平均相似度': new_analysis['product_heterogeneity']['avg_inter_similarity'],
                    '异质性得分': new_analysis['product_heterogeneity']['heterogeneity_score'],
                    '组内对比数': new_analysis['product_heterogeneity']['intra_count'],
                    '组间对比数': new_analysis['product_heterogeneity']['inter_count']
                },
                '预算异质性': {
                    '组内平均相似度': new_analysis['budget_heterogeneity']['avg_intra_similarity'],
                    '组间平均相似度': new_analysis['budget_heterogeneity']['avg_inter_similarity'],
                    '异质性得分': new_analysis['budget_heterogeneity']['heterogeneity_score'],
                    '组内对比数': new_analysis['budget_heterogeneity']['intra_count'],
                    '组间对比数': new_analysis['budget_heterogeneity']['inter_count']
                },
                '模式异质性': {
                    '组内平均相似度': new_analysis['pattern_heterogeneity']['avg_intra_similarity'],
                    '组间平均相似度': new_analysis['pattern_heterogeneity']['avg_inter_similarity'],
                    '异质性得分': new_analysis['pattern_heterogeneity']['heterogeneity_score'],
                    '组内对比数': new_analysis['pattern_heterogeneity']['intra_count'],
                    '组间对比数': new_analysis['pattern_heterogeneity']['inter_count']
                },
                '综合指标': {
                    '异质性得分': new_analysis['comprehensive_metrics']['heterogeneity_score'],
                    '聚类质量': new_analysis['comprehensive_metrics']['clustering_quality'],
                    '聚类名称': new_analysis['comprehensive_metrics']['cluster_names'],
                    '聚类规模': new_analysis['comprehensive_metrics']['cluster_sizes']
                },
                '质量验证': {
                    '聚类稳定性': new_analysis['quality_validation']['cluster_stability'],
                    '语义一致性': new_analysis['quality_validation']['semantic_consistency'],
                    '期望符合度': new_analysis['quality_validation']['expectation_conformity'],
                    '统计显著性': new_analysis['quality_validation']['statistical_significance']
                }
            },
            '结果解释': {
                '总体评估': interpretation['overall_assessment'],
                '维度分析': {
                    '商品选择': {
                        '得分': interpretation['dimensional_analysis']['product_selection']['score'],
                        '组内相似度': interpretation['dimensional_analysis']['product_selection']['intra_similarity'],
                        '组间相似度': interpretation['dimensional_analysis']['product_selection']['inter_similarity'],
                        '评估': interpretation['dimensional_analysis']['product_selection']['assessment']
                    },
                    '预算分配': {
                        '得分': interpretation['dimensional_analysis']['budget_allocation']['score'],
                        '组内相似度': interpretation['dimensional_analysis']['budget_allocation']['intra_similarity'],
                        '组间相似度': interpretation['dimensional_analysis']['budget_allocation']['inter_similarity'],
                        '评估': interpretation['dimensional_analysis']['budget_allocation']['assessment']
                    },
                    '购买模式': {
                        '得分': interpretation['dimensional_analysis']['purchase_pattern']['score'],
                        '组内相似度': interpretation['dimensional_analysis']['purchase_pattern']['intra_similarity'],
                        '组间相似度': interpretation['dimensional_analysis']['purchase_pattern']['inter_similarity'],
                        '评估': interpretation['dimensional_analysis']['purchase_pattern']['assessment']
                    }
                },
                '优势': interpretation['strengths'],
                '弱点': interpretation['weaknesses'],
                '建议': interpretation['recommendations']
            },
            '诊断摘要': {
                '聚类分析': {
                    '聚类总数': diagnostic['cluster_analysis']['total_clusters'],
                    '聚类名称': diagnostic['cluster_analysis']['cluster_names'],
                    '聚类规模': diagnostic['cluster_analysis']['cluster_sizes'],
                    '聚类质量': diagnostic['cluster_analysis']['clustering_quality']
                },
                '相似度分布': {
                    '商品组内组间对比': {
                        '组内平均': diagnostic['similarity_distribution']['product_intra_vs_inter']['intra_avg'],
                        '组间平均': diagnostic['similarity_distribution']['product_intra_vs_inter']['inter_avg'],
                        '差值': diagnostic['similarity_distribution']['product_intra_vs_inter']['difference']
                    },
                    '预算组内组间对比': {
                        '组内平均': diagnostic['similarity_distribution']['budget_intra_vs_inter']['intra_avg'],
                        '组间平均': diagnostic['similarity_distribution']['budget_intra_vs_inter']['inter_avg'],
                        '差值': diagnostic['similarity_distribution']['budget_intra_vs_inter']['difference']
                    }
                },
                '质量指标': diagnostic['quality_metrics'],
                '关键洞察': diagnostic['key_insights']
            }
        }
        
        return chinese_report
    
    def _interpret_new_heterogeneity_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """解释新的异质性结果"""
        comprehensive = results['comprehensive']
        product_het = results['product_heterogeneity']
        budget_het = results['budget_heterogeneity']
        pattern_het = results['pattern_heterogeneity']
        validation = results['validation']
        
        interpretation = {
            'overall_assessment': '',
            'dimensional_analysis': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # 总体评估
        overall_score = comprehensive['heterogeneity_score']
        if overall_score >= 0.7:
            interpretation['overall_assessment'] = '优秀异质性：系统很好地实现了"相似家庭相似推荐，不同家庭差异推荐"的目标'
        elif overall_score >= 0.5:
            interpretation['overall_assessment'] = '良好异质性：系统在一定程度上体现了家庭差异，仍有提升空间'
        elif overall_score >= 0.3:
            interpretation['overall_assessment'] = '中等异质性：系统显示出一些个性化特征，但差异化程度有限'
        else:
            interpretation['overall_assessment'] = '较低异质性：系统推荐缺乏足够的个性化差异'
        
        # 维度分析
        interpretation['dimensional_analysis'] = {
            'product_selection': {
                'score': product_het['heterogeneity_score'],
                'intra_similarity': product_het['avg_intra_similarity'],
                'inter_similarity': product_het['avg_inter_similarity'],
                'assessment': self._assess_dimension_score(product_het['heterogeneity_score'], '商品选择')
            },
            'budget_allocation': {
                'score': budget_het['heterogeneity_score'],
                'intra_similarity': budget_het['avg_intra_similarity'],
                'inter_similarity': budget_het['avg_inter_similarity'],
                'assessment': self._assess_dimension_score(budget_het['heterogeneity_score'], '预算分配')
            },
            'purchase_pattern': {
                'score': pattern_het['heterogeneity_score'],
                'intra_similarity': pattern_het['avg_intra_similarity'],
                'inter_similarity': pattern_het['avg_inter_similarity'],
                'assessment': self._assess_dimension_score(pattern_het['heterogeneity_score'], '购买模式')
            }
        }
        
        # 优势分析
        if overall_score > 0.6:
            interpretation['strengths'].append('整体异质性表现良好，体现了较好的个性化水平')
        
        if comprehensive['clustering_quality'] > 0.3:
            interpretation['strengths'].append(f'家庭画像聚类质量良好(轮廓系数: {comprehensive["clustering_quality"]:.3f})')
        
        if validation['cluster_stability'] > 0.6:
            interpretation['strengths'].append('聚类结果稳定，重现性好')
        
        if validation['expectation_conformity'] > 0.6:
            interpretation['strengths'].append('消费行为符合家庭画像期望，具有合理性')
        
        # 找出表现最好的维度
        best_dimension = max([
            ('商品选择', product_het['heterogeneity_score']),
            ('预算分配', budget_het['heterogeneity_score']),
            ('购买模式', pattern_het['heterogeneity_score'])
        ], key=lambda x: x[1])
        
        if best_dimension[1] > 0.6:
            interpretation['strengths'].append(f'{best_dimension[0]}维度异质性表现突出({best_dimension[1]:.3f})')
        
        # 弱点和建议
        if overall_score < 0.5:
            interpretation['weaknesses'].append('整体异质性偏低，需要提升个性化程度')
            interpretation['recommendations'].append('优化LLM提示词，强化基于家庭画像的差异化决策')
        
        if comprehensive['clustering_quality'] < 0.2:
            interpretation['weaknesses'].append('家庭画像聚类质量较低，影响异质性评估准确性')
            interpretation['recommendations'].append('改进家庭画像特征工程，提高聚类区分度')
        
        if validation['cluster_stability'] < 0.4:
            interpretation['weaknesses'].append('聚类结果不够稳定')
            interpretation['recommendations'].append('检查特征选择和聚类参数设置')
        
        # 针对性建议
        weak_dimensions = [
            ('商品选择', product_het['heterogeneity_score']),
            ('预算分配', budget_het['heterogeneity_score']),
            ('购买模式', pattern_het['heterogeneity_score'])
        ]
        weak_dimensions.sort(key=lambda x: x[1])
        
        if weak_dimensions[0][1] < 0.4:
            dimension_name = weak_dimensions[0][0]
            if dimension_name == '商品选择':
                interpretation['recommendations'].append('增强商品推荐的个性化算法，考虑家庭特征对商品偏好的影响')
            elif dimension_name == '预算分配':
                interpretation['recommendations'].append('改进预算分配逻辑，更好地反映不同家庭的支出优先级')
            elif dimension_name == '购买模式':
                interpretation['recommendations'].append('优化购买行为模式生成，体现不同家庭的消费习惯差异')
        
        return interpretation
    
    def _assess_dimension_score(self, score: float, dimension: str) -> str:
        """评估维度得分"""
        if score >= 0.7:
            return f'{dimension}维度异质性优秀，同类家庭推荐一致，不同类家庭差异明显'
        elif score >= 0.5:
            return f'{dimension}维度异质性良好，有一定的个性化体现'
        elif score >= 0.3:
            return f'{dimension}维度异质性中等，个性化程度有限'
        else:
            return f'{dimension}维度异质性较低，缺乏明显的个性化差异'
    
    def _create_diagnostic_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """创建诊断摘要"""
        comprehensive = results['comprehensive']
        
        # 识别最相似的组间对和最不一致的组内对
        diagnostic = {
            'cluster_analysis': {
                'total_clusters': len(comprehensive['cluster_names']),
                'cluster_names': comprehensive['cluster_names'],
                'cluster_sizes': comprehensive['cluster_sizes'],
                'clustering_quality': comprehensive['clustering_quality']
            },
            'similarity_distribution': {
                'product_intra_vs_inter': {
                    'intra_avg': results['product_heterogeneity']['avg_intra_similarity'],
                    'inter_avg': results['product_heterogeneity']['avg_inter_similarity'],
                    'difference': results['product_heterogeneity']['avg_intra_similarity'] - results['product_heterogeneity']['avg_inter_similarity']
                },
                'budget_intra_vs_inter': {
                    'intra_avg': results['budget_heterogeneity']['avg_intra_similarity'],
                    'inter_avg': results['budget_heterogeneity']['avg_inter_similarity'],
                    'difference': results['budget_heterogeneity']['avg_intra_similarity'] - results['budget_heterogeneity']['avg_inter_similarity']
                }
            },
            'quality_metrics': results['validation'],
            'key_insights': []
        }
        
        # 关键洞察
        if comprehensive['heterogeneity_score'] > 0.6:
            diagnostic['key_insights'].append('系统成功实现了基于家庭画像的差异化推荐')
        
        if results['validation']['expectation_conformity'] > 0.7:
            diagnostic['key_insights'].append('推荐结果符合常识期望，具有较好的可解释性')
        
        if results['product_heterogeneity']['heterogeneity_score'] > results['budget_heterogeneity']['heterogeneity_score']:
            diagnostic['key_insights'].append('商品选择个性化程度高于预算分配个性化程度')
        else:
            diagnostic['key_insights'].append('预算分配个性化程度高于商品选择个性化程度')
        
        return diagnostic
