#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
属性基准管理器
用于收集和计算家庭属性的平均值，作为消费决策的社会参考
"""

import os
import json
import logger
from typing import List, Dict, Optional
from datetime import datetime

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class AttributeBenchmarkManager:
    """管理家庭属性基准数据，提供社会平均值作为参考"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化基准管理器
        
        Args:
            output_dir: 家庭状态文件所在的输出目录
        """
        self.output_dir = output_dir
    
    def collect_family_attributes(self, family_ids: List[str], target_month: Optional[int] = None) -> List[Dict]:
        """
        收集多个家庭的属性数据
        
        Args:
            family_ids: 家庭ID列表
            target_month: 目标月份（如果为None，则读取当前状态）
            
        Returns:
            属性数据列表
        """
        # ========================================
        # 🔧 调试：打印属性收集的详细信息
        # ========================================
        # logger.info(f"🔍 开始收集家庭属性数据:")
        # logger.info(f"   - 输出目录: {self.output_dir}")
        # logger.info(f"   - 家庭ID列表: {family_ids}")
        # logger.info(f"   - 目标月份: {target_month if target_month is not None else '当前状态'}")
        # logger.info(f"   - 输出目录是否存在: {os.path.exists(self.output_dir)}")
        
        all_attributes = []
        files_not_found = []
        files_no_data = []
        files_success = []
        
        for family_id in family_ids:
            state_file = os.path.join(self.output_dir, f"family_{family_id}", "family_state.json")
            
            if not os.path.exists(state_file):
                logger.debug(f"   ❌ 家庭 {family_id} 状态文件不存在: {state_file}")
                files_not_found.append(family_id)
                continue
            
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ========================================
                # 🔧 修复：选择数据源 - 适配新的文件格式
                # 问题：history 在 current_state.nutrition_reference.history 中，且是数组格式
                # 解决：从数组中查找指定月份的数据
                # ========================================
                if target_month is not None:
                    # 从历史记录读取特定月份
                    # 新格式：history 在 nutrition_reference 中，是数组 [{month: 1, supply: {...}, ...}]
                    nutrition_ref = data.get('current_state', {}).get('nutrition_reference', {})
                    history_list = nutrition_ref.get('history', [])
                    
                    logger.debug(f"   🔍 家庭 {family_id} 查找月份 {target_month}，历史记录数组长度: {len(history_list)}")
                    
                    # 在数组中查找指定月份
                    state_data = None
                    for record in history_list:
                        if record.get('month') == target_month:
                            # 找到了！重构数据格式以匹配期望的结构
                            # 使用 supply 作为营养库存（代表该月购买的食物供给）
                            state_data = {
                                'nutrition_stock': {
                                    attr: record.get('supply', {}).get(attr, 0)
                                    for attr in ['carbohydrate_g', 'protein_g', 'fat_g', 'water_g', 'vitamin_level', 'mineral_level']
                                },
                                'life_quality': data.get('current_state', {}).get('life_quality', {}),
                                'non_food_inventory': []  # 简化处理
                            }
                            logger.debug(f"   ✅ 家庭 {family_id} 从历史记录读取月份 {target_month} (supply数据)")
                            break
                    
                    if state_data is None:
                        available_months = [r.get('month') for r in history_list]
                        logger.debug(f"   ⚠️ 家庭 {family_id} 没有月份 {target_month} 的历史数据 (可用月份: {available_months})")
                        files_no_data.append(f"{family_id}(月份{target_month})")
                        continue
                else:
                    # 读取当前状态
                    state_data = data.get('current_state', {})
                    current_month = data.get('current_month', 0)
                    logger.debug(f"   ✅ 家庭 {family_id} 读取当前状态 (当前月份: {current_month})")
                
                # 提取基础属性
                attributes = {
                    'family_id': family_id,
                    'family_size': data.get('family_size', 1),
                    'nutrition_stock': state_data.get('nutrition_stock', {}),
                    'life_quality': state_data.get('life_quality', {}),
                    'non_food_inventory': state_data.get('non_food_inventory', [])
                }
                
                all_attributes.append(attributes)
                files_success.append(family_id)
                
            except Exception as e:
                logger.warning(f"   ❌ 读取家庭 {family_id} 属性失败: {e}")
                import traceback
                logger.debug(f"   详细错误: {traceback.format_exc()}")
                continue
        
        # ========================================
        # 🔧 调试：打印收集结果汇总
        # ========================================
        logger.info(f"📊 属性收集结果汇总:")
        logger.info(f"   ✅ 成功: {len(files_success)} 个家庭 {files_success}")
        if files_not_found:
            logger.info(f"   ❌ 文件不存在: {len(files_not_found)} 个家庭 {files_not_found}")
        if files_no_data:
            logger.info(f"   ⚠️ 无数据: {len(files_no_data)} 个 {files_no_data}")
        
        logger.info(f"成功收集 {len(all_attributes)} 个家庭的属性数据")
        return all_attributes
    
    def calculate_benchmark(self, all_attributes: List[Dict], exclude_family_id: Optional[str] = None) -> Optional[Dict]:
        """
        计算属性基准（平均值）
        
        Args:
            all_attributes: 所有家庭的属性数据
            exclude_family_id: 排除的家庭ID（通常是当前家庭自己）
            
        Returns:
            基准数据字典，如果没有有效数据则返回None
        """
        # 过滤数据
        if exclude_family_id:
            filtered_attributes = [attr for attr in all_attributes if attr['family_id'] != exclude_family_id]
        else:
            filtered_attributes = all_attributes
        
        if not filtered_attributes:
            logger.warning("没有可用的家庭属性数据来计算基准")
            return None
        
        # 计算营养库存平均值
        nutrition_keys = ['carbohydrate_g', 'protein_g', 'fat_g', 'water_g', 'vitamin_level', 'mineral_level']
        nutrition_avg = {}
        
        for key in nutrition_keys:
            values = [attr['nutrition_stock'].get(key, 0) for attr in filtered_attributes]
            nutrition_avg[key] = sum(values) / len(values) if values else 0
        
        # 计算生活品质平均值
        quality_keys = ['functional_satisfaction', 'aesthetic_satisfaction', 'symbolic_satisfaction', 
                       'social_satisfaction', 'growth_satisfaction']
        quality_avg = {}
        
        for key in quality_keys:
            values = [attr['life_quality'].get(key, 0) for attr in filtered_attributes]
            quality_avg[key] = sum(values) / len(values) if values else 0
        
        # 计算非食物商品平均数量
        inventory_counts = [len(attr['non_food_inventory']) for attr in filtered_attributes]
        inventory_avg = sum(inventory_counts) / len(inventory_counts) if inventory_counts else 0
        
        # 计算人均指标
        per_capita_nutrition = {}
        for key in nutrition_keys:
            total = sum(attr['nutrition_stock'].get(key, 0) / attr['family_size'] 
                       for attr in filtered_attributes)
            per_capita_nutrition[key] = total / len(filtered_attributes) if filtered_attributes else 0
        
        per_capita_quality = {}
        for key in quality_keys:
            total = sum(attr['life_quality'].get(key, 0) / attr['family_size'] 
                       for attr in filtered_attributes)
            per_capita_quality[key] = total / len(filtered_attributes) if filtered_attributes else 0
        
        per_capita_inventory = sum(len(attr['non_food_inventory']) / attr['family_size'] 
                                   for attr in filtered_attributes) / len(filtered_attributes)
        
        # 构建基准数据
        benchmark = {
            'nutrition_stock_avg': nutrition_avg,
            'life_quality_avg': quality_avg,
            'non_food_inventory_avg': inventory_avg,
            'per_capita': {
                'nutrition_stock': per_capita_nutrition,
                'life_quality': per_capita_quality,
                'non_food_inventory': per_capita_inventory
            },
            'statistics': {
                'sample_size': len(filtered_attributes),
                'excluded_family': exclude_family_id,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"计算基准完成: 样本数={len(filtered_attributes)}, "
                   f"平均营养库存={nutrition_avg.get('carbohydrate_g', 0):.0f}g碳水, "
                   f"平均商品数={inventory_avg:.1f}个")
        
        return benchmark
    
    def get_benchmark(self, family_ids: List[str], exclude_family_id: Optional[str] = None, 
                     target_month: Optional[int] = None) -> Optional[Dict]:
        """
        获取基准数据（一站式接口）
        
        Args:
            family_ids: 家庭ID列表
            exclude_family_id: 排除的家庭ID
            target_month: 目标月份（None表示当前状态）
            
        Returns:
            基准数据
        """
        # 收集属性
        all_attributes = self.collect_family_attributes(family_ids, target_month)
        
        if not all_attributes:
            return None
        
        # 计算基准
        benchmark = self.calculate_benchmark(all_attributes, exclude_family_id)
        
        return benchmark
    
    def format_benchmark_for_prompt(self, benchmark: Optional[Dict], current_family_attrs: Dict) -> str:
        """
        将基准数据格式化为LLM prompt的一部分
        
        Args:
            benchmark: 基准数据
            current_family_attrs: 当前家庭属性
            
        Returns:
            格式化的prompt文本
        """
        if not benchmark:
            return ""
        
        family_size = current_family_attrs.get('family_size', 1)
        current_nutrition = current_family_attrs.get('nutrition_stock', {})
        current_quality = current_family_attrs.get('life_quality', {})
        current_items = len(current_family_attrs.get('non_food_inventory', []))
        
        # 计算当前家庭的人均值
        per_capita_current = {
            'carbohydrate_g': current_nutrition.get('carbohydrate_g', 0) / family_size,
            'protein_g': current_nutrition.get('protein_g', 0) / family_size,
            'fat_g': current_nutrition.get('fat_g', 0) / family_size,
            'water_g': current_nutrition.get('water_g', 0) / family_size,
            'non_food_items': current_items / family_size
        }
        
        # 基准人均值
        benchmark_per_capita = benchmark['per_capita']
        
        prompt = f"""
📊 COMMUNITY BENCHMARK (Based on {benchmark['statistics']['sample_size']} other families):

Average Nutrition Stock (per capita):
  • Carbohydrate: {benchmark_per_capita['nutrition_stock']['carbohydrate_g']:.0f}g
  • Protein: {benchmark_per_capita['nutrition_stock']['protein_g']:.0f}g
  • Fat: {benchmark_per_capita['nutrition_stock']['fat_g']:.0f}g
  • Water: {benchmark_per_capita['nutrition_stock']['water_g']:.0f}g

Average Non-food Items (per capita): {benchmark_per_capita['non_food_inventory']:.1f} items

YOUR FAMILY'S POSITION (per capita):
  • Carbohydrate: {per_capita_current['carbohydrate_g']:.0f}g ({self._compare_value(per_capita_current['carbohydrate_g'], benchmark_per_capita['nutrition_stock']['carbohydrate_g'])})
  • Protein: {per_capita_current['protein_g']:.0f}g ({self._compare_value(per_capita_current['protein_g'], benchmark_per_capita['nutrition_stock']['protein_g'])})
  • Fat: {per_capita_current['fat_g']:.0f}g ({self._compare_value(per_capita_current['fat_g'], benchmark_per_capita['nutrition_stock']['fat_g'])})
  • Water: {per_capita_current['water_g']:.0f}g ({self._compare_value(per_capita_current['water_g'], benchmark_per_capita['nutrition_stock']['water_g'])})
  • Non-food items: {per_capita_current['non_food_items']:.1f} ({self._compare_value(per_capita_current['non_food_items'], benchmark_per_capita['non_food_inventory'])})

💡 RECOMMENDATION: 
{self._generate_recommendation(per_capita_current, benchmark_per_capita)}
"""
        return prompt
    
    def _compare_value(self, current: float, average: float) -> str:
        """比较当前值与平均值"""
        if average == 0:
            return "N/A"
        
        ratio = current / average
        if ratio < 0.5:
            return "⚠️ WELL BELOW average"
        elif ratio < 0.8:
            return "⬇️ Below average"
        elif ratio < 1.2:
            return "✅ Similar to average"
        elif ratio < 1.5:
            return "⬆️ Above average"
        else:
            return "📈 Well above average"
    
    def _generate_recommendation(self, current: Dict, benchmark: Dict) -> str:
        """生成消费建议"""
        recommendations = []
        
        # 检查营养缺口
        nutrition_deficit = []
        if current['carbohydrate_g'] < benchmark['nutrition_stock']['carbohydrate_g'] * 0.5:
            nutrition_deficit.append("carbohydrate")
        if current['protein_g'] < benchmark['nutrition_stock']['protein_g'] * 0.5:
            nutrition_deficit.append("protein")
        if current['fat_g'] < benchmark['nutrition_stock']['fat_g'] * 0.5:
            nutrition_deficit.append("fat")
        
        if nutrition_deficit:
            recommendations.append(f"Priority: Increase food purchases (especially {', '.join(nutrition_deficit)}) to catch up with community levels.")
        
        # 检查非食物商品
        if current['non_food_items'] < benchmark['non_food_inventory'] * 0.7:
            recommendations.append("Consider purchasing more non-food items for life quality improvement.")
        elif current['non_food_items'] > benchmark['non_food_inventory'] * 1.5:
            recommendations.append("Your non-food inventory is well above average. Focus on maintaining rather than expanding.")
        
        if not recommendations:
            recommendations.append("Your consumption levels are well-balanced with the community average.")
        
        return " ".join(recommendations)

