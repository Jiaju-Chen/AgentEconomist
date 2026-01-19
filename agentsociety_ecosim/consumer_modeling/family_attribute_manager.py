"""
家庭属性系统 v4.0
基于食物/非食物分类的新属性体系 + 月度清零 + 社会比较

核心功能：
1. 营养库存管理（食物）- 月度清零
2. 营养参考数据（历史追踪与决策支持）
3. 生活品质管理（非食物）- 社会比较模式
4. 商品清单管理
5. 月度更新逻辑
6. 文件保存/加载

v3.0 特性（食物部分）：
- 食物营养每月清零，不跨月累积
- 记录每月供给、消耗、结余、满足率
- 追踪连续亏损月数
- 保留最近6个月历史数据
- 识别严重不足的营养素
- 提供历史趋势分析

v4.0 新特性（非食物部分）：
- 非食物商品每月计算供给，不累积
- 通过社会比较计算生活品质得分（百分位排名）
- 记录每月商品供给、得分、排名信息
- 商品按有效期管理，过期自动清理
- 文件保存只保留统计数据，不保存商品清单
- 支持多家庭协同的社会比较分析
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from copy import deepcopy
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class FamilyAttributeSystem:
    """家庭属性系统 - 管理营养库存和生活品质"""
    
    def __init__(self, family_id: str, family_size: int, config_file: str = None):
        """
        初始化属性系统
        
        Args:
            family_id: 家庭ID
            family_size: 家庭规模
            config_file: 配置文件路径
        """
        self.family_id = family_id
        self.family_size = family_size
        self.current_month = 0
        
        # 加载配置（使用相对路径）
        if config_file:
            self.config_file = config_file
        else:
            # 默认配置文件路径（相对于当前文件）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_file = os.path.join(current_dir, "family_attribute_config.json")
        self.config = self._load_config()
        
        # 加载商品属性映射
        self.product_attributes = None
        self._load_product_attributes()
        
        # 营养库存（食物系统）- 初始化为0
        self.nutrition_stock = {
            "carbohydrate_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "water_g": 0.0,
            "vitamin_level": 0.0,
            "mineral_level": 0.0
        }
        
        # 营养参考数据（v3.0新增 - 月度清零模式）
        self.nutrition_reference = {
            "last_month_supply": {
                "carbohydrate_g": 0.0,
                "protein_g": 0.0,
                "fat_g": 0.0,
                "water_g": 0.0,
                "vitamin_level": 0.0,
                "mineral_level": 0.0
            },
            "last_month_consumption": {
                "carbohydrate_g": 0.0,
                "protein_g": 0.0,
                "fat_g": 0.0,
                "water_g": 0.0,
                "vitamin_level": 0.0,
                "mineral_level": 0.0
            },
            "last_month_balance": {
                "carbohydrate_g": 0.0,
                "protein_g": 0.0,
                "fat_g": 0.0,
                "water_g": 0.0,
                "vitamin_level": 0.0,
                "mineral_level": 0.0
            },
            "deficit_streak": {
                "carbohydrate_g": 0,
                "protein_g": 0,
                "fat_g": 0,
                "water_g": 0,
                "vitamin_level": 0,
                "mineral_level": 0
            },
            "history": []  # 保留最近6个月的历史记录
        }
        
        # 生活品质（非食物系统）
        self.life_quality = {
            "functional_satisfaction": 0.0,
            "aesthetic_satisfaction": 0.0,
            "symbolic_satisfaction": 0.0,
            "social_satisfaction": 0.0,
            "growth_satisfaction": 0.0
        }
        
        # 生活品质参考数据（v4.0新增：社会比较系统）
        self.life_quality_reference = {
            "current_month_supply": {
                "functional_satisfaction": 0.0,
                "aesthetic_satisfaction": 0.0,
                "symbolic_satisfaction": 0.0,
                "social_satisfaction": 0.0,
                "growth_satisfaction": 0.0
            },
            "last_month_supply": {
                "functional_satisfaction": 0.0,
                "aesthetic_satisfaction": 0.0,
                "symbolic_satisfaction": 0.0,
                "social_satisfaction": 0.0,
                "growth_satisfaction": 0.0
            },
            "last_month_score": {
                "functional_satisfaction": 0.0,
                "aesthetic_satisfaction": 0.0,
                "symbolic_satisfaction": 0.0,
                "social_satisfaction": 0.0,
                "growth_satisfaction": 0.0
            },
            "last_month_ranking": {
                "functional_satisfaction": {"percentile": 0, "rank": 0, "total": 0},
                "aesthetic_satisfaction": {"percentile": 0, "rank": 0, "total": 0},
                "symbolic_satisfaction": {"percentile": 0, "rank": 0, "total": 0},
                "social_satisfaction": {"percentile": 0, "rank": 0, "total": 0},
                "growth_satisfaction": {"percentile": 0, "rank": 0, "total": 0}
            },
            "history": []  # 保留最近6个月的历史记录
        }
        
        # 非食物商品清单
        self.non_food_inventory = []
        
        logger.info(f"✅ 家庭属性系统初始化完成: family_id={family_id}, size={family_size}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"配置文件不存在: {self.config_file}")
            return {}
    
    def _load_product_attributes(self):
        """加载商品属性映射文件"""
        product_file = self.config.get("product_attribute_file", "")
        if not product_file:
            # 默认路径：相对于配置文件所在目录
            config_dir = os.path.dirname(self.config_file)
            product_file = os.path.join(config_dir, "household_data", "averaged_mapping_ordered.json")
        elif not os.path.isabs(product_file):
            # 如果是相对路径，相对于配置文件所在目录
            config_dir = os.path.dirname(self.config_file)
            product_file = os.path.join(config_dir, product_file)
        
        if os.path.exists(product_file):
            try:
                with open(product_file, 'r', encoding='utf-8') as f:
                    self.product_attributes = json.load(f)
                logger.info(f"✅ 商品属性文件加载成功: {product_file}")
            except Exception as e:
                logger.error(f"❌ 加载商品属性文件失败: {e}")
                self.product_attributes = {"product_mappings": []}
        else:
            logger.warning(f"⚠️ 商品属性文件不存在: {product_file}")
            self.product_attributes = {"product_mappings": []}
    
    # ========== 商品处理接口 ==========
    
    def get_product_attributes(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        获取商品的原始属性（新格式）
        
        Args:
            product_id: 商品ID
            
        Returns:
            商品属性字典或None
            {
                "is_food": bool,
                "nutrition_supply": {...} or None,
                "satisfaction_attributes": {...} or None,
                "duration_months": int (默认12)
            }
        """
        if not self.product_attributes:
            return None
        
        for product in self.product_attributes.get('product_mappings', []):
            if product.get('product_id') == product_id:
                return {
                    "is_food": product.get('is_food', False),
                    "nutrition_supply": product.get('nutrition_supply',None),
                    "satisfaction_attributes": product.get('satisfaction_attributes',None),
                    "duration_months": product.get('duration_months',None)
                }
        
        return None
    
    def add_purchased_products(self, products_list: List[Dict[str, Any]]):
        """
        添加购买的商品
        
        Args:
            products_list: 商品列表
                [{
                    "product_id": str,
                    "product_name": str,
                    "quantity": float,
                    ...
                }]
        """
        logger.info(f"📦 处理 {len(products_list)} 个购买的商品")
        
        food_count = 0
        non_food_count = 0
        
        for item in products_list:
            product_id = item.get('product_id', '')
            product_name = item.get('product_name', item.get('name', ''))
            quantity = float(item.get('quantity', 1))
            
            if not product_id or quantity <= 0:
                continue
            
            # 获取商品属性
            product_attrs = item.get('attributes')
            if not product_attrs:
                inline_attrs = {}
                if 'is_food' in item:
                    inline_attrs['is_food'] = item.get('is_food')
                if item.get('nutrition_supply'):
                    inline_attrs['nutrition_supply'] = item.get('nutrition_supply')
                if item.get('satisfaction_attributes'):
                    inline_attrs['satisfaction_attributes'] = item.get('satisfaction_attributes')
                if item.get('duration_months') is not None:
                    inline_attrs['duration_months'] = item.get('duration_months')
                if inline_attrs:
                    product_attrs = inline_attrs
            if not product_attrs:
                product_attrs = self.get_product_attributes(product_id)
            if not product_attrs:
                logger.warning(f"⚠️ 商品属性未找到: {product_id}")
                continue
            
            is_food = product_attrs.get('is_food', False)
            
            if is_food:
                # 食物：立即转换为营养值
                self._add_food_nutrition(product_attrs, quantity)
                food_count += 1
            else:
                # 非食物：添加到清单
                self._add_non_food_item(product_id, product_name, product_attrs, quantity)
                non_food_count += 1
        
        logger.info(f"✅ 商品处理完成: 食物{food_count}个, 非食物{non_food_count}个")
    
    def _add_food_nutrition(self, product_attrs: Dict, quantity: float):
        """添加食物的营养值"""
        nutrition = product_attrs.get('nutrition_supply', {})
        
        # 累加所有营养库存（统一逻辑）
        self.nutrition_stock['carbohydrate_g'] += nutrition.get('carbohydrate_g', 0) * quantity
        self.nutrition_stock['protein_g'] += nutrition.get('protein_g', 0) * quantity
        self.nutrition_stock['fat_g'] += nutrition.get('fat_g', 0) * quantity
        self.nutrition_stock['water_g'] += nutrition.get('water_g', 0) * quantity
        
        # 维生素/矿物质也使用累加逻辑（修复：改为+=，并乘以quantity）
        vitamin_index = nutrition.get('vitamin_index', 0)
        mineral_index = nutrition.get('mineral_index', 0)
        
        self.nutrition_stock['vitamin_level'] += vitamin_index * 100 * quantity
        self.nutrition_stock['mineral_level'] += mineral_index * 100 * quantity
    
    def _add_non_food_item(self, product_id: str, product_name: str, product_attrs: Dict, quantity: int):
        """添加非食物商品到清单"""
        satisfaction_attrs = product_attrs.get('satisfaction_attributes', {})
        duration = product_attrs.get('duration_months', 12)
        
        # 每个商品添加quantity次（独立追踪）
        for _ in range(int(quantity)):
            self.non_food_inventory.append({
                "product_id": product_id,
                "product_name": product_name,
                "purchase_month": self.current_month,
                "duration_total": duration,
                "duration_left": duration,
                "satisfaction_attributes": deepcopy(satisfaction_attrs)
            })
    
    # ========== 月度更新接口 ==========
    
    def monthly_update(self, new_month: int, all_families: List['FamilyAttributeSystem'] = None):
        """
        月度更新逻辑（v4.0更新）
        
        Args:
            new_month: 新的月份数
            all_families: 所有家庭的属性系统列表（用于非食物部分的社会比较，可选）
        """
        logger.info(f"📅 开始月度更新: 月份 {self.current_month} → {new_month}")
        
        self.current_month = new_month
        
        # 1. 消耗营养（食物部分：月度清零）
        self._consume_nutrition()
        
        # 2. 更新生活品质（非食物部分：社会比较模式）
        self._update_life_quality_monthly(all_families)
        
        logger.info(f"✅ 月度更新完成")
    
    def _consume_nutrition(self):
        """
        消耗营养（v3.0月度清零方案）
        
        流程：
        1. 记录本月供给量（当前库存）
        2. 计算消耗标准
        3. 计算结余和满足率
        4. 更新参考数据和连续亏损计数
        5. 保存历史记录
        6. 清零营养库存
        """
        consumption_config = self.config.get('nutrition_consumption', {})
        nutrition_ref_config = self.config.get('nutrition_reference', {})
        history_months = nutrition_ref_config.get('history_months', 6)
        
        # 1. 记录本月供给量（当月购买的食物提供的营养）
        supply = deepcopy(self.nutrition_stock)
        
        # 2. 计算消耗标准（根据家庭规模）
        consumption = {}
        consumption['carbohydrate_g'] = consumption_config.get('carbohydrate_g_per_month', 9000) * self.family_size
        consumption['protein_g'] = consumption_config.get('protein_g_per_month', 1800) * self.family_size
        consumption['fat_g'] = consumption_config.get('fat_g_per_month', 2100) * self.family_size
        consumption['water_g'] = consumption_config.get('water_g_per_month', 60000) * self.family_size
        consumption['vitamin_level'] = consumption_config.get('vitamin_decay_per_month', 30) * self.family_size
        consumption['mineral_level'] = consumption_config.get('mineral_decay_per_month', 30) * self.family_size
        
        # 3. 计算结余和满足率
        balance = {}
        satisfaction_rate = {}
        deficit_nutrients = []
        
        for nutrient in self.nutrition_stock.keys():
            supply_value = supply[nutrient]
            consumption_value = consumption[nutrient]
            
            # 计算结余（供给 - 消耗）
            balance[nutrient] = supply_value - consumption_value
            
            # 计算满足率（%）
            if consumption_value > 0:
                satisfaction_rate[nutrient] = (supply_value / consumption_value) * 100
            else:
                satisfaction_rate[nutrient] = 100.0
            
            # 记录亏损的营养素
            if balance[nutrient] < 0:
                deficit_nutrients.append(nutrient)
        
        # 4. 更新参考数据
        self.nutrition_reference['last_month_supply'] = supply
        self.nutrition_reference['last_month_consumption'] = consumption
        self.nutrition_reference['last_month_balance'] = balance
        
        # 更新连续亏损计数
        for nutrient in self.nutrition_stock.keys():
            if balance[nutrient] < 0:
                self.nutrition_reference['deficit_streak'][nutrient] += 1
            else:
                self.nutrition_reference['deficit_streak'][nutrient] = 0
        
        # 5. 保存历史记录
        monthly_record = {
            "month": self.current_month,
            "supply": supply,
            "consumption": consumption,
            "balance": balance,
            "satisfaction_rate": satisfaction_rate
        }
        
        self.nutrition_reference['history'].append(monthly_record)
        
        # 只保留最近N个月的历史
        if len(self.nutrition_reference['history']) > history_months:
            self.nutrition_reference['history'] = self.nutrition_reference['history'][-history_months:]
        
        # 6. 🔥 清零营养库存（月度清零的核心）
        for nutrient in self.nutrition_stock.keys():
            self.nutrition_stock[nutrient] = 0.0
        
        # 日志输出
        avg_satisfaction = sum(satisfaction_rate.values()) / len(satisfaction_rate) if satisfaction_rate else 0
        if deficit_nutrients:
            logger.warning(
                f"⚠️ 家庭 {self.family_id} 月份 {self.current_month} 营养不足: "
                f"{len(deficit_nutrients)}个营养素亏损, 平均满足率: {avg_satisfaction:.1f}%"
            )
        else:
            logger.info(
                f"✅ 家庭 {self.family_id} 月份 {self.current_month} 营养充足, "
                f"平均满足率: {avg_satisfaction:.1f}%"
            )
        
        logger.info(f"🍎 营养消耗完成，库存已清零")
    
    # ========== 非食物部分：月度更新（v4.0 社会比较模式）==========
    
    def _calculate_monthly_supply(self) -> Dict[str, float]:
        """
        计算本月商品提供的满足度供给
        
        Returns:
            各维度的月度供给总和
        """
        supply = {
            "functional_satisfaction": 0.0,
            "aesthetic_satisfaction": 0.0,
            "symbolic_satisfaction": 0.0,
            "social_satisfaction": 0.0,
            "growth_satisfaction": 0.0
        }
        
        # 遍历库存中的所有未过期商品
        for item in self.non_food_inventory:
            if item.get('duration_left', 0) <= 0:
                continue  # 跳过已过期的
            
            attrs = item.get('satisfaction_attributes', {})
            
            for attr_name, attr_data in attrs.items():
                if isinstance(attr_data, dict):
                    monthly_supply = attr_data.get('monthly_supply', 0)
                    
                    # 映射到生活品质维度（去掉_utility后缀）
                    dimension = attr_name.replace('_utility', '_satisfaction')
                    
                    if dimension in supply:
                        supply[dimension] += monthly_supply
        
        return supply
    
    def _calculate_social_comparison_score(
        self, 
        my_supply: Dict[str, float],
        all_families_supply: List[Dict[str, float]]
    ) -> tuple:
        """
        通过社会比较计算生活品质得分
        
        Args:
            my_supply: 本家庭的供给
            all_families_supply: 所有家庭的供给列表
            
        Returns:
            (score, ranking) - 各维度得分(0-100)和排名信息
        """
        score = {}
        ranking = {}
        
        for dimension in my_supply.keys():
            my_value = my_supply[dimension]
            
            # 收集所有家庭的该维度供给
            all_values = [f.get(dimension, 0) for f in all_families_supply]
            
            if len(all_values) == 0:
                # 没有参考数据，使用默认值
                percentile = 50
                rank = 0
                avg_value = 0
            else:
                all_values_sorted = sorted(all_values)
                
                # 计算排名（比我低的家庭数量）
                rank = sum(1 for v in all_values if v < my_value)
                
                # 计算百分位
                percentile = int((rank / len(all_values)) * 100)
                
                # 计算平均值
                avg_value = sum(all_values) / len(all_values)
            
            # 得分 = 百分位
            score[dimension] = float(percentile)
            
            ranking[dimension] = {
                "percentile": percentile,
                "rank": rank + 1,  # 从1开始
                "total": len(all_values),
                "my_supply": round(my_value, 2),
                "avg_supply": round(avg_value, 2)
            }
        
        return score, ranking
    
    def _update_life_quality_monthly(self, all_families: List['FamilyAttributeSystem'] = None):
        """
        月度更新生活品质（非食物部分）- v4.0 社会比较模式
        
        核心逻辑：
        1. 计算本月商品供给
        2. 如果有上月数据 + 有足够的家庭数据 → 社会比较计算得分
        3. 如果无上月数据或家庭数不足 → 直接使用供给值转换
        4. 记录历史数据
        5. 更新 last_month 数据
        6. 减少商品有效期，清空过期商品
        
        Args:
            all_families: 所有家庭的属性系统列表（用于社会比较）
        """
        # 1. 计算本月商品供给
        current_supply = self._calculate_monthly_supply()
        self.life_quality_reference['current_month_supply'] = deepcopy(current_supply)
        
        # 2. 判断是否有上月数据
        has_last_month_data = any(
            v > 0.001 for v in self.life_quality_reference['last_month_supply'].values()
        )
        
        comparison_mode = "direct"
        
        if has_last_month_data and all_families and len(all_families) > 1:
            # 有上月数据 → 尝试进行社会比较
            
            # 收集所有家庭的上月供给数据
            all_families_supply = []
            for family in all_families:
                if hasattr(family, 'life_quality_reference'):
                    last_supply = family.life_quality_reference.get('last_month_supply', {})
                    if any(v > 0.001 for v in last_supply.values()):
                        all_families_supply.append(last_supply)
            
            # 判断是否有足够的家庭数据
            min_families = self.config.get('life_quality_reference', {}).get('min_families_for_comparison', 3)
            
            if len(all_families_supply) >= min_families:
                # 社会比较计算得分
                score, ranking = self._calculate_social_comparison_score(
                    current_supply, 
                    all_families_supply
                )
                
                # 更新 life_quality
                self.life_quality = score
                
                # 记录排名信息
                self.life_quality_reference['last_month_ranking'] = ranking
                
                comparison_mode = "social"
                
                logger.info(
                    f"🏆 家庭 {self.family_id} 月份 {self.current_month} "
                    f"使用社会比较模式，参考了 {len(all_families_supply)} 个家庭"
                )
            else:
                # 家庭数不足，直接使用供给值
                multiplier = self.config.get('life_quality_reference', {}).get('supply_to_score_multiplier', 10)
                self.life_quality = {k: min(100, v * multiplier) for k, v in current_supply.items()}
                
                logger.info(
                    f"📊 家庭 {self.family_id} 月份 {self.current_month} "
                    f"家庭数不足({len(all_families_supply)}个)，直接使用供给值"
                )
        else:
            # 无上月数据（第一个月）→ 直接使用供给值
            multiplier = self.config.get('life_quality_reference', {}).get('supply_to_score_multiplier', 10)
            self.life_quality = {k: min(100, v * multiplier) for k, v in current_supply.items()}
            
            logger.info(
                f"📊 家庭 {self.family_id} 月份 {self.current_month} "
                f"首月或无上月数据，直接使用供给值"
            )
        
        # 3. 记录历史
        monthly_record = {
            "month": self.current_month,
            "supply": deepcopy(current_supply),
            "score": deepcopy(self.life_quality),
            "ranking": deepcopy(self.life_quality_reference.get('last_month_ranking', {})),
            "comparison_mode": comparison_mode
        }
        
        self.life_quality_reference['history'].append(monthly_record)
        
        # 保留最近N个月
        history_months = self.config.get('life_quality_reference', {}).get('history_months', 6)
        if len(self.life_quality_reference['history']) > history_months:
            self.life_quality_reference['history'] = \
                self.life_quality_reference['history'][-history_months:]
        
        # 4. 更新 last_month 数据
        self.life_quality_reference['last_month_supply'] = deepcopy(current_supply)
        self.life_quality_reference['last_month_score'] = deepcopy(self.life_quality)
        
        # 5. 减少有效期，清空过期商品
        for item in self.non_food_inventory:
            item['duration_left'] -= 1
        
        before_count = len(self.non_food_inventory)
        self.non_food_inventory = [
            item for item in self.non_food_inventory 
            if item.get('duration_left', 0) > 0
        ]
        after_count = len(self.non_food_inventory)
        
        expired = before_count - after_count
        if expired > 0:
            logger.info(f"🗑️  移除 {expired} 个过期商品")
        
        logger.info(f"🏠 生活品质更新完成")
    
    def _get_family_coefficient(self) -> float:
        """获取家庭规模系数（共享效应）"""
        coefficients = self.config.get('family_size_coefficients', {})
        
        if self.family_size >= 6:
            return coefficients.get('6+', 0.6)
        else:
            return coefficients.get(str(self.family_size), 1.0)
    
    # ========== 查询接口 ==========
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前状态快照
        
        Returns:
            {
                "nutrition_stock": {...},
                "life_quality": {...},
                "non_food_inventory": [...],
                "current_month": int
            }
        """
        return {
            "nutrition_stock": deepcopy(self.nutrition_stock),
            "life_quality": deepcopy(self.life_quality),
            "non_food_inventory": deepcopy(self.non_food_inventory),
            "current_month": self.current_month
        }
    
    def calculate_needs(self) -> Dict[str, Any]:
        """
        计算当月需求
        
        Returns:
            {
                "nutrition_needs": {...},
                "quality_needs": {...}
            }
        """
        consumption = self.config.get('nutrition_consumption', {})
        decay = self.config.get('life_quality_decay', {})
        family_coef = self._get_family_coefficient()
        
        return {
            "nutrition_needs": {
                "carbohydrate_g": consumption.get('carbohydrate_g_per_month', 9000) * self.family_size,
                "protein_g": consumption.get('protein_g_per_month', 1800) * self.family_size,
                "fat_g": consumption.get('fat_g_per_month', 2100) * self.family_size,
                "water_g": consumption.get('water_g_per_month', 60000) * self.family_size,
                "vitamin_level": consumption.get('vitamin_decay_per_month', 30) * self.family_size,
                "mineral_level": consumption.get('mineral_decay_per_month', 30) * self.family_size
            },
            "quality_needs": {
                dimension: decay.get(dimension, 0) * family_coef
                for dimension in self.life_quality.keys()
            }
        }
    
    def get_nutrition_reference(self) -> Dict[str, Any]:
        """
        获取营养参考数据（v3.0新增）
        
        Returns:
            {
                "last_month_balance": {...},        # 上月结余
                "deficit_streak": {...},            # 连续亏损月数
                "avg_satisfaction_rate": float,     # 平均满足率
                "critical_nutrients": [...],        # 严重不足的营养素列表
                "history_summary": {...},           # 历史趋势分析
                "history_length": int               # 历史记录长度
            }
        """
        nutrition_ref_config = self.config.get('nutrition_reference', {})
        deficit_alert_threshold = nutrition_ref_config.get('deficit_alert_threshold', 3)
        critical_satisfaction_rate = nutrition_ref_config.get('critical_satisfaction_rate', 50)
        
        result = {
            "last_month_balance": deepcopy(self.nutrition_reference.get('last_month_balance', {})),
            "deficit_streak": deepcopy(self.nutrition_reference.get('deficit_streak', {})),
            "history_length": len(self.nutrition_reference.get('history', []))
        }
        
        # 计算平均满足率
        history = self.nutrition_reference.get('history', [])
        if history:
            latest = history[-1]
            satisfaction_rates = latest.get('satisfaction_rate', {})
            if satisfaction_rates:
                result['avg_satisfaction_rate'] = sum(satisfaction_rates.values()) / len(satisfaction_rates)
            else:
                result['avg_satisfaction_rate'] = 0.0
        else:
            result['avg_satisfaction_rate'] = 0.0
        
        # 识别严重不足的营养素
        critical_nutrients = []
        deficit_streak = self.nutrition_reference.get('deficit_streak', {})
        last_month_balance = self.nutrition_reference.get('last_month_balance', {})
        
        for nutrient, streak in deficit_streak.items():
            balance = last_month_balance.get(nutrient, 0)
            
            # 判断是否严重不足：连续亏损>=阈值 或 满足率<临界值
            is_critical = False
            reason_parts = []
            
            if streak >= deficit_alert_threshold:
                is_critical = True
                reason_parts.append(f"连续{streak}个月不足")
            
            if history and satisfaction_rates:
                rate = satisfaction_rates.get(nutrient, 100)
                if rate < critical_satisfaction_rate:
                    is_critical = True
                    reason_parts.append(f"满足率仅{rate:.1f}%")
            
            if is_critical:
                critical_nutrients.append({
                    "nutrient": nutrient,
                    "reason": "; ".join(reason_parts),
                    "deficit_amount": abs(balance) if balance < 0 else 0,
                    "streak": streak
                })
        
        result['critical_nutrients'] = critical_nutrients
        
        # 历史趋势分析
        if len(history) >= 2:
            history_summary = {}
            for nutrient in self.nutrition_stock.keys():
                balances = [record.get('balance', {}).get(nutrient, 0) for record in history]
                
                # 计算趋势
                recent_avg = sum(balances[-3:]) / len(balances[-3:]) if len(balances) >= 3 else balances[-1]
                latest_balance = balances[-1]
                
                if latest_balance > recent_avg * 1.1:
                    trend = "改善"
                elif latest_balance < recent_avg * 0.9:
                    trend = "恶化"
                else:
                    trend = "平稳"
                
                history_summary[nutrient] = {
                    "trend": trend,
                    "recent_avg_balance": recent_avg,
                    "latest_balance": latest_balance
                }
            
            result['history_summary'] = history_summary
        else:
            result['history_summary'] = {}
        
        return result
    
    def get_last_month_balance(self) -> Dict[str, Any]:
        """
        获取上个月的营养结余数据（用于决策参考）
        
        Returns:
            {
                "data_available": bool,           # 是否有数据
                "month": int,                     # 月份
                "balance": {                      # 具体结余数值
                    "carbohydrate_g": float,      # 正值=盈余，负值=亏损
                    "protein_g": float,
                    ...
                },
                "message": str                    # 说明信息
            }
        """
        last_month_balance = self.nutrition_reference.get('last_month_balance', {})
        
        # 检查是否有有效数据（非全零）
        has_data = any(abs(v) > 0.01 for v in last_month_balance.values())
        
        if has_data:
            return {
                "data_available": True,
                "month": self.current_month - 1 if self.current_month > 0 else 0,
                "balance": deepcopy(last_month_balance),
                "message": f"上个月（第{self.current_month - 1}月）营养结余数据"
            }
        else:
            return {
                "data_available": False,
                "month": None,
                "balance": {},
                "message": "没有读取到上个月的数据（可能是第一个月或数据未初始化）"
            }
    
    @staticmethod
    def calculate_all_families_average_balance(families: List['FamilyAttributeSystem']) -> Dict[str, Any]:
        """
        计算所有家庭上个月结余的平均值（用于社会对比）
        
        Args:
            families: 家庭属性系统列表
            
        Returns:
            {
                "data_available": bool,
                "family_count": int,
                "avg_balance": {              # 所有家庭的平均结余
                    "carbohydrate_g": float,
                    "protein_g": float,
                    ...
                },
                "message": str
            }
        """
        if not families:
            return {
                "data_available": False,
                "family_count": 0,
                "avg_balance": {},
                "message": "没有读取到其他家庭数据"
            }
        
        # 收集所有家庭的结余数据
        valid_families = []
        all_balances = {}
        
        for family in families:
            last_balance = family.nutrition_reference.get('last_month_balance', {})
            # 检查是否有有效数据
            if last_balance and any(abs(v) > 0.01 for v in last_balance.values()):
                valid_families.append(family)
                
                # 累加各营养素的结余
                for nutrient, value in last_balance.items():
                    if nutrient not in all_balances:
                        all_balances[nutrient] = []
                    all_balances[nutrient].append(value)
        
        if not valid_families:
            return {
                "data_available": False,
                "family_count": 0,
                "avg_balance": {},
                "message": "没有读取到有效的家庭数据（可能所有家庭都是第一个月）"
            }
        
        # 计算平均值
        avg_balance = {}
        for nutrient, values in all_balances.items():
            avg_balance[nutrient] = sum(values) / len(values)
        
        return {
            "data_available": True,
            "family_count": len(valid_families),
            "avg_balance": avg_balance,
            "message": f"已读取{len(valid_families)}个家庭的上月结余数据"
        }
    
    def get_life_quality_reference(self) -> Dict[str, Any]:
        """
        获取生活品质参考数据（v4.0新增）
        
        Returns:
            {
                "current_month_supply": {...},     # 本月商品供给
                "last_month_supply": {...},        # 上月商品供给
                "last_month_score": {...},         # 上月得分
                "last_month_ranking": {...},       # 上月排名信息
                "history_length": int,             # 历史记录数量
                "comparison_mode": str             # 最近一次的比较模式
            }
        """
        history = self.life_quality_reference.get('history', [])
        
        return {
            "current_month_supply": deepcopy(self.life_quality_reference.get('current_month_supply', {})),
            "last_month_supply": deepcopy(self.life_quality_reference.get('last_month_supply', {})),
            "last_month_score": deepcopy(self.life_quality_reference.get('last_month_score', {})),
            "last_month_ranking": deepcopy(self.life_quality_reference.get('last_month_ranking', {})),
            "history_length": len(history),
            "comparison_mode": history[-1].get('comparison_mode', 'unknown') if history else 'none'
        }
    
    @staticmethod
    def calculate_all_families_supply(families: List['FamilyAttributeSystem']) -> Dict[str, Any]:
        """
        收集所有家庭的上月供给数据（v4.0新增：用于社会比较）
        
        Args:
            families: 家庭属性系统列表
            
        Returns:
            {
                "data_available": bool,
                "family_count": int,
                "supplies": [
                    {"family_id": "xxx", "supply": {...}},
                    ...
                ],
                "avg_supply": {...},               # 各维度平均供给
                "message": str
            }
        """
        if not families:
            return {
                "data_available": False,
                "family_count": 0,
                "supplies": [],
                "avg_supply": {},
                "message": "没有读取到其他家庭数据"
            }
        
        valid_families = []
        all_supplies = {}
        
        for family in families:
            if hasattr(family, 'life_quality_reference'):
                last_supply = family.life_quality_reference.get('last_month_supply', {})
                
                # 检查是否有有效数据
                if any(v > 0.001 for v in last_supply.values()):
                    valid_families.append({
                        "family_id": family.family_id,
                        "supply": deepcopy(last_supply)
                    })
                    
                    # 累加到 all_supplies
                    for dimension, value in last_supply.items():
                        if dimension not in all_supplies:
                            all_supplies[dimension] = []
                        all_supplies[dimension].append(value)
        
        if len(valid_families) == 0:
            return {
                "data_available": False,
                "family_count": 0,
                "supplies": [],
                "avg_supply": {},
                "message": "没有有效的家庭供给数据"
            }
        
        # 计算平均供给
        avg_supply = {}
        for dimension, values in all_supplies.items():
            avg_supply[dimension] = round(sum(values) / len(values), 2)
        
        return {
            "data_available": True,
            "family_count": len(valid_families),
            "supplies": valid_families,
            "avg_supply": avg_supply,
            "message": f"已收集 {len(valid_families)} 个家庭的供给数据"
        }
    
    # ========== 文件操作接口 ==========
    
    def save_to_file(self, custom_path: str = None):
        """
        保存状态到文件（单一文件包含所有月份数据）
        
        Args:
            custom_path: 自定义路径（可选）
        """
        if custom_path:
            filepath = custom_path
        else:
            output_dir = self.config.get('output_dir', 'output')
            # 如果 output_dir 是相对路径，相对于配置文件所在目录
            if not os.path.isabs(output_dir):
                config_dir = os.path.dirname(self.config_file)
                output_dir = os.path.join(config_dir, output_dir)
            filepath = os.path.join(
                output_dir,
                f"family_{self.family_id}",
                f"family_state.json"  # 改为单一文件
            )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 准备当前月份的状态（v4.0更新）
        # ✅ current_state 中的 history 数组已包含完整历史数据
        current_state = {
            "nutrition_stock": self.nutrition_stock,
            "nutrition_reference": self.nutrition_reference,  # v3.0新增，包含 history[]
            "life_quality": self.life_quality,
            "life_quality_reference": self.life_quality_reference,  # v4.0新增，包含 history[]
            # non_food_inventory 不保存（只保留统计数据）
            "timestamp": datetime.now().isoformat()
        }
        
        # 准备完整数据（方案1：精简版 - 消除冗余）
        data = {
            "family_id": self.family_id,
            "family_size": self.family_size,
            "current_month": self.current_month,
            "last_updated": datetime.now().isoformat(),
            "system_version": self.config.get('system_version', '4.0'),
            
            # ✅ 只保存当前状态（已包含完整历史）
            "current_state": current_state,
            
            # ❌ 删除 history 字段（方案1优化）
            # 原因：current_state.nutrition_reference.history 和 
            #      current_state.life_quality_reference.history 已包含所有历史数据
            #      保存 history 字段会导致数据冗余（每月数据重复保存多次）
            
            "derived_metrics": {
                "total_non_food_items": len(self.non_food_inventory),
                "months_recorded": self.current_month + 1  # 基于当前月份计算
            }
        }
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._round_values(data), f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 状态已保存（精简版-方案1）: {filepath} (月份: {self.current_month})")
    
    def load_from_file(self, filepath: str, target_month: int = None) -> bool:
        """
        从文件加载状态（方案1：只从current_state加载）
        
        Args:
            filepath: 文件路径
            target_month: 目标月份（方案1中此参数被忽略，保留用于向后兼容）
            
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ 文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复基本信息
            self.family_id = data.get('family_id', self.family_id)
            self.family_size = data.get('family_size', self.family_size)
            self.current_month = data.get('current_month', 0)
            
            # ✅ 简化加载逻辑（方案1）：只加载 current_state
            # 原因：新格式文件不再保存 history 字段，所有历史数据都在
            #      current_state.nutrition_reference.history 和
            #      current_state.life_quality_reference.history 中
            if target_month is not None and target_month != self.current_month:
                logger.warning(
                    f"⚠️ 方案1不支持加载历史月份 {target_month}，"
                    f"将加载当前状态（月份 {self.current_month}）。"
                    f"如需时光机功能，请使用包含 history 字段的旧格式文件。"
                )
            
            # 加载当前状态
            state_data = data.get('current_state', {})
            
            # 恢复属性数据
            self.nutrition_stock = state_data.get('nutrition_stock', self.nutrition_stock)
            
            # 恢复营养参考数据（v3.0新增，向后兼容）
            if 'nutrition_reference' in state_data:
                self.nutrition_reference = state_data.get('nutrition_reference')
            else:
                # 旧版本文件没有nutrition_reference，使用默认值
                logger.info("📝 旧版本文件，使用默认nutrition_reference")
            
            self.life_quality = state_data.get('life_quality', self.life_quality)
            
            # 恢复生活品质参考数据（v4.0新增，向后兼容）
            if 'life_quality_reference' in state_data:
                self.life_quality_reference = state_data.get('life_quality_reference')
            else:
                # 旧版本文件没有life_quality_reference，使用默认值
                logger.info("📝 旧版本文件，使用默认life_quality_reference")
            
            # non_food_inventory 不从文件加载（v4.0改动）
            self.non_food_inventory = []
            
            logger.info(f"✅ 状态加载成功: {filepath} (月份: {self.current_month})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载状态失败: {e}")
            return False
    
    def _round_values(self, obj):
        """递归处理数值，保留指定小数位数"""
        precision = self.config.get('file_settings', {}).get('float_precision', 2)
        
        if isinstance(obj, dict):
            return {key: self._round_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._round_values(item) for item in obj]
        elif isinstance(obj, float):
            return round(obj, precision)
        else:
            return obj


# 向后兼容：旧代码可能使用 FamilyAttributeManager 这个名字
FamilyAttributeManager = FamilyAttributeSystem
