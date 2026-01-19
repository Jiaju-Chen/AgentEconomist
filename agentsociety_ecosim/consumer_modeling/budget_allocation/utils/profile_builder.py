"""
家庭画像构建模块

本模块负责构建各种用于LLM的家庭描述文本：
- 家庭画像构建
- 历史消费数据描述
- 属性缺口引导文本

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import logger
from typing import Dict, List, Any

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class ProfileBuilder:
    """家庭画像构建器"""
    
    @staticmethod
    def build_family_profile_for_allocation(family_info: Dict) -> str:
        """
        为预算分配构建家庭画像描述
        
        Args:
            family_info: 家庭信息字典，包含demographics和consumption数据
            
        Returns:
            str: 家庭画像描述文本
        """
        demographics = family_info.get("demographics", {})
        
        # 提取关键信息
        num_members = demographics.get("num_members", 3)
        age_householder = demographics.get("age_householder", 40)
        has_children = demographics.get("has_children", False)
        education_level = demographics.get("education_level", "college")
        
        # 构建描述
        profile_parts = []
        profile_parts.append(f"Family of {num_members} members")
        profile_parts.append(f"Householder age: {age_householder}")
        
        if has_children:
            profile_parts.append("Has children")
        
        profile_parts.append(f"Education level: {education_level}")
        
        return ", ".join(profile_parts)
    
    @staticmethod
    def build_family_situation_for_llm(
        current_income: float, 
        total_balance: float, 
        family_profile: str = None
    ) -> str:
        """
        为LLM预算计算构建家庭情况描述
        
        Args:
            current_income: 当前月收入
            total_balance: 总余额
            family_profile: 家庭画像描述（可选）
            
        Returns:
            str: 完整的家庭情况描述
        """
        situation = []
        situation.append(f"Current monthly income: ${current_income:.2f}")
        situation.append(f"Total available balance: ${total_balance:.2f}")
        
        if family_profile:
            situation.append(f"Family profile: {family_profile}")
        
        return "\n".join(situation)
    
    @staticmethod
    def build_historical_description(
        historical_data: List[List[float]], 
        category_keys: List[str]
    ) -> str:
        """
        构建历史消费数据的描述
        
        Args:
            historical_data: 历史消费数据，形状为 (n_years, n_categories)
            category_keys: 类别键列表
            
        Returns:
            str: 历史数据描述文本
        """
        if not historical_data or not category_keys:
            return "No historical data available"
        
        n_years = len(historical_data)
        description_lines = []
        
        for year_idx, year_data in enumerate(historical_data):
            year_total = sum(year_data)
            if year_total > 0:
                # 计算每个类别的占比
                proportions = [f"{(val/year_total)*100:.1f}%" for val in year_data]
                description_lines.append(
                    f"Year {n_years - year_idx}: " + 
                    ", ".join([f"{cat}={prop}" for cat, prop in zip(category_keys, proportions)])
                )
        
        return "\n".join(description_lines) if description_lines else "Historical spending patterns not available"
    
    @staticmethod
    def build_attribute_guidance_prompt(
        attribute_gaps: Dict[str, float],
        attribute_to_category_mapping: Dict[str, Dict]
    ) -> str:
        """
        构建属性引导的prompt文本（简化版）
        
        Args:
            attribute_gaps: 属性缺口字典 {attribute_name: gap_value}
            attribute_to_category_mapping: 属性到类别的映射
            
        Returns:
            str: 属性引导文本
        """
        if not attribute_gaps:
            return ""
        
        # 只显示营养缺口（nutrition_开头的）
        nutrition_gaps = []
        for attr, gap in attribute_gaps.items():
            if gap > 0 and attr.startswith('nutrition_'):
                # 简化名称
                clean_name = attr.replace('nutrition_', '').replace('_g', '').replace('_', ' ')
                nutrition_gaps.append((clean_name, gap))
        
        if not nutrition_gaps:
            return ""
        
        # 构建简洁的引导文本
        guidance = "\n**Nutrition Gaps**:\n"
        for name, gap in nutrition_gaps:
            guidance += f"  • {name}: {gap:.0f}\n"
        guidance += "→ Increase food_expenditure to address these gaps\n"
        
        return guidance
    
    @staticmethod
    def extract_family_profile_dict(family_profile) -> Dict:
        """
        从字符串或字典格式的家庭画像中提取字典
        
        Args:
            family_profile: 字符串或字典格式的家庭画像
            
        Returns:
            Dict: 家庭画像字典
        """
        if isinstance(family_profile, dict):
            return family_profile
        
        if isinstance(family_profile, str):
            # 简单解析字符串为字典（假设格式为 "key1: value1, key2: value2"）
            profile_dict = {}
            try:
                parts = family_profile.split(",")
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        profile_dict[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Failed to parse family profile string: {e}")
            return profile_dict
        
        return {}

