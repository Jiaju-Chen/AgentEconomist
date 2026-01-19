#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSID数据整合脚本
整合2011-2021年的PSID数据，生成家庭画像，并保存为JSON格式
前20个家庭ID的数据处理
"""

import os
import json
import pandas as pd
import logging
import asyncio
import aiohttp
import concurrent.futures
from typing import Dict, Any, List, Optional
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PSIDDataIntegrator:
    def __init__(self, data_path: str):
        """
        初始化PSID数据整合器
        
        Args:
            data_path: PSID处理后数据文件夹路径
        """
        self.data_path = data_path
        self.years = [2011, 2013, 2015, 2017, 2019, 2021]
        self.data_files = {
            year: f"psid_{year}_processed.csv" 
            for year in self.years
        }
        
        # 变量分类定义
        self.basic_info_vars = [
            'household_id', 'family_size', 'head_age', 'head_gender', 
            'life_satisfaction', 'head_marital_status', 'spouse_age', 
            'spouse_gender', 'num_children', 'youngest_child_age', 'state_code'
        ]
        
        self.wealth_vars = [
            'housing_type', 'home_equity', 'total_wealth', 'num_vehicles'
        ]
        
        self.income_expenditure_vars = [
            'total_income', 'total_expenditure'
        ]
        
        self.expenditure_categories = [
            'food_expenditure', 'clothing_expenditure', 'education_expenditure',
            'childcare_expenditure', 'electronics_expenditure', 'home_furnishing_equipment',
            'other_recreation_expenditure', 'housing_expenditure', 'utilities_expenditure',
            'transportation_expenditure', 'healthcare_expenditure', 'travel_expenditure',
            'phone_internet_expenditure'
        ]
        
        # LLM配置
        self.api_key = "sk-JeCvnVJdFk1SbiUc8Klw6t0wRn4KjT4G9DD7V1zjT9n26NIw"
        self.base_url = "http://35.220.164.252:3888/v1/"
        self.model_name = "gpt-3.5-turbo"
        
        # 变量含义映射
        self.variable_meanings = {
            'head_gender': {1: 'male', 2: 'female'},
            'life_satisfaction': {
                1: 'Completely satisfied', 2: 'Very satisfied', 3: 'Somewhat satisfied',
                4: 'Not very satisfied', 5: 'Not at all satisfied', 8: 'Cannot judge', 9: 'No answer/Refused'
            },
            'head_marital_status': {
                1: 'Married', 2: 'Never married', 3: 'Widowed', 4: 'Divorced/Annulled',
                5: 'Separated', 8: 'Unknown', 9: 'No answer/Refused'
            },
            'spouse_gender': {1: 'male', 2: 'female'},
            'housing_type': {
                1: 'Own/buying', 5: 'Pay rent', 8: 'Neither own nor rent'
            }
        }
        
    def load_data(self) -> Dict[int, pd.DataFrame]:
        """加载所有年份的数据"""
        data = {}
        for year in self.years:
            file_path = os.path.join(self.data_path, self.data_files[year])
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    data[year] = df
                    logger.info(f"成功加载 {year} 年数据，共 {len(df)} 行")
                except Exception as e:
                    logger.error(f"加载 {year} 年数据失败: {e}")
            else:
                logger.warning(f"文件不存在: {file_path}")
        return data
    
    def safe_get_value(self, df: pd.DataFrame, household_id: int, column: str) -> Any:
        """
        安全获取数值，保留null值表示变量在某些年份不存在
        
        Args:
            df: 数据框
            household_id: 家庭ID
            column: 列名
            
        Returns:
            数值或None（表示null）
        """
        try:
            # 查找对应家庭ID的行
            family_rows = df[df['household_id'] == household_id]
            if family_rows.empty:
                return None
            
            if column not in df.columns:
                return None
                
            value = family_rows.iloc[0][column]
            
            # 保留pandas的NaN和None作为null值
            if pd.isna(value):
                return None
            if value == '' or str(value).lower() == 'nan':
                return None
                
            # 对于数值类型，尝试转换，失败则返回None
            if isinstance(value, (int, float)):
                return float(value) if not pd.isna(value) else None
            
            return value
        except Exception as e:
            logger.debug(f"获取值时出错 - 家庭ID: {household_id}, 列: {column}, 错误: {e}")
            return None
    
    def interpret_coded_value(self, column: str, value: Any) -> Any:
        """
        解释编码值，转换为有意义的描述
        
        Args:
            column: 列名
            value: 原始值
            
        Returns:
            解释后的值或原始值
        """
        if value is None or pd.isna(value):
            return None
            
        if column in self.variable_meanings:
            mapping = self.variable_meanings[column]
            if isinstance(value, (int, float)) and int(value) in mapping:
                return mapping[int(value)]
        
        return value
    
    def get_family_ids(self, data: Dict[int, pd.DataFrame], limit: int = 100, start_id: int = None, end_id: int = None, id_range: List[int] = None) -> List[int]:
        """
        获取家庭ID列表，支持多种模式
        
        Args:
            data: 各年份数据字典
            limit: 数量限制（当使用前N个模式时）
            start_id: 起始ID（包含）
            end_id: 结束ID（包含）
            id_range: 直接指定ID列表
            
        Returns:
            家庭ID列表
        """
        all_family_ids = set()
        
        # 收集所有年份中的家庭ID
        for year, df in data.items():
            if 'household_id' in df.columns:
                family_ids = df['household_id'].dropna().astype(int).tolist()
                all_family_ids.update(family_ids)
        
        all_family_ids = sorted(list(all_family_ids))
        
        # 如果直接指定ID列表
        if id_range is not None:
            filtered_ids = [fid for fid in id_range if fid in all_family_ids]
            logger.info(f"指定ID列表：找到 {len(filtered_ids)} 个有效家庭ID（从 {len(id_range)} 个指定ID中）")
            return filtered_ids
        
        # 如果指定ID范围
        if start_id is not None or end_id is not None:
            filtered_ids = []
            for fid in all_family_ids:
                if start_id is not None and fid < start_id:
                    continue
                if end_id is not None and fid > end_id:
                    continue
                filtered_ids.append(fid)
            logger.info(f"ID范围 {start_id}-{end_id}：找到 {len(filtered_ids)} 个家庭ID")
            return filtered_ids
        
        # 默认模式：取前N个
        sorted_ids = all_family_ids[:limit]
        logger.info(f"找到 {len(all_family_ids)} 个独特家庭ID，选择前 {limit} 个")
        return sorted_ids
    
    def integrate_family_data(self, family_id: int, data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        整合单个家庭的跨年数据，按新的JSON格式组织
        
        Args:
            family_id: 家庭ID
            data: 各年份数据字典
            
        Returns:
            整合后的家庭数据
        """
        # 新的格式结构
        family_data = {
            "household_id": family_id,
            "basic_family_info": {},
            "family_wealth_situation": {},
            "total_income_expenditure": {},
            "expenditure_categories": {}
        }
        
        # 1. 家庭基本信息（使用2021年数据，如果没有则用最新可用数据）
        # 添加车辆数到基本信息中，只保留最后一年的数据
        basic_info_year = None
        for year in reversed(self.years):  # 从2021开始往前找
            if year in data and family_id in data[year]['household_id'].values:
                basic_info_year = year
                break
        
        if basic_info_year:
            df = data[basic_info_year]
            for var in self.basic_info_vars:
                if var != 'household_id':  # household_id已经设置
                    raw_value = self.safe_get_value(df, family_id, var)
                    # 对编码值进行解释
                    interpreted_value = self.interpret_coded_value(var, raw_value)
                    family_data["basic_family_info"][var] = interpreted_value
            
            # 添加车辆数到基本信息
            num_vehicles = self.safe_get_value(df, family_id, 'num_vehicles')
            family_data["basic_family_info"]["num_vehicles"] = num_vehicles
        
        # 2. 家庭财富情况（只保留多年数据用于分析，不保存详细年份值）
        wealth_by_year = {}
        income_by_year = {}
        
        for year in self.years:
            if year in data and family_id in data[year]['household_id'].values:
                df = data[year]
                
                # 财富数据（用于分析）
                year_wealth = {}
                for var in ['housing_type', 'home_equity', 'total_wealth']:  # 排除num_vehicles
                    raw_value = self.safe_get_value(df, family_id, var)
                    interpreted_value = self.interpret_coded_value(var, raw_value)
                    year_wealth[var] = interpreted_value
                wealth_by_year[str(year)] = year_wealth
                
                # 收入支出数据（用于分析）
                year_income = {}
                for var in self.income_expenditure_vars:
                    year_income[var] = self.safe_get_value(df, family_id, var)
                income_by_year[str(year)] = year_income
        
        # 临时保存用于分析
        family_data["_temp_wealth_by_year"] = wealth_by_year
        family_data["_temp_income_by_year"] = income_by_year
        
        # 3. 总收支情况（格式：[2011, 2013, 2015, 2017, 2019, 2021]）
        total_income_array = []
        total_expenditure_array = []
        
        for year in self.years:
            if str(year) in income_by_year:
                total_income_array.append(income_by_year[str(year)].get('total_income'))
                total_expenditure_array.append(income_by_year[str(year)].get('total_expenditure'))
            else:
                total_income_array.append(None)
                total_expenditure_array.append(None)
        
        family_data["total_income_expenditure"]["total_income"] = total_income_array
        family_data["total_income_expenditure"]["total_expenditure"] = total_expenditure_array
        
        # 4. 各类支出情况（格式：[2011, 2013, 2015, 2017, 2019, 2021]）
        for category in self.expenditure_categories:
            category_array = []
            for year in self.years:
                if year in data and family_id in data[year]['household_id'].values:
                    df = data[year]
                    value = self.safe_get_value(df, family_id, category)
                    category_array.append(value)
                else:
                    category_array.append(None)
            family_data["expenditure_categories"][category] = category_array
        
        return family_data
    
    def call_llm_chat_completion(self, prompt: str, system_content: str) -> str:
        """
        通用LLM对话接口，返回模型回复内容。
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""
    
    def generate_family_profile(self, family_data: Dict[str, Any]) -> str:
        """
        使用LLM生成精简的英文家庭画像
        
        Args:
            family_data: 家庭数据
            
        Returns:
            英文家庭画像文本
        """
        # 构建英文提示词
        basic_info = family_data.get("basic_family_info", {})
        expenditure_categories = family_data.get("expenditure_categories", {})
        income_expenditure = family_data.get("total_income_expenditure", {})
        
        # 准备精简的英文数据摘要
        family_summary = f"""
Household ID: {family_data.get('household_id')}
Basic Info: {basic_info.get('family_size', 0)} people, head age {basic_info.get('head_age', 'N/A')}, {basic_info.get('head_gender', 'N/A')}, {basic_info.get('head_marital_status', 'N/A')}, {basic_info.get('num_children', 0)} children, {basic_info.get('num_vehicles', 0)} vehicles

Income trend (2011-2021): {income_expenditure.get('total_income', [])}
Expenditure trend (2011-2021): {income_expenditure.get('total_expenditure', [])}

Key spending categories (2011-2021):
"""
        
        # 添加主要支出类别
        key_categories = ['food_expenditure', 'clothing_expenditure', 'housing_expenditure', 'transportation_expenditure']
        for category in key_categories:
            if category in expenditure_categories:
                family_summary += f"- {category}: {expenditure_categories[category]}\n"
        
        system_content = """Generate a concise family profile in English. Focus on:
1. Demographics and life stage
2. Economic status and spending capacity  
3. Consumption patterns and preferences
4. Product selection tendencies

Keep the response under 150 words and professional."""
        
        prompt = f"""Analyze this household data and generate a concise family profile:

{family_summary}

Focus on consumer behavior insights for product selection and budget allocation."""
        
        return self.call_llm_chat_completion(prompt, system_content)
    
    def generate_wealth_analysis(self, family_data: Dict[str, Any]) -> str:
        """
        使用LLM生成精简的家庭财富分析
        
        Args:
            family_data: 家庭数据
            
        Returns:
            英文财富分析文本
        """
        wealth_by_year = family_data.get("_temp_wealth_by_year", {})
        income_by_year = family_data.get("_temp_income_by_year", {})
        
        # 准备精简的财富数据摘要
        wealth_summary = f"""
Household ID: {family_data.get('household_id')}

Wealth trends:
"""
        
        # 提取关键数据点
        years = sorted([int(y) for y in wealth_by_year.keys()])
        if years:
            first_year = str(years[0])
            last_year = str(years[-1])
            
            if first_year in wealth_by_year and last_year in wealth_by_year:
                first_wealth = wealth_by_year[first_year].get('total_wealth')
                last_wealth = wealth_by_year[last_year].get('total_wealth')
                wealth_summary += f"Total wealth: {first_year}: ${first_wealth}, {last_year}: ${last_wealth}\n"
                
                first_income = income_by_year.get(first_year, {}).get('total_income')
                last_income = income_by_year.get(last_year, {}).get('total_income')
                wealth_summary += f"Income: {first_year}: ${first_income}, {last_year}: ${last_income}\n"
        
        system_content = """Analyze the family's financial situation objectively. Focus on:
1. Wealth accumulation trends
2. Income stability patterns
3. Financial capacity assessment

Provide specific data points and keep under 100 words."""
        
        prompt = f"""Analyze this family's wealth situation objectively with specific data:

{wealth_summary}

Provide factual analysis of financial trends and capacity."""
        
        return self.call_llm_chat_completion(prompt, system_content)
    
    def integrate_all_families_batch(self, limit: int = 100, start_id: int = None, end_id: int = None, id_range: List[int] = None, batch_size: int = 64) -> Dict[str, Any]:
        """
        分批整合家庭数据，使用并发处理加速，支持多种ID选择模式，每批处理后保存
        
        Args:
            limit: 处理的家庭数量限制（当使用前N个模式时）
            start_id: 起始家庭ID（包含）
            end_id: 结束家庭ID（包含）
            id_range: 直接指定家庭ID列表
            batch_size: 每批处理的家庭数量
            
        Returns:
            所有家庭的整合数据
        """
        logger.info("开始加载数据...")
        data = self.load_data()
        
        if not data:
            logger.error("没有成功加载任何数据文件")
            return {}
        
        logger.info("获取家庭ID列表...")
        family_ids = self.get_family_ids(data, limit, start_id, end_id, id_range)
        
        if not family_ids:
            logger.error("没有找到任何家庭ID")
            return {}
        
        # 创建临时保存文件路径
        temp_output_file = os.path.join(self.data_path, "temp_integrated_families.json")
        
        # 初始化或加载现有数据
        if os.path.exists(temp_output_file):
            logger.info("发现临时文件，加载已处理的数据...")
            all_families_data = self.load_existing_json(temp_output_file)
            processed_ids = set(all_families_data.get("families", {}).keys())
            # 过滤掉已处理的家庭ID
            family_ids = [fid for fid in family_ids if str(fid) not in processed_ids]
            logger.info(f"已处理 {len(processed_ids)} 个家庭，剩余 {len(family_ids)} 个家庭需要处理")
        else:
            all_families_data = {
                "metadata": {
                    "total_families": len(family_ids),
                    "years_covered": self.years,
                    "processing_date": pd.Timestamp.now().isoformat(),
                    "data_description": "PSID 2011-2021 family consumption data integration, null values preserved for variables not available in certain years",
                    "batch_size": batch_size
                },
                "families": {}
            }
        
        # 分批处理
        total_batches = (len(family_ids) + batch_size - 1) // batch_size
        logger.info(f"开始分批处理 {len(family_ids)} 个家庭的数据，每批 {batch_size} 个，共 {total_batches} 批...")
        
        for batch_num in range(0, len(family_ids), batch_size):
            batch_ids = family_ids[batch_num:batch_num + batch_size]
            current_batch_num = batch_num // batch_size + 1
            
            logger.info(f"处理第 {current_batch_num}/{total_batches} 批，包含家庭ID: {batch_ids}")
            
            def process_single_family(family_id: int) -> tuple:
                """处理单个家庭数据的函数"""
                try:
                    logger.info(f"处理家庭 {family_id}")
                    
                    # 整合家庭数据
                    family_data = self.integrate_family_data(family_id, data)
                    
                    # 生成家庭画像
                    logger.info(f"为家庭 {family_id} 生成家庭画像...")
                    family_profile = self.generate_family_profile(family_data)
                    
                    # 生成财富分析
                    logger.info(f"为家庭 {family_id} 生成财富分析...")
                    wealth_analysis = self.generate_wealth_analysis(family_data)
                    
                    # 添加生成的内容到最终结构
                    family_data["family_wealth_situation"]["wealth_analysis"] = wealth_analysis
                    family_data["family_profile"] = family_profile
                    
                    # 清理临时数据
                    if "_temp_wealth_by_year" in family_data:
                        del family_data["_temp_wealth_by_year"]
                    if "_temp_income_by_year" in family_data:
                        del family_data["_temp_income_by_year"]
                    
                    logger.info(f"家庭 {family_id} 处理完成")
                    return family_id, family_data, None
                    
                except Exception as e:
                    logger.error(f"处理家庭 {family_id} 时出错: {e}")
                    return family_id, None, str(e)
            
            # 使用并发处理当前批次
            batch_success_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # 增加并发数提高处理速度
                futures = [executor.submit(process_single_family, family_id) for family_id in batch_ids]
                
                for future in concurrent.futures.as_completed(futures):
                    family_id, family_data, error = future.result()
                    
                    if error:
                        logger.error(f"家庭 {family_id} 处理失败: {error}")
                        continue
                    
                    if family_data:
                        all_families_data["families"][str(family_id)] = family_data
                        batch_success_count += 1
            
            # 保存当前批次的进度
            logger.info(f"第 {current_batch_num} 批处理完成，成功处理 {batch_success_count}/{len(batch_ids)} 个家庭")
            logger.info(f"保存临时进度到文件...")
            
            # 按家庭ID排序
            sorted_families = {}
            for family_id in sorted(all_families_data["families"].keys(), key=int):
                sorted_families[family_id] = all_families_data["families"][family_id]
            all_families_data["families"] = sorted_families
            
            # 更新元数据
            all_families_data["metadata"]["total_families"] = len(all_families_data["families"])
            all_families_data["metadata"]["last_update"] = pd.Timestamp.now().isoformat()
            all_families_data["metadata"]["completed_batches"] = current_batch_num
            
            # 保存临时文件
            self.save_to_json(all_families_data, temp_output_file)
            logger.info(f"已完成 {current_batch_num}/{total_batches} 批，累计处理 {len(all_families_data['families'])} 个家庭")
        
        logger.info(f"所有批次处理完成，共处理 {len(all_families_data['families'])} 个家庭")
        
        # 清理临时文件
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logger.info("已删除临时进度文件")
        
        return all_families_data
    
    def load_existing_json(self, file_path: str) -> Dict[str, Any]:
        """
        加载现有的JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            现有数据或空数据结构
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"成功加载现有数据文件: {file_path}")
                return data
            except Exception as e:
                logger.error(f"加载现有文件失败: {e}")
                return self._create_empty_data_structure()
        else:
            logger.info(f"文件不存在，创建新的数据结构: {file_path}")
            return self._create_empty_data_structure()
    
    def _create_empty_data_structure(self) -> Dict[str, Any]:
        """创建空的数据结构"""
        return {
            "metadata": {
                "total_families": 0,
                "years_covered": self.years,
                "processing_date": pd.Timestamp.now().isoformat(),
                "data_description": "PSID 2011-2021 family consumption data integration, null values preserved for variables not available in certain years"
            },
            "families": {}
        }
    
    def append_families_to_existing(self, new_families_data: Dict[str, Any], existing_file: str) -> Dict[str, Any]:
        """
        将新的家庭数据追加到现有JSON文件中，自动去重并按家庭ID排序
        
        Args:
            new_families_data: 新的家庭数据
            existing_file: 现有JSON文件路径
            
        Returns:
            合并后的完整数据
        """
        # 加载现有数据
        existing_data = self.load_existing_json(existing_file)
        
        # 追加新的家庭数据
        new_families = new_families_data.get("families", {})
        existing_families = existing_data.get("families", {})
        
        # 记录重复和新增的家庭
        duplicate_count = 0
        new_count = 0
        
        logger.info(f"开始合并数据：现有 {len(existing_families)} 个家庭，新增 {len(new_families)} 个家庭")
        
        for family_id, family_data in new_families.items():
            if family_id in existing_families:
                logger.warning(f"家庭 {family_id} 已存在，将被覆盖")
                duplicate_count += 1
            else:
                new_count += 1
            existing_families[family_id] = family_data
        
        # 按家庭ID排序（转换为整数排序）
        logger.info("按家庭ID排序...")
        sorted_families = {}
        sorted_family_ids = sorted(existing_families.keys(), key=lambda x: int(x))
        
        for family_id in sorted_family_ids:
            sorted_families[family_id] = existing_families[family_id]
        
        # 更新数据结构
        existing_data["families"] = sorted_families
        existing_data["metadata"]["total_families"] = len(sorted_families)
        existing_data["metadata"]["last_update"] = pd.Timestamp.now().isoformat()
        
        logger.info(f"数据合并完成：新增 {new_count} 个家庭，覆盖 {duplicate_count} 个家庭")
        logger.info(f"总计 {len(sorted_families)} 个家庭，已按家庭ID排序")
        
        return existing_data

    def save_to_json(self, data: Dict[str, Any], output_file: str):
        """
        保存数据到JSON文件
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"数据已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存文件失败: {e}")

def process_specific_range(start_id: int, end_id: int, append_to_existing: bool = True, batch_size: int = 64):
    """
    处理指定范围ID的家庭数据，使用分批处理避免内存过载
    
    Args:
        start_id: 起始家庭ID（包含）
        end_id: 结束家庭ID（包含）
        append_to_existing: 是否追加到现有文件
        batch_size: 每批处理的家庭数量
    """
    # 设置路径
    data_path = "/root/github_clone/agentsociety-ecosim/agentsociety_ecosim/consumer_modeling/household_data/PSID/extracted_data/processed_data"
    output_file = os.path.join(data_path, "integrated_psid_families_data.json")
    
    # 创建整合器
    integrator = PSIDDataIntegrator(data_path)
    
    # 处理指定范围的家庭
    logger.info(f"开始处理家庭ID范围 {start_id} 到 {end_id}，每批处理 {batch_size} 个家庭")
    integrated_data = integrator.integrate_all_families_batch(start_id=start_id, end_id=end_id, batch_size=batch_size)
    
    if integrated_data and integrated_data.get("families"):
        if append_to_existing:
            # 追加到现有文件
            logger.info("追加数据到现有文件...")
            final_data = integrator.append_families_to_existing(integrated_data, output_file)
            integrator.save_to_json(final_data, output_file)
        else:
            # 保存为新文件
            new_output_file = os.path.join(data_path, f"families_range_{start_id}_{end_id}.json")
            integrator.save_to_json(integrated_data, new_output_file)
        
        # 输出统计信息
        num_families = len(integrated_data["families"])
        logger.info(f"范围处理完成！处理了 {num_families} 个家庭的数据（ID范围: {start_id}-{end_id}）")
    else:
        logger.error("指定范围内没有找到任何家庭数据")

def process_specific_ids(id_list: List[int], append_to_existing: bool = True, batch_size: int = 64):
    """
    处理指定ID列表的家庭数据，使用分批处理避免内存过载
    
    Args:
        id_list: 家庭ID列表
        append_to_existing: 是否追加到现有文件
        batch_size: 每批处理的家庭数量
    """
    # 设置路径
    data_path = "/root/github_clone/agentsociety-ecosim/agentsociety_ecosim/consumer_modeling/household_data/PSID/extracted_data/processed_data"
    output_file = os.path.join(data_path, "integrated_psid_families_data.json")
    
    # 创建整合器
    integrator = PSIDDataIntegrator(data_path)
    
    # 处理指定ID列表的家庭
    logger.info(f"开始处理指定的 {len(id_list)} 个家庭ID，每批处理 {batch_size} 个家庭")
    integrated_data = integrator.integrate_all_families_batch(id_range=id_list, batch_size=batch_size)
    
    if integrated_data and integrated_data.get("families"):
        if append_to_existing:
            # 追加到现有文件
            logger.info("追加数据到现有文件...")
            final_data = integrator.append_families_to_existing(integrated_data, output_file)
            integrator.save_to_json(final_data, output_file)
        else:
            # 保存为新文件
            new_output_file = os.path.join(data_path, f"families_specified_ids.json")
            integrator.save_to_json(integrated_data, new_output_file)
        
        # 输出统计信息
        num_families = len(integrated_data["families"])
        logger.info(f"指定ID处理完成！处理了 {num_families} 个家庭的数据")
    else:
        logger.error("指定ID列表中没有找到任何家庭数据")

def main():
    """主函数"""
    # 设置路径
    data_path = "/root/github_clone/agentsociety-ecosim/agentsociety_ecosim/consumer_modeling/household_data/PSID/extracted_data/processed_data"
    output_file = os.path.join(data_path, "integrated_psid_families_data.json")
    
    # 创建整合器
    integrator = PSIDDataIntegrator(data_path)
    
    # 示例：处理前500个家庭（创建初始文件）
    logger.info("处理前500个家庭...")
    integrated_data = integrator.integrate_all_families(limit=500)
    
    if integrated_data and integrated_data.get("families"):
        # 保存结果
        integrator.save_to_json(integrated_data, output_file)
        
        # 输出统计信息
        num_families = len(integrated_data["families"])
        logger.info(f"整合完成！处理了 {num_families} 个家庭的数据")
        
        # 显示一个家庭的示例数据结构
        if integrated_data["families"]:
            first_family_id = list(integrated_data["families"].keys())[0]
            first_family = integrated_data["families"][first_family_id]
            print(f"\n示例家庭 {first_family_id} 的数据结构:")
            print("- basic_family_info:", list(first_family.get("basic_family_info", {}).keys()))
            print("- family_wealth_situation:", list(first_family.get("family_wealth_situation", {}).keys()))
            print("- expenditure_categories:", list(first_family.get("expenditure_categories", {}).keys()))
            print("- family_profile 长度:", len(first_family.get("family_profile", "")))
            
            # 显示支出类别示例
            expenditure_categories = first_family.get("expenditure_categories", {})
            if expenditure_categories:
                example_category = list(expenditure_categories.keys())[0]
                print(f"- {example_category} 数据格式: 6年数据数组, 长度 {len(expenditure_categories[example_category])}")
            
            # 显示总收支格式
            income_expenditure = first_family.get("total_income_expenditure", {})
            if income_expenditure:
                print(f"- total_income 数据格式: 6年数据数组, 长度 {len(income_expenditure.get('total_income', []))}")
                print(f"- total_expenditure 数据格式: 6年数据数组, 长度 {len(income_expenditure.get('total_expenditure', []))}")
    else:
        logger.error("数据整合失败")

if __name__ == "__main__":
    # 你可以选择以下几种运行方式：
    
    # 方式1：运行主函数（处理前500个家庭）
    # main()
    
    # 方式2：处理指定范围的家庭ID（501-1000），使用分批处理，每批64个家庭，32个线程
    process_specific_range(4001, 8000, append_to_existing=True, batch_size=64)
    
    # 方式3：处理指定的家庭ID列表
    # specific_ids = [1001, 1002, 1003, 1005, 1010]
    # process_specific_ids(specific_ids, append_to_existing=True, batch_size=64)
