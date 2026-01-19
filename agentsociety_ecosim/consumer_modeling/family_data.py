import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

# 获取当前文件所在目录（consumer_modeling 目录）
_CURRENT_DIR = Path(__file__).parent.resolve()

# 更新：使用基于文件位置的相对路径
PSID_INTEGRATED_DATA_PATH = str(_CURRENT_DIR / "household_data" / "PSID" / "extracted_data" / "processed_data" / "integrated_psid_families_data.json")

# 保留旧的路径作为备用
FAMILY_DATA_PATH = str(_CURRENT_DIR / "household_data" / "processed_data" / "processed_data_2010_with_recommendations.json")
FAMILY_CONSUMPTION_PROFILE_PATH = str(_CURRENT_DIR / "household_data" / "processed_data" / "household_consumption_with_family_profile.json")

def load_psid_integrated_data():
    """加载PSID整合数据，返回字典。"""
    if not os.path.exists(PSID_INTEGRATED_DATA_PATH):
        raise FileNotFoundError(f"PSID整合数据文件不存在: {PSID_INTEGRATED_DATA_PATH}")
    with open(PSID_INTEGRATED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_family_data():
    """加载所有家庭信息，返回列表。"""
    if not os.path.exists(FAMILY_DATA_PATH):
        raise FileNotFoundError(f"家庭数据文件不存在: {FAMILY_DATA_PATH}")
    with open(FAMILY_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 新增：加载带消费和画像的家庭数据
def load_family_consumption_and_profile():
    """加载所有家庭的消费数据和画像，返回列表。"""
    if not os.path.exists(FAMILY_CONSUMPTION_PROFILE_PATH):
        raise FileNotFoundError(f"家庭消费与画像数据文件不存在: {FAMILY_CONSUMPTION_PROFILE_PATH}")
    with open(FAMILY_CONSUMPTION_PROFILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_family_by_id(family_id: int) -> Optional[Dict[str, Any]]:
    """根据家庭ID查询家庭信息，优先使用PSID数据，返回字典。"""
    try:
        # 优先使用PSID整合数据
        psid_data = load_psid_integrated_data()
        family_id_str = str(family_id)
        if family_id_str in psid_data.get("families", {}):
            return psid_data["families"][family_id_str]
    except FileNotFoundError:
        pass
    
    # 备用：使用旧的数据格式
    try:
        data = load_family_data()
        for family in data:
            if str(family.get("fid")) == str(family_id):
                return family
    except FileNotFoundError:
        pass
    
    return None

def get_family_consumption_and_profile_by_id(family_id: int) -> Optional[Dict[str, Any]]:
    """
    根据家庭id返回该家庭的消费数据和画像，优先使用PSID数据。
    :param family_id: 家庭id
    :return: 包含消费数据和画像的字典，未找到则返回None
    """
    try:
        # 优先使用PSID整合数据
        psid_data = load_psid_integrated_data()
        family_id_str = str(family_id)
        if family_id_str in psid_data.get("families", {}):
            family_data = psid_data["families"][family_id_str]
            
            # 转换支出数据格式为年份字典
            expenditure_categories = family_data.get('expenditure_categories', {})
            consumption = {}
            years = [2011, 2013, 2015, 2017, 2019, 2021]
            
            for i, year in enumerate(years):
                year_consumption = {}
                for category, values in expenditure_categories.items():
                    if i < len(values) and values[i] is not None:
                        year_consumption[category] = values[i]
                    else:
                        year_consumption[category] = 0.0
                consumption[str(year)] = year_consumption
            
            return {
                'family_profile': family_data.get('family_profile', ''),
                'consumption': consumption,
                'basic_family_info': family_data.get('basic_family_info', {}),
                'family_wealth_situation': family_data.get('family_wealth_situation', {}),
                'total_income_expenditure': family_data.get('total_income_expenditure', {})
            }
    except FileNotFoundError:
        logger.error(f"PSID整合数据文件不存在: {PSID_INTEGRATED_DATA_PATH}")
        pass
    
    # 备用：使用旧的数据格式
    try:
        data = load_family_consumption_and_profile()
        for family in data:
            if str(family.get("family_id")) == str(family_id):
                return family
    except FileNotFoundError:
        pass
    
    return None

def get_latest_expenditures_by_family_id(family_id: int, category_keys=None):
    """
    获取指定家庭最近一年的消费大类支出，返回dict。category_keys为需要的消费类别列表。
    """
    family_info = get_family_consumption_and_profile_by_id(family_id)
    if not family_info or "consumption" not in family_info:
        if category_keys:
            return {k: 0.0 for k in category_keys}
        return {}
    consumption = family_info["consumption"]
    if not consumption:
        if category_keys:
            return {k: 0.0 for k in category_keys}
        return {}
    latest_year = sorted(consumption.keys(), reverse=True)[0]
    if category_keys:
        return {k: consumption[latest_year].get(k, 0.0) for k in category_keys}
    return consumption[latest_year]

