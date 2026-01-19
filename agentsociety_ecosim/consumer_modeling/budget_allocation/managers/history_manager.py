"""
历史数据管理器

负责保存和管理家庭预算分配的历史数据，包括：
- 月度消费概览
- 大类预算历史
- 小类预算历史
- 购物清单历史

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

import os
import json
import datetime
import logger
from typing import Dict, Any

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)


class HistoryManager:
    """
    历史数据管理器
    
    管理4类历史文件：
    1. monthly_consumption_history.json - 月度消费概览
    2. category_budget_history.json - 大类预算历史
    3. subcategory_budget_history.json - 小类预算历史
    4. shopping_plan_history.json - 购物清单历史
    """
    
    def __init__(self, output_base_dir: str = None):
        """
        初始化历史管理器
        
        Args:
            output_base_dir: 输出基础目录，默认为当前模块的output目录
        """
        if output_base_dir is None:
            # 默认路径：consumer_modeling/output
            self.output_base_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "..", 
                "output"
            )
        else:
            self.output_base_dir = output_base_dir
    
    def save_allocation_results_with_history(
        self, 
        family_id: str, 
        current_month: int, 
        monthly_budget: float, 
        category_budget: Dict[str, float],
        subcategory_budget: Dict, 
        shopping_plan: Dict
    ) -> None:
        """
        保存预算分配结果到4个独立的JSON文件，支持历史数据管理
        为每个家庭创建专门的目录，包含4个历史文件：
        1. monthly_consumption_history.json - 主历史文件（所有月份概览）
        2. category_budget_history.json - 大类预算历史
        3. subcategory_budget_history.json - 小类预算历史
        4. shopping_plan_history.json - 购物清单历史
        
        Args:
            family_id: 家庭ID
            current_month: 当前月份
            monthly_budget: 月度总预算
            category_budget: 大类预算分配
            subcategory_budget: 小类预算分配
            shopping_plan: 商品清单
        """
        try:
            # 创建家庭专属目录
            family_dir = os.path.join(self.output_base_dir, f"family_{family_id}")
            os.makedirs(family_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().isoformat()
            
            # 1. 保存主历史文件（月度消费概览）
            self._save_monthly_consumption_history(
                family_dir, family_id, current_month, monthly_budget, 
                category_budget, subcategory_budget, shopping_plan, timestamp
            )
            
            # 2. 保存大类预算历史
            self._save_category_budget_history(
                family_dir, family_id, current_month, category_budget, timestamp
            )
            
            # 3. 保存小类预算历史
            self._save_subcategory_budget_history(
                family_dir, family_id, current_month, subcategory_budget, timestamp
            )
            
            # 4. 保存购物清单历史
            self._save_shopping_plan_history(
                family_dir, family_id, current_month, shopping_plan, timestamp
            )
            
            print(f"家庭{family_id}第{current_month}月预算分配结果已保存到: {family_dir}")
            
        except Exception as e:
            # 保存失败不影响主要功能，只记录错误
            logger.error(f"保存预算分配结果时发生错误: {e}", exc_info=True)
            print(f"警告: 保存预算分配结果时发生错误: {e}")
    
    def _save_monthly_consumption_history(
        self, 
        family_dir: str, 
        family_id: str, 
        current_month: int,
        monthly_budget: float, 
        category_budget: Dict[str, float],
        subcategory_budget: Dict, 
        shopping_plan: Dict, 
        timestamp: str
    ) -> None:
        """保存月度消费概览历史"""
        file_path = os.path.join(family_dir, "monthly_consumption_history.json")
        
        # 计算统计信息
        total_category_budget = sum(category_budget.values()) if category_budget else 0
        total_subcategory_budget = sum(
            sum(subcat.values()) if isinstance(subcat, dict) else subcat
            for subcat in subcategory_budget.values()
        ) if subcategory_budget else 0
        
        total_products = 0
        total_shopping_cost = 0
        if isinstance(shopping_plan, dict):
            for category_data in shopping_plan.values():
                if isinstance(category_data, dict):
                    for subcat_products in category_data.values():
                        if isinstance(subcat_products, list):
                            total_products += len(subcat_products)
                            total_shopping_cost += sum(p.get('total_spent', 0) for p in subcat_products)
        
        current_record = {
            "month": current_month,
            "monthly_budget": monthly_budget,
            "total_category_budget": total_category_budget,
            "total_subcategory_budget": total_subcategory_budget,
            "total_products_selected": total_products,
            "total_shopping_cost": total_shopping_cost,
            "budget_utilization_rate": (total_shopping_cost / monthly_budget) if monthly_budget > 0 else 0,
            "timestamp": timestamp,
            "categories_count": len(category_budget) if category_budget else 0,
            "subcategories_count": sum(
                len(subcat) if isinstance(subcat, dict) else 1
                for subcat in subcategory_budget.values()
            ) if subcategory_budget else 0
        }
        
        self._update_history_file(file_path, family_id, current_month, current_record, "monthly_consumption")
    
    def _save_category_budget_history(
        self, 
        family_dir: str, 
        family_id: str, 
        current_month: int,
        category_budget: Dict[str, float], 
        timestamp: str
    ) -> None:
        """保存大类预算历史"""
        file_path = os.path.join(family_dir, "category_budget_history.json")
        
        current_record = {
            "month": current_month,
            "category_budget": category_budget,
            "total_budget": sum(category_budget.values()) if category_budget else 0,
            "timestamp": timestamp
        }
        
        self._update_history_file(file_path, family_id, current_month, current_record, "category_budget")
    
    def _save_subcategory_budget_history(
        self, 
        family_dir: str, 
        family_id: str, 
        current_month: int,
        subcategory_budget: Dict, 
        timestamp: str
    ) -> None:
        """保存小类预算历史"""
        file_path = os.path.join(family_dir, "subcategory_budget_history.json")
        
        # 计算小类统计信息
        total_subcategory_budget = 0
        
        if subcategory_budget:
            for category, subcat_data in subcategory_budget.items():
                if isinstance(subcat_data, dict): 
                    total_subcategory_budget += sum(subcat_data.values())
                else:
                    total_subcategory_budget += subcat_data if isinstance(subcat_data, (int, float)) else 0
        
        current_record = {
            "month": current_month,
            "subcategory_budget": subcategory_budget,
            "total_subcategory_budget": total_subcategory_budget,
            "timestamp": timestamp
        }
        
        self._update_history_file(file_path, family_id, current_month, current_record, "subcategory_budget")
    
    def _save_shopping_plan_history(
        self, 
        family_dir: str, 
        family_id: str, 
        current_month: int,
        shopping_plan: Dict, 
        timestamp: str
    ) -> None:
        """保存购物清单历史"""
        file_path = os.path.join(family_dir, "shopping_plan_history.json")
        
        # 计算购物统计信息
        shopping_stats = {
            "total_products": 0,
            "total_cost": 0,
            "categories_with_products": 0,
            "subcategories_with_products": 0,
            "category_breakdown": {}
        }
        
        if isinstance(shopping_plan, dict):
            for category, category_data in shopping_plan.items():
                if isinstance(category_data, dict):
                    category_products = 0
                    category_cost = 0
                    subcats_with_products = 0
                    
                    for subcat, products in category_data.items():
                        if isinstance(products, list) and products:
                            subcats_with_products += 1
                            category_products += len(products)
                            category_cost += sum(p.get('total_spent', 0) for p in products)
                    
                    if category_products > 0:
                        shopping_stats["categories_with_products"] += 1
                        shopping_stats["subcategories_with_products"] += subcats_with_products
                        shopping_stats["category_breakdown"][category] = {
                            "products_count": category_products,
                            "total_cost": category_cost,
                            "subcategories_count": subcats_with_products
                        }
                    
                    shopping_stats["total_products"] += category_products
                    shopping_stats["total_cost"] += category_cost
                elif isinstance(category_data, (int, float)):
                    # 没有具体商品的类别（如住房、交通等）
                    shopping_stats["category_breakdown"][category] = {
                        "budget_allocated": category_data,
                        "note": "No specific products (service category)"
                    }
        
        current_record = {
            "month": current_month,
            "shopping_plan": shopping_plan,
            "shopping_stats": shopping_stats,
            "timestamp": timestamp
        }
        
        self._update_history_file(file_path, family_id, current_month, current_record, "shopping_plan")
    
    def _update_history_file(
        self, 
        file_path: str, 
        family_id: str, 
        current_month: int, 
        current_record: Dict, 
        file_type: str
    ) -> None:
        """通用的历史文件更新函数"""
        # 读取现有数据
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"无法读取现有{file_type}文件 {file_path}: {e}")
                print(f"警告: 无法读取现有{file_type}文件 {file_path}: {e}")
                existing_data = {}
        
        # 初始化数据结构
        if "family_id" not in existing_data:
            existing_data = {
                "family_id": family_id,
                "file_type": file_type,
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "monthly_records": []
            }
        else:
            existing_data["last_updated"] = datetime.datetime.now().isoformat()
        
        # 查找是否已存在当前月份的记录
        monthly_records = existing_data.get("monthly_records", [])
        month_exists = False
        
        for i, record in enumerate(monthly_records):
            if record.get("month") == current_month:
                # 覆盖现有记录
                monthly_records[i] = current_record
                month_exists = True
                break
        
        # 如果月份不存在，添加新记录
        if not month_exists:
            monthly_records.append(current_record)
        
        # 按月份排序
        monthly_records.sort(key=lambda x: x.get("month", 0))
        existing_data["monthly_records"] = monthly_records
        
        # 保存到文件
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"保存{file_type}历史文件失败 {file_path}: {e}")
            raise

