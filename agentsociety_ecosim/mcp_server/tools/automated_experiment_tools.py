"""
自动化实验工具 - 根据问题自动生成配置、启动实验、分析结果
"""
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class QuestionAnalysis:
    """问题分析结果"""
    question_type: str  # innovation, redistribution, labor_productivity, tariff, other
    question_text: str
    identified_keywords: List[str]
    recommended_parameters: Dict[str, Any]
    recommended_config_name: str
    description: str


class AutomatedExperimentTools:
    """自动化实验工具类"""
    
    # 问题类型识别规则
    QUESTION_PATTERNS = {
        "innovation": {
            "keywords": ["innovation", "innovate", "promoting", "policy", "economic performance", 
                        "technological", "R&D", "research", "development"],
            "description": "创新促进政策对经济表现的影响",
            "parameter_groups": {
                "innovation": {
                    "enable_innovation": True,
                    "innovation_probability": 0.15,  # 提高创新概率
                    "innovation_labor_productivity_improvement": 0.15,  # 提高创新幅度
                    "innovation_price_reduction": 0.15,
                    "innovation_profit_margin_improvement": 0.15,
                }
            }
        },
        "redistribution": {
            "keywords": ["universal basic income", "UBI", "basic income", "redistribution", 
                        "welfare", "social security", "poverty", "income inequality", 
                        "people's lives", "standard of living"],
            "description": "全民基本收入政策对人民生活的影响",
            "parameter_groups": {
                "redistribution": {
                    "redistribution_strategy": "progressive",  # 渐进式再分配
                    "redistribution_poverty_weight": 0.5,  # 增加贫困权重
                    "redistribution_unemployment_weight": 0.3,
                    "redistribution_family_size_weight": 0.2,
                }
            }
        },
        "labor_productivity": {
            "keywords": ["AI", "artificial intelligence", "labor market", "automation", 
                        "productivity", "efficiency", "reshap", "reshape", "workforce",
                        "employment", "job market"],
            "description": "AI智能体对劳动力市场的重塑",
            "parameter_groups": {
                "production": {
                    "labor_productivity_factor": 150.0,  # 提高劳动生产率（默认100）
                    "labor_elasticity": 0.8,  # 提高劳动弹性
                },
                "labor_market": {
                    "dismissal_rate": 0.05,  # 降低裁员率（AI提高效率，减少裁员）
                    "enable_dynamic_job_posting": True,  # 启用动态招聘
                }
            }
        },
        "tariff": {
            "keywords": ["tariff", "tax", "trade", "breaking news", "event", "stock market",
                        "liberation day", "market shock", "economic shock"],
            "description": "关税等突发新闻事件对股票市场的影响",
            "parameter_groups": {
                "tax_policy": {
                    "vat_rate": 0.25,  # 提高增值税率模拟关税影响
                    "corporate_tax_rate": 0.45,
                },
                "market": {
                    "enable_price_adjustment": True,  # 启用价格调整
                    "price_adjustment_rate": 0.15,  # 提高价格调整速率
                }
            }
        }
    }
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """
        分析问题并识别类型
        
        Args:
            question: 问题文本
        
        Returns:
            问题分析结果
        """
        question_lower = question.lower()
        identified_keywords = []
        matched_type = "other"
        matched_pattern = None
        max_matches = 0
        
        # 检查每个问题类型
        for q_type, pattern in self.QUESTION_PATTERNS.items():
            matches = [kw for kw in pattern["keywords"] if kw.lower() in question_lower]
            if len(matches) > max_matches:
                max_matches = len(matches)
                matched_type = q_type
                matched_pattern = pattern
                identified_keywords = matches
        
        if matched_pattern:
            # 合并所有推荐的参数
            recommended_parameters = {}
            for group_name, params in matched_pattern["parameter_groups"].items():
                recommended_parameters.update(params)
            
            # 生成配置文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"{matched_type}_policy_{timestamp}"
            
            return QuestionAnalysis(
                question_type=matched_type,
                question_text=question,
                identified_keywords=identified_keywords,
                recommended_parameters=recommended_parameters,
                recommended_config_name=config_name,
                description=matched_pattern["description"]
            )
        else:
            # 未识别的问题类型，返回默认配置
            return QuestionAnalysis(
                question_type="other",
                question_text=question,
                identified_keywords=[],
                recommended_parameters={},
                recommended_config_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="未识别的实验类型，使用默认配置"
            )
    
    def generate_config_from_question(self, question: str, base_config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        根据问题生成配置并返回参数
        
        Args:
            question: 问题文本
            base_config_name: 基础配置名称（可选，如果提供则基于此配置修改）
        
        Returns:
            包含配置信息的字典
        """
        analysis = self.analyze_question(question)
        
        result = {
            "success": True,
            "question_analysis": {
                "question_type": analysis.question_type,
                "question_text": analysis.question_text,
                "identified_keywords": analysis.identified_keywords,
                "description": analysis.description,
            },
            "config_name": analysis.recommended_config_name,
            "parameters": analysis.recommended_parameters,
            "guidance": self._get_guidance(analysis.question_type),
        }
        
        return result
    
    def _get_guidance(self, question_type: str) -> str:
        """获取问题类型的指导说明"""
        guidance_map = {
            "innovation": """
            创新促进政策实验配置指导：
            1. 提高创新相关参数：innovation_probability（创新概率）、innovation_*_improvement（创新改进幅度）
            2. 可以通过enable_innovation控制是否启用创新机制
            3. 建议同时启用竞争市场模式（enable_competitive_market）以观察创新对市场份额的影响
            4. 分析指标：创新事件数、创新与市场占有率的相关性、GDP增长、企业生产率
            """,
            "redistribution": """
            全民基本收入政策实验配置指导：
            1. 调整再分配策略：redistribution_strategy（设置为progressive或aggressive）
            2. 增加再分配权重：redistribution_poverty_weight、redistribution_unemployment_weight等
            3. 可以调整个人所得税率（income_tax_rate）为再分配提供资金来源
            4. 分析指标：消费者总支出、总属性值、GDP、收入分布（基尼系数）、家庭储蓄率
            """,
            "labor_productivity": """
            AI重塑劳动力市场实验配置指导：
            1. 提高生产效率：labor_productivity_factor（劳动生产率因子，默认100，可提高到150-200）
            2. 调整劳动弹性：labor_elasticity（提高劳动弹性，如0.8-0.9）
            3. 调整裁员率：dismissal_rate（AI提高效率可能降低裁员需求）
            4. 启用动态招聘：enable_dynamic_job_posting
            5. 分析指标：劳动生产率、失业率、GDP、企业利润率、市场占有率
            """,
            "tariff": """
            关税政策冲击实验配置指导：
            1. 提高税率：vat_rate（增值税率）、corporate_tax_rate（企业所得税率）
            2. 启用价格调整：enable_price_adjustment，提高price_adjustment_rate
            3. 可以调整市场消费率：inherent_market_consumption_rate
            4. 分析指标：GDP、总支出、价格变化、企业利润、市场占有率变化
            """,
            "other": """
            未识别的实验类型，建议：
            1. 手动设置相关参数
            2. 参考现有配置模板
            3. 根据问题关键词调整相应的政策参数
            """
        }
        return guidance_map.get(question_type, guidance_map["other"]).strip()
    
    def get_experiment_workflow(self, question: str) -> Dict[str, Any]:
        """
        获取完整的实验工作流指导
        
        Args:
            question: 问题文本
        
        Returns:
            工作流步骤和指导
        """
        analysis = self.analyze_question(question)
        
        workflow = {
            "question": question,
            "question_type": analysis.question_type,
            "steps": [
                {
                    "step": 1,
                    "action": "analyze_question",
                    "description": f"分析问题类型：{analysis.question_type}",
                    "result": {
                        "keywords": analysis.identified_keywords,
                        "description": analysis.description
                    }
                },
                {
                    "step": 2,
                    "action": "generate_config",
                    "description": f"生成YAML配置文件：{analysis.recommended_config_name}",
                    "parameters": analysis.recommended_parameters,
                    "tool": "save_current_config_to_yaml",
                    "tool_params": {
                        "config_name": analysis.recommended_config_name,
                        "description": analysis.description
                    }
                },
                {
                    "step": 3,
                    "action": "load_config",
                    "description": f"加载配置：{analysis.recommended_config_name}",
                    "tool": "load_yaml_config",
                    "tool_params": {
                        "config_name": analysis.recommended_config_name
                    }
                },
                {
                    "step": 4,
                    "action": "start_simulation",
                    "description": "启动仿真实验",
                    "tool": "start_simulation",
                    "tool_params": {}
                },
                {
                    "step": 5,
                    "action": "monitor_simulation",
                    "description": "监控仿真状态",
                    "tool": "get_simulation_status",
                    "tool_params": {}
                },
                {
                    "step": 6,
                    "action": "capture_experiment",
                    "description": "捕捉实验目录（当仿真完成后）",
                    "tool": "capture_experiment",
                    "tool_params": {
                        "experiment_name": "exp_XXXh_XXm_YYYYMMDD_HHMMSS",  # 从仿真输出获取
                        "status": "completed"
                    }
                },
                {
                    "step": 7,
                    "action": "analyze_experiment",
                    "description": "分析实验数据",
                    "tool": "analyze_experiment",
                    "tool_params": {
                        "experiment_name": "exp_XXXh_XXm_YYYYMMDD_HHMMSS",
                        "include_innovation": analysis.question_type == "innovation"
                    },
                    "metrics_to_check": {
                        "macro": ["gdp", "total_expenditure", "total_attribute_value"],
                        "micro": ["firm_product_quantity", "firm_product_quality"],
                        "innovation": ["market_share_correlation", "innovation_events"] if analysis.question_type == "innovation" else None
                    }
                },
                {
                    "step": 8,
                    "action": "get_results",
                    "description": "获取分析结果",
                    "tool": "get_analysis_result",
                    "tool_params": {
                        "experiment_name": "exp_XXXh_XXm_YYYYMMDD_HHMMSS"
                    }
                }
            ],
            "guidance": self._get_guidance(analysis.question_type),
            "recommended_config_name": analysis.recommended_config_name,
            "recommended_parameters": analysis.recommended_parameters
        }
        
        return workflow


# 全局实例
_automated_tools_instance: Optional[AutomatedExperimentTools] = None


def get_automated_tools() -> AutomatedExperimentTools:
    """获取自动化实验工具实例"""
    global _automated_tools_instance
    if _automated_tools_instance is None:
        _automated_tools_instance = AutomatedExperimentTools()
    return _automated_tools_instance

