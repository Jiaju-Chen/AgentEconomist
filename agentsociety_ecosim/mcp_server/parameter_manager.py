#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数管理器 - 管理仿真系统的所有配置参数

职责：
1. 参数的获取、设置、验证
2. 参数预设的保存和加载
3. 参数元数据管理（类型、范围、描述等）
4. YAML配置文件加载
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ParameterMetadata:
    """参数元数据"""
    name: str
    description: str
    type: str
    category: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    unit: str = ""
    impact: str = ""
    economic_impact: Optional[str] = None
    impact_level: Optional[str] = None
    formula: Optional[str] = None


@dataclass
class ValidationResult:
    """参数验证结果"""
    success: bool
    valid: bool
    old_value: Any
    new_value: Any
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ParameterPreset:
    """参数预设"""
    name: str
    description: str
    parameters: Dict[str, Any]
    created_at: str


class ParameterManager:
    """
    参数管理器 - 单例模式
    
    管理经济仿真系统的所有参数
    """
    
    _instance = None
    
    def __init__(self, config: Any):
        """
        初始化参数管理器
        
        Args:
            config: 配置对象（SimpleConfig或SimulationConfig）
        """
        if ParameterManager._instance is not None:
            raise RuntimeError("Use ParameterManager.get_instance() instead")
        
        self.config = config
        self.metadata: Dict[str, ParameterMetadata] = {}
        self.presets_dir = Path(__file__).parent / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        
        # YAML配置文件目录
        self.config_dir = Path(__file__).parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # 初始化参数元数据
        self._init_metadata()
        
        ParameterManager._instance = self
    
    @classmethod
    def get_instance(cls, config: Any = None):
        """获取单例实例"""
        if cls._instance is None:
            if config is None:
                raise ValueError("First call to get_instance() requires config")
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例"""
        cls._instance = None
    
    def _init_metadata(self):
        """初始化参数元数据"""
        # 税收政策
        self.metadata.update({
            "income_tax_rate": ParameterMetadata(
                name="income_tax_rate",
                description="个人所得税率",
                type="float",
                category="tax_policy",
                min_value=0.0,
                max_value=1.0,
                default_value=0.45,
                unit="比例",
                impact="影响家庭可支配收入和政府税收"
            ),
            "vat_rate": ParameterMetadata(
                name="vat_rate",
                description="增值税率",
                type="float",
                category="tax_policy",
                min_value=0.0,
                max_value=1.0,
                default_value=0.20,
                unit="比例"
            ),
            "corporate_tax_rate": ParameterMetadata(
                name="corporate_tax_rate",
                description="企业所得税率",
                type="float",
                category="tax_policy",
                min_value=0.0,
                max_value=1.0,
                default_value=0.42,
                unit="比例"
            ),
        })
        
        # 劳动力市场
        self.metadata.update({
            "dismissal_rate": ParameterMetadata(
                name="dismissal_rate",
                description="裁员率",
                type="float",
                category="labor_market",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1
            ),
            "enable_dismissal": ParameterMetadata(
                name="enable_dismissal",
                description="是否启用裁员机制",
                type="bool",
                category="labor_market",
                default_value=True
            ),
            "unemployment_threshold": ParameterMetadata(
                name="unemployment_threshold",
                description="失业率阈值",
                type="float",
                category="labor_market",
                min_value=0.0,
                max_value=1.0,
                default_value=0.4
            ),
        })
        
        # 生产参数
        self.metadata.update({
            "labor_productivity_factor": ParameterMetadata(
                name="labor_productivity_factor",
                description="劳动生产率因子",
                type="float",
                category="production",
                min_value=0.0,
                default_value=100.0
            ),
            "labor_elasticity": ParameterMetadata(
                name="labor_elasticity",
                description="劳动弹性",
                type="float",
                category="production",
                min_value=0.0,
                max_value=1.0,
                default_value=0.7
            ),
        })
        
        # 系统规模
        self.metadata.update({
            "num_households": ParameterMetadata(
                name="num_households",
                description="家庭数量",
                type="int",
                category="system_scale",
                min_value=1,
                default_value=100
            ),
            "num_iterations": ParameterMetadata(
                name="num_iterations",
                description="仿真迭代次数（月数）",
                type="int",
                category="system_scale",
                min_value=1,
                default_value=12
            ),
            "max_concurrent_tasks": ParameterMetadata(
                name="max_concurrent_tasks",
                description="最大并发任务数",
                type="int",
                category="performance",
                min_value=1,
                default_value=300
            ),
            "max_llm_concurrent": ParameterMetadata(
                name="max_llm_concurrent",
                description="LLM最大并发数",
                type="int",
                category="performance",
                min_value=1,
                default_value=600
            ),
            "max_firm_concurrent": ParameterMetadata(
                name="max_firm_concurrent",
                description="企业最大并发数",
                type="int",
                category="performance",
                min_value=1,
                default_value=50
            ),
        })
        
        # 再分配策略
        self.metadata.update({
            "redistribution_strategy": ParameterMetadata(
                name="redistribution_strategy",
                description="再分配策略",
                type="str",
                category="redistribution",
                default_value="none"
            ),
            "redistribution_poverty_weight": ParameterMetadata(
                name="redistribution_poverty_weight",
                description="贫困权重",
                type="float",
                category="redistribution",
                min_value=0.0,
                max_value=1.0,
                default_value=0.3
            ),
            "redistribution_unemployment_weight": ParameterMetadata(
                name="redistribution_unemployment_weight",
                description="失业权重",
                type="float",
                category="redistribution",
                min_value=0.0,
                max_value=1.0,
                default_value=0.2
            ),
            "redistribution_family_size_weight": ParameterMetadata(
                name="redistribution_family_size_weight",
                description="家庭规模权重",
                type="float",
                category="redistribution",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1
            ),
        })
        
        # 批量LLM优化配置
        self.metadata.update({
            "use_batch_budget_allocation": ParameterMetadata(
                name="use_batch_budget_allocation",
                description="是否使用批量预算分配",
                type="bool",
                category="performance",
                default_value=False
            ),
            "batch_size": ParameterMetadata(
                name="batch_size",
                description="批量处理的家庭数量",
                type="int",
                category="performance",
                min_value=1,
                default_value=10
            ),
            "batch_llm_timeout": ParameterMetadata(
                name="batch_llm_timeout",
                description="批量LLM请求超时时间（秒）",
                type="int",
                category="performance",
                min_value=1,
                default_value=120
            ),
        })
        
        # 生产与补货配置
        self.metadata.update({
            "profit_to_production_ratio": ParameterMetadata(
                name="profit_to_production_ratio",
                description="利润转化为生产预算的比例",
                type="float",
                category="production",
                min_value=0.0,
                max_value=1.0,
                default_value=0.7
            ),
            "min_production_per_product": ParameterMetadata(
                name="min_production_per_product",
                description="每个商品最小生产量",
                type="float",
                category="production",
                min_value=0.0,
                default_value=5.0
            ),
            "base_production_rate": ParameterMetadata(
                name="base_production_rate",
                description="[已废弃]基础生产率",
                type="float",
                category="production",
                min_value=0.0,
                default_value=100.0
            ),
            "high_demand_multiplier": ParameterMetadata(
                name="high_demand_multiplier",
                description="[已废弃]高需求商品补货倍数",
                type="float",
                category="production",
                min_value=0.0,
                default_value=1.5
            ),
            "low_demand_multiplier": ParameterMetadata(
                name="low_demand_multiplier",
                description="[已废弃]低需求商品补货倍数",
                type="float",
                category="production",
                min_value=0.0,
                default_value=0.7
            ),
        })
        
        # 监控配置
        self.metadata.update({
            "monitor_interval": ParameterMetadata(
                name="monitor_interval",
                description="监控间隔（秒）",
                type="float",
                category="monitoring",
                min_value=0.1,
                default_value=5.0
            ),
            "enable_monitoring": ParameterMetadata(
                name="enable_monitoring",
                description="是否启用监控",
                type="bool",
                category="monitoring",
                default_value=False
            ),
        })
        
        # 工作发布配置
        self.metadata.update({
            "enable_dynamic_job_posting": ParameterMetadata(
                name="enable_dynamic_job_posting",
                description="是否启用动态招聘",
                type="bool",
                category="labor_market",
                default_value=False
            ),
            "first_month_job_rate": ParameterMetadata(
                name="first_month_job_rate",
                description="第一个月发布工作比例",
                type="float",
                category="labor_market",
                min_value=0.0,
                max_value=1.0,
                default_value=0.9
            ),
            "job_posting_multiplier": ParameterMetadata(
                name="job_posting_multiplier",
                description="工作发布倍数",
                type="float",
                category="labor_market",
                min_value=0.0,
                default_value=0.1
            ),
        })
        
        # 商品配置
        self.metadata.update({
            "min_per_cat": ParameterMetadata(
                name="min_per_cat",
                description="每类最少商品数",
                type="int",
                category="market",
                min_value=1,
                default_value=20
            ),
            "multiplier": ParameterMetadata(
                name="multiplier",
                description="商品数量乘数（控制总商品数）",
                type="int",
                category="market",
                min_value=1,
                default_value=12
            ),
            "random_state": ParameterMetadata(
                name="random_state",
                description="随机种子",
                type="int",
                category="system_scale",
                min_value=0,
                default_value=42
            ),
            "enable_competitive_market": ParameterMetadata(
                name="enable_competitive_market",
                description="是否启用竞争市场模式（创新破坏理论）",
                type="bool",
                category="market",
                default_value=False,
                economic_impact="启用后同类企业销售相同商品相互竞争市场份额，关闭后企业销售不同商品无直接竞争",
                impact_level="high",
                formula="竞争模式下同类商品由多家企业供应，通过价格和质量竞争市场份额"
            ),
            "enable_price_adjustment": ParameterMetadata(
                name="enable_price_adjustment",
                description="是否启用价格根据销量自动调整",
                type="bool",
                category="market",
                default_value=True,
                economic_impact="启用后商品价格会根据供需关系自动调整，关闭后价格保持固定",
                impact_level="high"
            ),
            "price_adjustment_rate": ParameterMetadata(
                name="price_adjustment_rate",
                description="价格调整幅度",
                type="float",
                category="market",
                min_value=0.0,
                max_value=0.5,
                default_value=0.1,
                economic_impact="控制价格调整的速度，数值越大价格变动越快",
                impact_level="medium"
            ),
        })
        
        # 固有市场配置
        self.metadata.update({
            "enable_inherent_market": ParameterMetadata(
                name="enable_inherent_market",
                description="是否启用固有市场",
                type="bool",
                category="market",
                default_value=True
            ),
            "inherent_market_consumption_rate": ParameterMetadata(
                name="inherent_market_consumption_rate",
                description="固有市场每月消耗商品的比例",
                type="float",
                category="market",
                min_value=0.0,
                max_value=1.0,
                default_value=0.65
            ),
            "inherent_market_focus_new_products": ParameterMetadata(
                name="inherent_market_focus_new_products",
                description="固有市场是否优先消耗新商品",
                type="bool",
                category="market",
                default_value=True
            ),
        })
    
        # 创新参数
        self.metadata.update({
            "innovation_probability": ParameterMetadata(
                name="innovation_probability",
                description="创新概率（企业每个月尝试创新的概率）",
                type="float",
                category="innovation",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                unit="比例",
                impact="控制企业创新的频率，影响市场竞争动态",
                impact_level="high"
            ),
            "innovation_labor_productivity_improvement": ParameterMetadata(
                name="innovation_labor_productivity_improvement",
                description="劳动生产率创新改进幅度",
                type="float",
                category="innovation",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                unit="比例",
                impact="每次劳动生产率创新的提升幅度",
                impact_level="high"
            ),
            "innovation_price_reduction": ParameterMetadata(
                name="innovation_price_reduction",
                description="价格创新降低幅度",
                type="float",
                category="innovation",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                unit="比例",
                impact="每次价格创新的降低幅度",
                impact_level="high"
            ),
            "innovation_profit_margin_improvement": ParameterMetadata(
                name="innovation_profit_margin_improvement",
                description="利润率创新改进幅度",
                type="float",
                category="innovation",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                unit="比例",
                impact="每次利润率创新的提升幅度",
                impact_level="high"
            ),
            "innovation_attribute_improvement": ParameterMetadata(
                name="innovation_attribute_improvement",
                description="商品属性创新改进幅度",
                type="float",
                category="innovation",
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                unit="比例",
                impact="每次商品属性创新的提升幅度",
                impact_level="medium"
            ),
            "enable_innovation": ParameterMetadata(
                name="enable_innovation",
                description="是否启用创新机制",
                type="bool",
                category="innovation",
                default_value=True,
                impact="控制企业是否可以进行创新",
                impact_level="high"
            ),
        })
    
    def get_parameter(self, parameter_name: str) -> Dict[str, Any]:
        """获取参数信息"""
        if parameter_name not in self.metadata:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        meta = self.metadata[parameter_name]
        current_value = getattr(self.config, parameter_name)
        
        return {
            "name": meta.name,
            "value": current_value,
            "description": meta.description,
            "type": meta.type,
            "category": meta.category,
            "min_value": meta.min_value,
            "max_value": meta.max_value,
            "default_value": meta.default_value,
            "unit": meta.unit
        }
    
    def get_all_parameters(self, category: str = "all", format: str = "json") -> Any:
        """获取所有参数"""
        params = {}
        
        for name, meta in self.metadata.items():
            if category == "all" or meta.category == category:
                params[name] = self.get_parameter(name)
        
        if format == "json":
            return params
        else:
            return params
    
    def set_parameter(self, parameter_name: str, value: Any, validate: bool = True) -> ValidationResult:
        """设置参数"""
        if parameter_name not in self.metadata:
            return ValidationResult(
                success=False,
                valid=False,
                old_value=None,
                new_value=value,
                errors=[f"Unknown parameter: {parameter_name}"]
            )
        
        meta = self.metadata[parameter_name]
        old_value = getattr(self.config, parameter_name)
        
        # 验证
        if validate:
            if meta.type == "float" and not isinstance(value, (int, float)):
                return ValidationResult(
                    success=False,
                    valid=False,
                    old_value=old_value,
                    new_value=value,
                    errors=["Type mismatch: expected float"]
                )
            
            if meta.min_value is not None and value < meta.min_value:
                return ValidationResult(
                    success=False,
                    valid=False,
                    old_value=old_value,
                    new_value=value,
                    errors=[f"Value below minimum: {meta.min_value}"]
                )
            
            if meta.max_value is not None and value > meta.max_value:
                return ValidationResult(
                    success=False,
                    valid=False,
                    old_value=old_value,
                    new_value=value,
                    errors=[f"Value above maximum: {meta.max_value}"]
                )
        
        # 设置
        setattr(self.config, parameter_name, value)
        
        return ValidationResult(
            success=True,
            valid=True,
            old_value=old_value,
            new_value=value
        )
    
    def batch_set_parameters(self, parameters: Dict[str, Any], scenario_name: Optional[str] = None) -> Dict[str, ValidationResult]:
        """批量设置参数"""
        results = {}
        for name, value in parameters.items():
            results[name] = self.set_parameter(name, value)
        return results
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证参数（不修改）"""
        results = {}
        for name, value in parameters.items():
            if name not in self.metadata:
                results[name] = {"valid": False, "error": "Unknown parameter"}
            else:
                meta = self.metadata[name]
                valid = True
                errors = []
                
                if meta.min_value is not None and value < meta.min_value:
                    valid = False
                    errors.append(f"Below minimum: {meta.min_value}")
                
                if meta.max_value is not None and value > meta.max_value:
                    valid = False
                    errors.append(f"Above maximum: {meta.max_value}")
                
                results[name] = {"valid": valid, "errors": errors}
        
        return results
    
    def reset_parameters(self, parameters: Optional[List[str]] = None):
        """重置参数为默认值"""
        params_to_reset = parameters if parameters else list(self.metadata.keys())
        
        for name in params_to_reset:
            if name in self.metadata:
                meta = self.metadata[name]
                setattr(self.config, name, meta.default_value)
    
    def save_preset(self, name: str, parameters: Dict[str, Any], description: str = ""):
        """保存预设"""
        preset = ParameterPreset(
            name=name,
            description=description,
            parameters=parameters,
            created_at=datetime.now().isoformat()
        )
        
        preset_file = self.presets_dir / f"{name}.json"
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(preset), f, indent=2, ensure_ascii=False)
    
    def load_preset(self, name: str) -> ParameterPreset:
        """加载预设"""
        preset_file = self.presets_dir / f"{name}.json"
        
        if not preset_file.exists():
            raise FileNotFoundError(f"Preset not found: {name}")
        
        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ParameterPreset(**data)
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """列出所有预设"""
        presets = []
        
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                preset = self.load_preset(preset_file.stem)
                presets.append({
                    "name": preset.name,
                    "description": preset.description,
                    "created_at": preset.created_at,
                    "parameter_count": len(preset.parameters)
                })
            except Exception:
                continue
        
        return presets
    
    def get_parameter_ranges(self) -> Dict[str, Any]:
        """获取参数范围"""
        ranges = {}
        
        for name, meta in self.metadata.items():
            ranges[name] = {
                "type": meta.type,
                "min": meta.min_value,
                "max": meta.max_value,
                "default": meta.default_value
            }
        
        return ranges
    
    # ==================== YAML配置文件支持 ====================
    
    def load_yaml_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            config_name: 配置文件名（不含.yaml后缀），例如 "default"
            
        Returns:
            加载的参数字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 将嵌套的YAML结构展平为参数字典
        params = self._flatten_yaml_structure(yaml_data)
        
        return params
    
    def _flatten_yaml_structure(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将分组的YAML结构展平为参数字典
        
        例如：
        {
            "tax_policy": {
                "income_tax_rate": 0.45
            }
        }
        转换为：
        {
            "income_tax_rate": 0.45
        }
        """
        params = {}
        
        for category, category_params in yaml_data.items():
            if isinstance(category_params, dict):
                params.update(category_params)
            else:
                # 如果不是字典，直接作为参数（兼容扁平结构）
                params[category] = category_params
        
        return params
    
    def apply_yaml_config(self, config_name: str = "default", validate: bool = True) -> Dict[str, Any]:
        """
        加载并应用YAML配置到当前config对象
        
        Args:
            config_name: 配置文件名
            validate: 是否验证参数
            
        Returns:
            {
                "success": bool,
                "loaded_parameters": int,
                "applied_parameters": int,
                "errors": List[str]
            }
        """
        try:
            # 加载YAML配置
            params = self.load_yaml_config(config_name)
            
            # 应用参数
            results = {}
            errors = []
            
            for param_name, value in params.items():
                # 检查参数是否存在于元数据中
                if param_name not in self.metadata:
                    errors.append(f"Unknown parameter: {param_name}")
                    continue
                
                # 使用set_parameter应用参数（包含验证）
                result = self.set_parameter(param_name, value, validate=validate)
                results[param_name] = result
                
                if not result.success:
                    errors.extend(result.errors)
            
            applied_count = sum(1 for r in results.values() if r.success)
            
            return {
                "success": len(errors) == 0,
                "config_name": config_name,
                "loaded_parameters": len(params),
                "applied_parameters": applied_count,
                "errors": errors,
                "details": {name: {"success": r.success, "old_value": r.old_value, "new_value": r.new_value} 
                           for name, r in results.items()}
            }
            
        except FileNotFoundError as e:
            return {
                "success": False,
                "config_name": config_name,
                "loaded_parameters": 0,
                "applied_parameters": 0,
                "errors": [str(e)]
            }
        except Exception as e:
            return {
                "success": False,
                "config_name": config_name,
                "loaded_parameters": 0,
                "applied_parameters": 0,
                "errors": [f"Failed to load config: {str(e)}"]
            }
    
    def list_yaml_configs(self) -> List[Dict[str, str]]:
        """
        列出所有可用的YAML配置文件
        
        Returns:
            配置文件列表，每个包含 name 和 path
        """
        configs = []
        
        if not self.config_dir.exists():
            return configs
        
        for yaml_file in self.config_dir.glob("*.yaml"):
            configs.append({
                "name": yaml_file.stem,
                "path": str(yaml_file),
                "size": yaml_file.stat().st_size
            })
        
        return configs
    
    def save_current_config_to_yaml(self, config_name: str, description: str = "") -> Dict[str, Any]:
        """
        将当前配置保存为YAML文件
        
        Args:
            config_name: 配置文件名（不含.yaml后缀）
            description: 配置描述（作为YAML注释）
            
        Returns:
            保存结果
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # 收集当前所有参数值，按类别组织
        yaml_structure = {
            "tax_policy": {},
            "labor_market": {},
            "production": {},
            "market": {},
            "system_scale": {},
            "redistribution": {},
            "performance": {},
            "monitoring": {}
        }
        
        for param_name, meta in self.metadata.items():
            current_value = getattr(self.config, param_name, None)
            category = meta.category
            
            # 映射category到YAML结构的key
            if category == "tax_policy":
                yaml_structure["tax_policy"][param_name] = current_value
            elif category == "labor_market":
                yaml_structure["labor_market"][param_name] = current_value
            elif category == "production":
                yaml_structure["production"][param_name] = current_value
            elif category == "market":
                yaml_structure["market"][param_name] = current_value
            elif category == "system_scale":
                yaml_structure["system_scale"][param_name] = current_value
            elif category == "redistribution":
                yaml_structure["redistribution"][param_name] = current_value
            elif category == "performance":
                yaml_structure["performance"][param_name] = current_value
            elif category == "monitoring":
                yaml_structure["monitoring"][param_name] = current_value
        
        # 移除空分类
        yaml_structure = {k: v for k, v in yaml_structure.items() if v}
        
        # 写入YAML文件
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if description:
                    f.write(f"# {description}\n")
                    f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
                
                yaml.dump(yaml_structure, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            return {
                "success": True,
                "message": f"Config saved to {config_file}",
                "path": str(config_file)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to save config: {str(e)}"
            }
