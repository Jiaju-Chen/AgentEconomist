"""
实验分析工具 - 捕捉实验目录并调用分析器进行分析
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加economist项目到路径
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
economist_dir = project_root.parent / "economist"
if str(economist_dir) not in sys.path:
    sys.path.insert(0, str(economist_dir))

try:
    from experiment_analyzer import ExperimentAnalyzer, AnalysisConfig
    ANALYZER_AVAILABLE = True
except ImportError as e:
    ANALYZER_AVAILABLE = False
    print(f"⚠️  实验分析器不可用: {e}")


@dataclass
class ExperimentManifest:
    """实验清单"""
    experiment_name: str
    experiment_dir: str
    created_time: str
    status: str  # pending, running, completed
    analysis_status: str  # not_started, in_progress, completed, failed
    analysis_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ExperimentAnalyzerTools:
    """实验分析工具类"""
    
    def __init__(self, output_dir: Optional[str] = None, manifest_file: str = "experiment_manifest.json"):
        """
        初始化实验分析工具
        
        Args:
            output_dir: 实验输出目录（默认：项目根目录/output）
            manifest_file: manifest文件名
        """
        if output_dir is None:
            current_file = Path(__file__)
            # 从 mcp_server/tools/ 找到项目根目录
            # 路径结构: agentsociety-ecosim/agentsociety_ecosim/mcp_server/tools/
            project_root = current_file.parent.parent.parent.parent  # 到 agentsociety-ecosim/
            self.output_dir = project_root / "output"
        else:
            self.output_dir = Path(output_dir)
        
        self.manifest_file = self.output_dir / manifest_file
        self.manifests: Dict[str, ExperimentManifest] = {}
        self._load_manifest()
    
    def _load_manifest(self):
        """加载实验清单"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for exp_name, exp_data in data.items():
                        self.manifests[exp_name] = ExperimentManifest(**exp_data)
            except Exception as e:
                print(f"⚠️  加载manifest失败: {e}")
                self.manifests = {}
        else:
            self.manifests = {}
    
    def _save_manifest(self):
        """保存实验清单"""
        try:
            data = {name: asdict(manifest) for name, manifest in self.manifests.items()}
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存manifest失败: {e}")
    
    def capture_experiment(self, experiment_name: str, experiment_dir: Optional[str] = None, 
                          status: str = "pending") -> Dict[str, Any]:
        """
        捕捉实验目录并保存到manifest
        
        Args:
            experiment_name: 实验名称（如 exp_100h_12m_20251121_221420）
            experiment_dir: 实验目录路径（如果为None，则使用 output_dir / experiment_name）
            status: 实验状态（pending, running, completed）
        
        Returns:
            操作结果
        """
        if experiment_dir is None:
            experiment_dir = str(self.output_dir / experiment_name)
        else:
            experiment_dir = str(Path(experiment_dir).resolve())
        
        # 检查目录是否存在
        if not Path(experiment_dir).exists():
            return {
                "success": False,
                "error": f"实验目录不存在: {experiment_dir}",
                "experiment_name": experiment_name
            }
        
        # 创建或更新manifest
        manifest = ExperimentManifest(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            created_time=datetime.now().isoformat(),
            status=status,
            analysis_status="not_started"
        )
        
        self.manifests[experiment_name] = manifest
        self._save_manifest()
        
        return {
            "success": True,
            "message": f"已捕捉实验: {experiment_name}",
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "status": status
        }
    
    def update_experiment_status(self, experiment_name: str, status: str) -> Dict[str, Any]:
        """
        更新实验状态
        
        Args:
            experiment_name: 实验名称
            status: 新状态（pending, running, completed）
        
        Returns:
            操作结果
        """
        if experiment_name not in self.manifests:
            return {
                "success": False,
                "error": f"实验 {experiment_name} 不在manifest中"
            }
        
        self.manifests[experiment_name].status = status
        self._save_manifest()
        
        return {
            "success": True,
            "message": f"已更新实验状态: {experiment_name} -> {status}",
            "experiment_name": experiment_name,
            "status": status
        }
    
    def analyze_experiment(self, experiment_name: str, 
                          innovation_types: Optional[List[str]] = None,
                          include_innovation: bool = True) -> Dict[str, Any]:
        """
        分析实验数据
        
        Args:
            experiment_name: 实验名称
            innovation_types: 创新类型列表（如果为None，默认包含主要类型）
            include_innovation: 是否包含创新分析
        
        Returns:
            分析结果
        """
        if not ANALYZER_AVAILABLE:
            return {
                "success": False,
                "error": "实验分析器不可用，请检查experiment_analyzer模块"
            }
        
        if experiment_name not in self.manifests:
            return {
                "success": False,
                "error": f"实验 {experiment_name} 不在manifest中，请先调用capture_experiment"
            }
        
        manifest = self.manifests[experiment_name]
        experiment_dir = manifest.experiment_dir
        
        if not Path(experiment_dir).exists():
            return {
                "success": False,
                "error": f"实验目录不存在: {experiment_dir}"
            }
        
        try:
            # 更新分析状态
            manifest.analysis_status = "in_progress"
            self._save_manifest()
            
            # 创建分析器
            analyzer = ExperimentAnalyzer(experiment_dir)
            
            # 准备分析配置
            metrics_to_analyze = ["consumer_metrics", "firm_product_metrics", "macro_metrics"]
            if include_innovation:
                metrics_to_analyze.append("innovation_analysis")
            
            # 如果未指定创新类型，使用默认值
            if innovation_types is None and include_innovation:
                innovation_types = [
                    "labor_productivity_factor",
                    "price",
                    "profit_margin",
                ]
            
            config = AnalysisConfig(
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                metrics_to_analyze=metrics_to_analyze,
                innovation_types=innovation_types,
                market_share_type="quantity_share_pct"
            )
            
            # 执行分析
            metrics = analyzer.analyze_all_metrics(config)
            
            # 提取关键指标
            result = {
                "experiment_name": experiment_name,
                "analysis_time": datetime.now().isoformat(),
            }
            
            # 宏观指标
            if metrics.macro_metrics:
                result["macro_metrics"] = {
                    "gdp": metrics.macro_metrics.gdp,
                    "total_revenue": metrics.macro_metrics.total_revenue,
                    "total_expenditure": metrics.macro_metrics.total_expenditure,
                }
            
            # 消费者指标
            if metrics.consumer_metrics:
                result["consumer_metrics"] = {
                    "total_expenditure": metrics.consumer_metrics.total_expenditure,
                    "total_attribute_value": metrics.consumer_metrics.total_attribute_value,
                    "total_nutrition_value": metrics.consumer_metrics.total_nutrition_value,
                    "total_satisfaction_value": metrics.consumer_metrics.total_satisfaction_value,
                }
            
            # 微观指标：企业商品产量和质量
            if metrics.firm_product_metrics:
                result["micro_metrics"] = {
                    "total_food_quantity": metrics.firm_product_metrics.total_food_quantity,
                    "total_nonfood_quantity": metrics.firm_product_metrics.total_nonfood_quantity,
                    "avg_food_quality": metrics.firm_product_metrics.avg_food_quality,
                    "avg_nonfood_quality": metrics.firm_product_metrics.avg_nonfood_quality,
                    "firm_count": len(metrics.firm_product_metrics.firm_products),
                }
            
            # 创新指标：创新后n个月市场占有率增量相关系数
            if metrics.innovation_analysis and include_innovation:
                innovation_corr = metrics.innovation_analysis.innovation_market_share_correlation
                per_horizon = innovation_corr.get("per_horizon", {})
                
                result["innovation_metrics"] = {
                    "total_innovation_events": metrics.innovation_analysis.innovation_events_count,
                    "market_share_correlation": {}
                }
                
                for horizon, summary in per_horizon.items():
                    result["innovation_metrics"]["market_share_correlation"][horizon] = {
                        "correlation": summary.get("correlation"),
                        "avg_ratio": summary.get("avg_ratio"),
                        "data_points": summary.get("data_points"),
                    }
            
            # 保存结果到manifest
            manifest.analysis_status = "completed"
            manifest.analysis_result = result
            self._save_manifest()
            
            return {
                "success": True,
                "message": f"实验 {experiment_name} 分析完成",
                "result": result
            }
            
        except Exception as e:
            manifest.analysis_status = "failed"
            manifest.error_message = str(e)
            self._save_manifest()
            
            return {
                "success": False,
                "error": f"分析失败: {str(e)}",
                "experiment_name": experiment_name
            }
    
    def list_experiments(self) -> Dict[str, Any]:
        """列出所有实验"""
        return {
            "success": True,
            "experiments": {
                name: {
                    "experiment_name": manifest.experiment_name,
                    "experiment_dir": manifest.experiment_dir,
                    "status": manifest.status,
                    "analysis_status": manifest.analysis_status,
                    "created_time": manifest.created_time,
                }
                for name, manifest in self.manifests.items()
            }
        }
    
    def get_analysis_result(self, experiment_name: str) -> Dict[str, Any]:
        """获取分析结果"""
        if experiment_name not in self.manifests:
            return {
                "success": False,
                "error": f"实验 {experiment_name} 不在manifest中"
            }
        
        manifest = self.manifests[experiment_name]
        
        return {
            "success": True,
            "experiment_name": experiment_name,
            "analysis_status": manifest.analysis_status,
            "analysis_result": manifest.analysis_result,
            "error_message": manifest.error_message,
        }


# 全局实例
_analyzer_tools_instance: Optional[ExperimentAnalyzerTools] = None


def get_analyzer_tools() -> ExperimentAnalyzerTools:
    """获取实验分析工具实例"""
    global _analyzer_tools_instance
    if _analyzer_tools_instance is None:
        _analyzer_tools_instance = ExperimentAnalyzerTools()
    return _analyzer_tools_instance

