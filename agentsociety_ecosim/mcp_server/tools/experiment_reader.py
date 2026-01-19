"""实验记录读取器"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class ExperimentMetadata:
    experiment_name: str
    created_time: str
    total_months: int = 0
    has_report: bool = False
    has_monthly_stats: bool = False
    has_figures: bool = False

class ExperimentRecordReader:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.output_dir = project_root / "output"
        else:
            self.output_dir = Path(output_dir)
    
    def list_experiments(self, limit: int = 50) -> List[ExperimentMetadata]:
        if not self.output_dir.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(self.output_dir.glob("experiment_*"), reverse=True):
            if not exp_dir.is_dir():
                continue
            
            meta = ExperimentMetadata(
                experiment_name=exp_dir.name,
                created_time=exp_dir.name.replace("experiment_", ""),
                total_months=len(list((exp_dir / "monthly_statistics").glob("month_*.json"))) if (exp_dir / "monthly_statistics").exists() else 0,
                has_report=bool(list(exp_dir.glob("simulation_report_*.json"))),
                has_monthly_stats=(exp_dir / "monthly_statistics").exists(),
                has_figures=(exp_dir / "charts").exists() or (exp_dir / "output_fig").exists()
            )
            experiments.append(meta)
            
            if len(experiments) >= limit:
                break
        
        return experiments
    
    def read_experiment_report(self, experiment_name: str) -> Optional[Dict]:
        exp_dir = self.output_dir / experiment_name
        if not exp_dir.exists():
            return None
        
        # 首先尝试读取 simulation_report_*.json
        report_files = list(exp_dir.glob("simulation_report_*.json"))
        if report_files:
            with open(report_files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有报告文件，尝试从 data/ 目录构建报告
        data_dir = exp_dir / "data"
        if data_dir.exists():
            report = {}
            
            # 读取 economic_metrics_history.json
            econ_file = data_dir / "economic_metrics_history.json"
            if econ_file.exists():
                with open(econ_file, 'r', encoding='utf-8') as f:
                    report['economic_metrics'] = json.load(f)
            
            # 读取 household_monthly_metrics.json
            household_file = data_dir / "household_monthly_metrics.json"
            if household_file.exists():
                with open(household_file, 'r', encoding='utf-8') as f:
                    report['household_metrics'] = json.load(f)
            
            # 读取 firm_monthly_metrics.json
            firm_file = data_dir / "firm_monthly_metrics.json"
            if firm_file.exists():
                with open(firm_file, 'r', encoding='utf-8') as f:
                    report['firm_metrics'] = json.load(f)
            
            # 读取 performance_metrics.json
            perf_file = data_dir / "performance_metrics.json"
            if perf_file.exists():
                with open(perf_file, 'r', encoding='utf-8') as f:
                    report['performance'] = json.load(f)
            
            return report if report else None
        
        return None
    
    def get_experiment_summary(self, experiment_name: str) -> Optional[Dict]:
        report = self.read_experiment_report(experiment_name)
        if not report:
            return None
        
        # 如果是标准格式的报告
        if "summary_statistics" in report:
            return {
                "experiment_name": experiment_name,
                "total_households": report.get("summary_statistics", {}).get("total_households", 0),
                "total_firms": report.get("summary_statistics", {}).get("total_firms", 0),
                "total_months": report.get("simulation_info", {}).get("total_months", 0)
            }
        
        # 如果是从 data/ 构建的报告
        if "economic_metrics" in report:
            metrics = report['economic_metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                last_month = metrics[-1]
                return {
                    "experiment_name": experiment_name,
                    "total_households": last_month.get("employment_statistics", {}).get("total_households", 0),
                    "total_firms": 0,  # TODO: 从 firm_metrics 获取
                    "total_months": len(metrics),
                    "final_employment_rate": last_month.get("employment_statistics", {}).get("household_employment_rate", 0),
                    "final_unemployment_rate": last_month.get("employment_statistics", {}).get("household_unemployment_rate", 0),
                    "final_average_income": last_month.get("income_expenditure_analysis", {}).get("average_monthly_income", 0),
                    "final_average_expenditure": last_month.get("income_expenditure_analysis", {}).get("average_monthly_expenditure", 0),
                    "final_gini_coefficient": last_month.get("wealth_distribution", {}).get("gini_coefficient", 0),
                    "final_savings_rate": last_month.get("income_expenditure_analysis", {}).get("monthly_savings_rate", 0)
                }
        
        return None
    
    def get_timeseries_data(self, experiment_name: str, metric: str) -> Optional[List[Any]]:
        """获取时间序列数据"""
        report = self.read_experiment_report(experiment_name)
        if not report or "economic_metrics" not in report:
            return None
        
        metrics = report['economic_metrics']
        if not isinstance(metrics, list):
            return None
        
        # 支持的指标路径映射
        metric_paths = {
            "employment_rate": ("employment_statistics", "household_employment_rate"),
            "unemployment_rate": ("employment_statistics", "household_unemployment_rate"),
            "average_income": ("income_expenditure_analysis", "average_monthly_income"),
            "average_expenditure": ("income_expenditure_analysis", "average_monthly_expenditure"),
            "savings_rate": ("income_expenditure_analysis", "monthly_savings_rate"),
            "gini_coefficient": ("wealth_distribution", "gini_coefficient"),
            "average_wealth": ("wealth_distribution", "average_wealth")
        }
        
        if metric not in metric_paths:
            return None
        
        path = metric_paths[metric]
        timeseries = []
        for month_data in metrics:
            value = month_data
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else None
                if value is None:
                    break
            timeseries.append(value)
        
        return timeseries
    
    def read_monthly_statistics(self, experiment_name: str, stat_type: str) -> Optional[Dict]:
        """读取月度统计数据"""
        exp_dir = self.output_dir / experiment_name / "data"
        if not exp_dir.exists():
            return None
        
        # 支持的统计类型文件映射
        stat_files = {
            "economic": "economic_metrics_history.json",
            "household": "household_monthly_metrics.json",
            "firm": "firm_monthly_metrics.json",
            "performance": "performance_metrics.json",
            "dismissal": "dismissal_stats.json"
        }
        
        if stat_type not in stat_files:
            return None
        
        file_path = exp_dir / stat_files[stat_type]
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
