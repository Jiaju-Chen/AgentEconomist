"""历史实验分析 MCP 工具"""
import json
from typing import Optional, List
from .experiment_reader import ExperimentRecordReader
from .report_generator import ReportGenerator

def register_tools(mcp):
    reader = ExperimentRecordReader()
    report_gen = ReportGenerator(reader)
    
    @mcp.tool()
    async def list_history_experiments(limit: int = 20) -> str:
        """列出所有历史实验记录"""
        try:
            experiments = reader.list_experiments(limit=limit)
            result = {
                "success": True,
                "total": len(experiments),
                "experiments": [
                    {
                        "experiment_name": exp.experiment_name,
                        "created_time": exp.created_time,
                        "total_months": exp.total_months,
                        "has_report": exp.has_report,
                        "has_monthly_stats": exp.has_monthly_stats,
                        "has_figures": exp.has_figures
                    }
                    for exp in experiments
                ]
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    @mcp.tool()
    async def generate_experiment_report(experiment_name: str, format: str = "markdown") -> str:
        """生成历史实验的详细报告"""
        try:
            report = report_gen.generate_experiment_summary_report(experiment_name, format=format)
            if format == "json":
                return report
            else:
                return json.dumps({
                    "success": True,
                    "experiment_name": experiment_name,
                    "format": format,
                    "report": report
                }, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    @mcp.tool()
    async def get_experiment_summary(experiment_name: str) -> str:
        """获取实验的简要摘要信息"""
        try:
            summary = reader.get_experiment_summary(experiment_name)
            if not summary:
                return json.dumps({"success": False, "error": f"未找到实验 {experiment_name}"}, ensure_ascii=False)
            return json.dumps({"success": True, **summary}, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    @mcp.tool()
    async def get_experiment_timeseries(experiment_name: str, metric: str) -> str:
        """
        获取实验的时间序列数据
        
        支持的指标:
        - employment_rate: 就业率
        - unemployment_rate: 失业率
        - average_income: 平均收入
        - average_expenditure: 平均支出
        - savings_rate: 储蓄率
        - gini_coefficient: 基尼系数
        - average_wealth: 平均财富
        """
        try:
            timeseries = reader.get_timeseries_data(experiment_name, metric)
            if timeseries is None:
                return json.dumps({
                    "success": False, 
                    "error": f"未找到实验 {experiment_name} 或指标 {metric} 不存在"
                }, ensure_ascii=False)
            
            return json.dumps({
                "success": True,
                "experiment_name": experiment_name,
                "metric": metric,
                "data": timeseries,
                "months": len(timeseries)
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    @mcp.tool()
    async def compare_experiments(experiment_names: List[str], metrics: Optional[List[str]] = None, format: str = "markdown") -> str:
        """对比多个历史实验"""
        # TODO: 实现实验对比功能
        return json.dumps({"success": False, "error": "Not implemented yet"}, ensure_ascii=False)
    
    @mcp.tool()
    async def get_monthly_statistics(experiment_name: str, stat_type: str) -> str:
        """
        读取实验的月度统计数据
        
        支持的统计类型:
        - economic: 经济指标历史
        - household: 家庭月度指标
        - firm: 企业月度指标
        - performance: 性能指标
        - dismissal: 辞退统计
        """
        try:
            stats = reader.read_monthly_statistics(experiment_name, stat_type)
            if stats is None:
                return json.dumps({
                    "success": False,
                    "error": f"未找到实验 {experiment_name} 或统计类型 {stat_type} 不存在"
                }, ensure_ascii=False)
            
            return json.dumps({
                "success": True,
                "experiment_name": experiment_name,
                "stat_type": stat_type,
                "data": stats
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
