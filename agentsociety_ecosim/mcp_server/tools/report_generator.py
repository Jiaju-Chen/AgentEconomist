"""报告生成器"""
import json
from typing import List, Optional
from .experiment_reader import ExperimentRecordReader

class ReportGenerator:
    def __init__(self, reader: ExperimentRecordReader):
        self.reader = reader
    
    def generate_experiment_summary_report(self, experiment_name: str, format: str = "markdown") -> str:
        report_data = self.reader.read_experiment_report(experiment_name)
        
        if not report_data:
            return "实验报告未找到"
        
        if format == "json":
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        
        # Markdown 格式
        md = f"""# 实验报告: {experiment_name}

## 基本信息
- 实验名称: {experiment_name}
- 总月数: {report_data.get('simulation_info', {}).get('total_months', 'N/A')}
- 家庭数: {report_data.get('summary_statistics', {}).get('total_households', 'N/A')}
- 企业数: {report_data.get('summary_statistics', {}).get('total_firms', 'N/A')}

## 实验结果

实验已完成。详细数据请查看原始JSON报告。
"""
        return md
    
    def generate_comparison_report(self, experiment_names: List[str], metrics: List[str], format: str = "markdown") -> str:
        return f"对比报告：{len(experiment_names)} 个实验"
