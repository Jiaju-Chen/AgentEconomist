"""
结果分析工具

提供实验结果的对比和分析功能，基于 data/ 目录的 JSON 文件提取关键经济指标。
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import statistics

from ..core.manifest import load_manifest
from ..utils.path import to_absolute
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error


def _extract_key_metrics(data_dir: Path) -> Dict[str, Any]:
    """
    从 data 目录提取关键经济指标
    
    从以下文件提取数据：
    - economic_metrics_history.json: 月度经济指标历史
    - household_monthly_metrics.json: 家庭月度指标
    - firm_monthly_metrics.json: 企业月度指标
    - performance_metrics.json: 性能指标
    
    Returns:
        包含提取指标的字典
    """
    metrics = {}
    
    try:
        # 1. 读取经济指标历史
        econ_history_path = data_dir / "economic_metrics_history.json"
        if econ_history_path.exists():
            with open(econ_history_path, 'r') as f:
                econ_history = json.load(f)
            
            # 提取最终和平均指标
            if econ_history:
                final_month = econ_history[-1]
                
                # 就业数据
                emp = final_month.get("employment_statistics", {})
                metrics["employment"] = {
                    "final_rate": emp.get("household_employment_rate", 0),
                    "final_unemployment_rate": emp.get("household_unemployment_rate", 0),
                    "labor_utilization": emp.get("labor_utilization_rate", 0),
                    "total_households": emp.get("total_households", 0),
                    "employed_households": emp.get("employed_households", 0),
                }
                
                # 收入支出
                income_exp = final_month.get("income_expenditure_analysis", {})
                metrics["income_expenditure"] = {
                    "avg_monthly_income": income_exp.get("average_monthly_income", 0),
                    "avg_monthly_expenditure": income_exp.get("average_monthly_expenditure", 0),
                    "savings_rate": income_exp.get("monthly_savings_rate", 0),
                    "expenditure_income_ratio": income_exp.get("expenditure_income_ratio", 0),
                    "total_monthly_income": income_exp.get("total_monthly_income", 0),
                }
                
                # 不平等指标
                wealth_dist = final_month.get("wealth_distribution", {})
                metrics["inequality"] = {
                    "gini_coefficient": wealth_dist.get("gini_coefficient", 0),
                    "wealth_std": wealth_dist.get("wealth_std", 0),
                    "income_gini": wealth_dist.get("income_gini_coefficient", 0),
                }
                
                # 消费结构
                consumption = income_exp.get("monthly_consumption_structure", {})
                if consumption:
                    metrics["consumption_structure"] = consumption
                
                # 计算月度趋势（如果有多个月）
                if len(econ_history) > 1:
                    employment_rates = [m.get("employment_statistics", {}).get("household_employment_rate", 0) 
                                      for m in econ_history]
                    gini_coeffs = [m.get("wealth_distribution", {}).get("gini_coefficient", 0) 
                                  for m in econ_history]
                    incomes = [m.get("income_expenditure_analysis", {}).get("average_monthly_income", 0)
                              for m in econ_history]
                    
                    metrics["trends"] = {
                        "employment_trend": "increasing" if employment_rates[-1] > employment_rates[0] else "decreasing",
                        "gini_trend": "increasing" if gini_coeffs[-1] > gini_coeffs[0] else "decreasing",
                        "income_trend": "increasing" if incomes[-1] > incomes[0] else "decreasing",
                        "employment_change": employment_rates[-1] - employment_rates[0],
                        "gini_change": gini_coeffs[-1] - gini_coeffs[0],
                        "income_change": incomes[-1] - incomes[0],
                        "total_months": len(econ_history),
                    }
        
        # 2. 读取家庭月度数据
        household_path = data_dir / "household_monthly_metrics.json"
        if household_path.exists():
            with open(household_path, 'r') as f:
                household_data = json.load(f)
            
            # 计算最后一个月的家庭统计
            if household_data:
                # 鲁棒性检查：确保是字典格式
                if isinstance(household_data, dict):
                    last_month = max(household_data.keys())
                    households = household_data[last_month]
                elif isinstance(household_data, list):
                    # 处理旧格式或特殊格式：直接是家庭列表
                    households = household_data
                else:
                    print(f"Warning: Unexpected household_data type: {type(household_data)}")
                    households = []
                
                if households and isinstance(households, list):
                    savings_rates = [h.get("savings_rate", 0) for h in households if isinstance(h, dict)]
                    incomes = [h.get("monthly_income", 0) for h in households if isinstance(h, dict)]
                    expenditures = [h.get("monthly_expenditure", 0) for h in households if isinstance(h, dict)]
                    
                    metrics["household_stats"] = {
                        "avg_savings_rate": statistics.mean(savings_rates) if savings_rates else 0,
                        "median_income": statistics.median(incomes) if incomes else 0,
                        "income_std": statistics.stdev(incomes) if len(incomes) > 1 else 0,
                        "median_expenditure": statistics.median(expenditures) if expenditures else 0,
                        "total_households_sampled": len(households),
                    }
        
        # 3. 读取企业月度数据
        firm_path = data_dir / "firm_monthly_metrics.json"
        if firm_path.exists():
            with open(firm_path, 'r') as f:
                firm_data = json.load(f)
            
            if firm_data:
                # 鲁棒性检查：确保是字典格式
                if isinstance(firm_data, dict):
                    last_month = max(firm_data.keys())
                    firms = firm_data[last_month]
                elif isinstance(firm_data, list):
                    # 处理旧格式或特殊格式：直接是企业列表
                    firms = firm_data
                else:
                    print(f"Warning: Unexpected firm_data type: {type(firm_data)}")
                    firms = []
                
                if firms and isinstance(firms, list):
                    revenues = [f.get("monthly_revenue", 0) for f in firms if isinstance(f, dict)]
                    profits = [f.get("profit", 0) for f in firms if isinstance(f, dict)]
                    employees_list = [f.get("employees", 0) for f in firms if isinstance(f, dict)]
                    
                    metrics["firm_stats"] = {
                        "avg_revenue": statistics.mean(revenues) if revenues else 0,
                        "avg_profit": statistics.mean(profits) if profits else 0,
                        "median_revenue": statistics.median(revenues) if revenues else 0,
                        "total_firms": len(firms),
                        "avg_employees_per_firm": statistics.mean(employees_list) if employees_list else 0,
                    }
        
        # 4. 读取性能指标
        perf_path = data_dir / "performance_metrics.json"
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                perf_data = json.load(f)
            
            # 鲁棒性检查：处理 dict 或 list 格式
            if isinstance(perf_data, dict):
                metrics["performance"] = {
                    "total_time": perf_data.get("total_time", 0),
                    "avg_iteration_time": perf_data.get("avg_iteration_time", 0),
                }
            elif isinstance(perf_data, list) and len(perf_data) > 0:
                # 如果是 list，计算总时间和平均操作时间
                durations = [item.get("duration", 0) for item in perf_data if isinstance(item, dict) and "duration" in item]
                if durations:
                    metrics["performance"] = {
                        "total_time": sum(durations),
                        "avg_operation_time": statistics.mean(durations),
                        "total_operations": len(durations),
                        "max_operation_time": max(durations),
                        "min_operation_time": min(durations),
                    }
        
        # 5. 读取创新配置（如果存在）
        innovation_path = data_dir / "innovation_configs.json"
        if innovation_path.exists():
            with open(innovation_path, 'r') as f:
                innovation_data = json.load(f)
            
            # 鲁棒性检查：处理不同的数据格式
            if innovation_data:
                # 格式1：顶层有 simulation_config 键
                if "simulation_config" in innovation_data:
                    sim_config = innovation_data["simulation_config"]
                    metrics["innovation_config"] = {
                        "innovation_enabled": sim_config.get("innovation_enabled", False),
                        "total_innovation_events": innovation_data.get("total_innovation_events", 0),
                    }
                # 格式2：以公司ID为键的字典（统计创新配置的公司数量）
                elif isinstance(innovation_data, dict):
                    # 统计有创新配置的公司数量
                    total_firms_with_innovation = len(innovation_data)
                    metrics["innovation_config"] = {
                        "total_firms_with_innovation": total_firms_with_innovation,
                        "innovation_enabled": total_firms_with_innovation > 0,
                    }
    
    except Exception as e:
        print(f"Warning: Failed to extract some metrics: {e}")
    
    return metrics


def compare_experiments(
    manifest_path: str,
    group_names: Optional[str] = None
) -> str:
    """
    对比多个实验配置的结果（基于 data/ 目录的 JSON 文件）
    
    **数据来源**：从各实验的 output_dir/data/ 目录读取以下文件：
    - economic_metrics_history.json: 月度经济指标历史（就业、收入、不平等）
    - household_monthly_metrics.json: 家庭层面的收入、支出、储蓄率
    - firm_monthly_metrics.json: 企业层面的收入、利润、员工数
    - performance_metrics.json: 仿真性能指标
    - innovation_configs.json: 创新相关配置（如存在）
    
    **提取指标并计算组间差异**：
    - 就业率变化（百分比 & 绝对值）
    - 基尼系数变化（不平等程度）
    - 收入和储蓄率变化
    - 企业绩效对比（收入、利润）
    
    Args:
        manifest_path: manifest.yaml 路径
        group_names: 要对比的配置组（逗号分隔，如 "control,treatment"）
                    如果不提供，将对比所有配置组
    
    Returns:
        XML格式字符串，包含详细的经济指标对比、百分比变化和解释性标签
    """
    try:
        manifest = load_manifest(manifest_path)
        if not manifest:
            return format_tool_output("error", f"Manifest not found: {manifest_path}")
        
        # Parse group names
        if group_names:
            groups_list = [g.strip() for g in group_names.split(",")]
        else:
            groups_list = list(manifest.get("configurations", {}).keys())
        
        if len(groups_list) < 2:
            return format_tool_output("error", "Need at least 2 groups to compare")
        
        # Extract metrics for each group
        all_metrics = {}
        for group_name in groups_list:
            config = manifest.get("configurations", {}).get(group_name, {})
            output_dir = config.get("experiment_output_dir", "")
            
            if not output_dir:
                continue
            
            output_path = to_absolute(output_dir)
            data_dir = output_path / "data"
            
            if not data_dir.exists():
                continue
            
            all_metrics[group_name] = _extract_key_metrics(data_dir)
        
        if len(all_metrics) < 2:
            return format_tool_output(
                "error",
                "Could not extract metrics from enough groups. Make sure simulations have completed.",
                manifest_path=manifest_path
            )
        
        # Calculate differences (assume first group is baseline)
        baseline_name = groups_list[0]
        baseline = all_metrics[baseline_name]
        
        comparison_results = {
            "baseline_group": baseline_name,
            "groups_compared": groups_list,
            "comparisons": {}
        }
        
        for group_name in groups_list[1:]:
            if group_name not in all_metrics:
                continue
            
            treatment = all_metrics[group_name]
            diff = {}
            
            # Employment comparison
            if "employment" in baseline and "employment" in treatment:
                b_emp = baseline["employment"]["final_rate"]
                t_emp = treatment["employment"]["final_rate"]
                diff["employment_rate"] = {
                    "baseline": round(b_emp, 4),
                    "treatment": round(t_emp, 4),
                    "absolute_change": round(t_emp - b_emp, 4),
                    "percent_change": round((t_emp - b_emp) / b_emp * 100, 2) if b_emp > 0 else 0,
                    "interpretation": "improvement" if t_emp > b_emp else "decline"
                }
                
                b_unemp = baseline["employment"]["final_unemployment_rate"]
                t_unemp = treatment["employment"]["final_unemployment_rate"]
                diff["unemployment_rate"] = {
                    "baseline": round(b_unemp, 4),
                    "treatment": round(t_unemp, 4),
                    "absolute_change": round(t_unemp - b_unemp, 4),
                    "percent_change": round((t_unemp - b_unemp) / b_unemp * 100, 2) if b_unemp > 0 else 0,
                    "interpretation": "improvement" if t_unemp < b_unemp else "decline"
                }
            
            # Inequality comparison
            if "inequality" in baseline and "inequality" in treatment:
                b_gini = baseline["inequality"]["gini_coefficient"]
                t_gini = treatment["inequality"]["gini_coefficient"]
                diff["gini_coefficient"] = {
                    "baseline": round(b_gini, 4),
                    "treatment": round(t_gini, 4),
                    "absolute_change": round(t_gini - b_gini, 4),
                    "percent_change": round((t_gini - b_gini) / b_gini * 100, 2) if b_gini > 0 else 0,
                    "interpretation": "improvement (more equal)" if t_gini < b_gini else "decline (more unequal)"
                }
            
            # Income comparison
            if "income_expenditure" in baseline and "income_expenditure" in treatment:
                b_income = baseline["income_expenditure"]["avg_monthly_income"]
                t_income = treatment["income_expenditure"]["avg_monthly_income"]
                b_savings = baseline["income_expenditure"]["savings_rate"]
                t_savings = treatment["income_expenditure"]["savings_rate"]
                
                diff["avg_monthly_income"] = {
                    "baseline": round(b_income, 2),
                    "treatment": round(t_income, 2),
                    "absolute_change": round(t_income - b_income, 2),
                    "percent_change": round((t_income - b_income) / b_income * 100, 2) if b_income > 0 else 0,
                    "interpretation": "improvement" if t_income > b_income else "decline"
                }
                
                diff["savings_rate"] = {
                    "baseline": round(b_savings, 4),
                    "treatment": round(t_savings, 4),
                    "absolute_change": round(t_savings - b_savings, 4),
                    "interpretation": "higher savings" if t_savings > b_savings else "lower savings"
                }
            
            # Firm performance comparison
            if "firm_stats" in baseline and "firm_stats" in treatment:
                b_revenue = baseline["firm_stats"]["avg_revenue"]
                t_revenue = treatment["firm_stats"]["avg_revenue"]
                b_profit = baseline["firm_stats"]["avg_profit"]
                t_profit = treatment["firm_stats"]["avg_profit"]
                
                diff["firm_performance"] = {
                    "avg_revenue": {
                        "baseline": round(b_revenue, 2),
                        "treatment": round(t_revenue, 2),
                        "percent_change": round((t_revenue - b_revenue) / b_revenue * 100, 2) if b_revenue > 0 else 0
                    },
                    "avg_profit": {
                        "baseline": round(b_profit, 2),
                        "treatment": round(t_profit, 2),
                        "percent_change": round((t_profit - b_profit) / b_profit * 100, 2) if b_profit > 0 else 0
                    }
                }
            
            comparison_results["comparisons"][group_name] = diff
        
        # Format output
        comparison_text = json.dumps(comparison_results, indent=2, ensure_ascii=False)
        
        return format_tool_output(
            "success",
            f"Detailed comparison of {len(groups_list)} groups based on economic metrics from data/ directory",
            manifest_path=manifest_path,
            groups=", ".join(groups_list),
            comparison=comparison_text
        )
        
    except Exception as e:
        return handle_tool_error("compare_experiments", e)


def analyze_experiment_directory(
    manifest_path: str,
    group_name: str
) -> str:
    """
    分析单个实验目录的结果（提取关键经济指标）
    
    **数据来源**：从实验的 output_dir/data/ 目录读取以下 JSON 文件：
    - economic_metrics_history.json: 月度经济指标历史
    - household_monthly_metrics.json: 家庭层面数据
    - firm_monthly_metrics.json: 企业层面数据
    - performance_metrics.json: 性能指标
    - innovation_configs.json: 创新配置（可选）
    
    **提取的指标**：
    - 就业统计：就业率、失业率、劳动力利用率
    - 收入支出分析：平均收入、支出、储蓄率
    - 不平等指标：基尼系数、财富标准差
    - 消费结构：各类支出占比
    - 月度趋势：就业、收入、不平等的变化趋势
    - 家庭和企业统计：中位数收入、企业收入/利润
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称
    
    Returns:
        XML格式字符串，包含详细的经济指标分析和人类可读的摘要
    """
    try:
        manifest = load_manifest(manifest_path)
        if not manifest:
            return format_tool_output("error", f"Manifest not found: {manifest_path}")
        
        config = manifest.get("configurations", {}).get(group_name, {})
        if not config:
            return format_tool_output(
                "error",
                f"Configuration '{group_name}' not found",
                manifest_path=manifest_path
            )
        
        output_dir = config.get("experiment_output_dir", "")
        if not output_dir:
            return format_tool_output(
                "error",
                f"No output directory for group '{group_name}'",
                manifest_path=manifest_path
            )
        
        output_path = to_absolute(output_dir)
        data_dir = output_path / "data"
        
        # Check if data directory exists
        if not data_dir.exists():
            return format_tool_output(
                "error",
                f"Data directory not found: {data_dir}. Make sure simulation has completed.",
                manifest_path=manifest_path,
                group=group_name
            )
        
        # Extract key metrics
        metrics = _extract_key_metrics(data_dir)
        
        if not metrics:
            return format_tool_output(
                "error",
                f"Could not extract metrics from {data_dir}",
                manifest_path=manifest_path,
                group=group_name
            )
        
        # Format analysis results
        analysis_text = json.dumps(metrics, indent=2, ensure_ascii=False)
        
        # Generate summary text
        summary_parts = []
        
        if "employment" in metrics:
            emp = metrics["employment"]
            summary_parts.append(
                f"就业率: {emp.get('final_rate', 0):.2%}, "
                f"失业率: {emp.get('final_unemployment_rate', 0):.2%}"
            )
        
        if "inequality" in metrics:
            ineq = metrics["inequality"]
            summary_parts.append(
                f"基尼系数: {ineq.get('gini_coefficient', 0):.4f}"
            )
        
        if "income_expenditure" in metrics:
            inc_exp = metrics["income_expenditure"]
            summary_parts.append(
                f"平均月收入: {inc_exp.get('avg_monthly_income', 0):.2f}, "
                f"储蓄率: {inc_exp.get('savings_rate', 0):.2%}"
            )
        
        if "trends" in metrics:
            trends = metrics["trends"]
            summary_parts.append(
                f"趋势: 就业 {trends.get('employment_trend', 'N/A')}, "
                f"基尼 {trends.get('gini_trend', 'N/A')}"
            )
        
        summary = "; ".join(summary_parts)
        
        return format_tool_output(
            "success",
            f"Extracted key economic metrics for group '{group_name}' from data/ directory",
            manifest_path=manifest_path,
            group=group_name,
            summary=summary,
            detailed_metrics=analysis_text
        )
        
    except Exception as e:
        return handle_tool_error("analyze_experiment_directory", e)
