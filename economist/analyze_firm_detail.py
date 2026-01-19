"""分析特定企业在两种政策情况下每个月的产量和质量"""

from experiment_analyzer import ExperimentAnalyzer, AnalysisConfig
from pathlib import Path


def print_firm_monthly_details(firm_id: str, experiment1_dir: str, experiment2_dir: str, 
                               exp1_name: str = "Policy Enabled", 
                               exp2_name: str = "Policy Disabled"):
    """
    打印特定企业在两种政策情况下每个月的产量和质量
    
    Args:
        firm_id: 企业ID
        experiment1_dir: 第一个实验目录
        experiment2_dir: 第二个实验目录
        exp1_name: 第一个实验名称
        exp2_name: 第二个实验名称
    """
    # 创建分析器
    analyzer1 = ExperimentAnalyzer(experiment1_dir)
    analyzer2 = ExperimentAnalyzer(experiment2_dir)
    
    # 获取企业每个月的详细数据
    monthly_data1 = analyzer1.analyze_firm_monthly_products(firm_id)
    monthly_data2 = analyzer2.analyze_firm_monthly_products(firm_id)
    
    if not monthly_data1 and not monthly_data2:
        print(f"⚠️  未找到企业 {firm_id} 的数据")
        return
    
    # 获取企业名称
    firm_name = None
    if monthly_data1:
        firm_name = monthly_data1[list(monthly_data1.keys())[0]].get('firm_name', firm_id)
    elif monthly_data2:
        firm_name = monthly_data2[list(monthly_data2.keys())[0]].get('firm_name', firm_id)
    
    print("="*80)
    print(f"📊 企业月度产量和质量详情: {firm_name} ({firm_id})")
    print("="*80)
    
    # 获取所有月份
    all_months = sorted(set(list(monthly_data1.keys()) + list(monthly_data2.keys())))
    
    if not all_months:
        print("  无数据")
        return
    
    print(f"\n  {'月份':<6} {'政策':<15} {'食物产量':<12} {'食物质量':<12} {'非食物产量':<14} {'非食物质量':<14} {'总产量':<12}")
    print(f"  {'-'*80}")
    
    for month in all_months:
        data1 = monthly_data1.get(month)
        data2 = monthly_data2.get(month)
        
        if data1:
            print(f"  {month:<6} {exp1_name:<15} "
                  f"{data1['food_quantity']:>11.2f}  "
                  f"{data1['avg_food_quality']:>11.4f}  "
                  f"{data1['nonfood_quantity']:>13.2f}  "
                  f"{data1['avg_nonfood_quality']:>13.4f}  "
                  f"{data1['total_quantity']:>11.2f}")
        
        if data2:
            print(f"  {month:<6} {exp2_name:<15} "
                  f"{data2['food_quantity']:>11.2f}  "
                  f"{data2['avg_food_quality']:>11.4f}  "
                  f"{data2['nonfood_quantity']:>13.2f}  "
                  f"{data2['avg_nonfood_quality']:>13.4f}  "
                  f"{data2['total_quantity']:>11.2f}")
        
        if data1 and data2:
            # 计算差值
            food_qty_diff = data2['food_quantity'] - data1['food_quantity']
            food_quality_diff = data2['avg_food_quality'] - data1['avg_food_quality']
            nonfood_qty_diff = data2['nonfood_quantity'] - data1['nonfood_quantity']
            nonfood_quality_diff = data2['avg_nonfood_quality'] - data1['avg_nonfood_quality']
            total_diff = data2['total_quantity'] - data1['total_quantity']
            
            print(f"  {'差值':<6} {'':<15} "
                  f"{food_qty_diff:>11.2f}  "
                  f"{food_quality_diff:>11.4f}  "
                  f"{nonfood_qty_diff:>13.2f}  "
                  f"{nonfood_quality_diff:>13.4f}  "
                  f"{total_diff:>11.2f}")
            print(f"  {'-'*80}")
    
    # 汇总统计
    print(f"\n  📈 汇总统计:")
    if monthly_data1:
        total_food_qty1 = sum(d['food_quantity'] for d in monthly_data1.values())
        total_nonfood_qty1 = sum(d['nonfood_quantity'] for d in monthly_data1.values())
        avg_food_quality1 = sum(d['avg_food_quality'] for d in monthly_data1.values() if d['avg_food_quality'] > 0) / max(len([d for d in monthly_data1.values() if d['avg_food_quality'] > 0]), 1)
        avg_nonfood_quality1 = sum(d['avg_nonfood_quality'] for d in monthly_data1.values() if d['avg_nonfood_quality'] > 0) / max(len([d for d in monthly_data1.values() if d['avg_nonfood_quality'] > 0]), 1)
        
        print(f"    {exp1_name}:")
        print(f"      总食物产量: {total_food_qty1:,.2f}")
        print(f"      总非食物产量: {total_nonfood_qty1:,.2f}")
        print(f"      平均食物质量: {avg_food_quality1:.4f}")
        print(f"      平均非食物质量: {avg_nonfood_quality1:.4f}")
    
    if monthly_data2:
        total_food_qty2 = sum(d['food_quantity'] for d in monthly_data2.values())
        total_nonfood_qty2 = sum(d['nonfood_quantity'] for d in monthly_data2.values())
        avg_food_quality2 = sum(d['avg_food_quality'] for d in monthly_data2.values() if d['avg_food_quality'] > 0) / max(len([d for d in monthly_data2.values() if d['avg_food_quality'] > 0]), 1)
        avg_nonfood_quality2 = sum(d['avg_nonfood_quality'] for d in monthly_data2.values() if d['avg_nonfood_quality'] > 0) / max(len([d for d in monthly_data2.values() if d['avg_nonfood_quality'] > 0]), 1)
        
        print(f"    {exp2_name}:")
        print(f"      总食物产量: {total_food_qty2:,.2f}")
        print(f"      总非食物产量: {total_nonfood_qty2:,.2f}")
        print(f"      平均食物质量: {avg_food_quality2:.4f}")
        print(f"      平均非食物质量: {avg_nonfood_quality2:.4f}")


def main():
    # 示例：分析特定企业
    firm_id = "0CPRF0-E"#"05J0JH-E"#"066PR4-E"  # Inertia Dynamics Corp.
    
    experiment1_dir = "/root/project/agentsociety-ecosim/output/exp_100h_12m_20251121_221420"
    experiment2_dir = "/root/project/agentsociety-ecosim/output/suppressed"
    
    print_firm_monthly_details(
        firm_id=firm_id,
        experiment1_dir=experiment1_dir,
        experiment2_dir=experiment2_dir,
        exp1_name="Policy Enabled",
        exp2_name="Policy Disabled"
    )


if __name__ == "__main__":
    main()

