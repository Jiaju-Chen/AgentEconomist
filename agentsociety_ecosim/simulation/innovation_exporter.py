#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新数据导出模块
将创新相关的指标、参数、统计信息保存为文本文件
"""

import os
from typing import Dict, Any, List
from datetime import datetime


class InnovationDataExporter:
    """创新数据导出器"""

    def __init__(self, output_dir: str = "outputs/innovation_reports"):
        """
        初始化导出器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    async def export_monthly_innovation_report(
        self,
        economic_center,
        month: int,
        config: Any,
        production_stats: Dict[str, Any],
        firms: List[Any] = None  # 🆕 添加企业列表参数
    ):
        """
        导出月度创新报告

        Args:
            economic_center: 经济中心对象
            month: 月份
            config: 配置对象
            production_stats: 生产统计数据
            firms: 企业列表（用于获取企业ID）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/month_{month:02d}_innovation_report.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            # 1. 标题和时间
            f.write("="*80 + "\n")
            f.write(f"创新系统月度报告 - 第 {month} 月\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # 2. 系统配置参数
            f.write("【系统配置参数】\n")
            f.write("-"*80 + "\n")
            f.write(f"创新模块启用: {config.enable_innovation_module}\n")
            f.write(f"创新提升倍数 γ (gamma): {config.innovation_gamma}\n")
            f.write(f"创新强度 λ (lambda): {config.innovation_lambda}\n")
            f.write(f"研发边际递减 β (beta): {config.innovation_concavity_beta}\n")
            f.write(f"基础研发比例: {config.innovation_research_share:.1%}\n")
            f.write(f"政策鼓励创新: {config.policy_encourage_innovation}\n")
            f.write("\n")

            # 3. 生产统计中的创新数据
            if production_stats:
                f.write("【研发劳动力统计】\n")
                f.write("-"*80 + "\n")

                total_research_labor = production_stats.get('total_research_effective_labor', 0.0)
                f.write(f"总研发有效劳动力: {total_research_labor:.2f}\n")

                firm_research_labor = production_stats.get('firm_research_labor', {})
                if firm_research_labor:
                    f.write(f"\n各企业研发劳动力分配:\n")
                    for firm_id, research_eff in sorted(firm_research_labor.items(),
                                                       key=lambda x: x[1], reverse=True):
                        f.write(f"  {firm_id}: {research_eff:.2f}\n")
                f.write("\n")

                # 5. 创新到达率和到达次数
                f.write("【创新事件统计】\n")
                f.write("-"*80 + "\n")

                firm_innovation_arrival_rate = production_stats.get('firm_innovation_arrival_rate', {})
                firm_innovation_arrivals = production_stats.get('firm_innovation_arrivals', {})

                if firm_innovation_arrival_rate:
                    f.write(f"企业创新到达率 Λ_t = λ × (研发劳动力)^β:\n\n")
                    f.write(f"{'企业ID':<30} {'研发劳动力':<15} {'到达率Λ_t':<15} {'本月到达次数':<15}\n")
                    f.write("-"*80 + "\n")

                    for firm_id in sorted(firm_innovation_arrival_rate.keys()):
                        research_labor = firm_research_labor.get(firm_id, 0.0)
                        arrival_rate = firm_innovation_arrival_rate.get(firm_id, 0.0)
                        arrivals = firm_innovation_arrivals.get(firm_id, 0)

                        f.write(f"{firm_id:<30} {research_labor:<15.2f} {arrival_rate:<15.4f} {arrivals:<15}\n")

                    # 统计发生创新的企业
                    firms_with_innovation = [fid for fid, arr in firm_innovation_arrivals.items() if arr > 0]
                    f.write("\n")
                    f.write(f"本月发生创新的企业数: {len(firms_with_innovation)} 家\n")
                    if firms_with_innovation:
                        f.write(f"发生创新的企业: {', '.join(firms_with_innovation[:5])}")
                        if len(firms_with_innovation) > 5:
                            f.write(f" 等{len(firms_with_innovation)}家")
                        f.write("\n")
                f.write("\n")

            # 6. 企业创新策略分布
            f.write("【企业创新策略分布】\n")
            f.write("-"*80 + "\n")

            # 获取所有企业的创新策略
            all_strategies = await self._get_all_firm_strategies(economic_center, firms)

            encouraged_firms = [fid for fid, s in all_strategies.items() if s['strategy'] == 'encouraged']
            suppressed_firms = [fid for fid, s in all_strategies.items() if s['strategy'] == 'suppressed']

            f.write(f"鼓励创新企业 ({len(encouraged_firms)} 家):\n")
            for firm_id in encouraged_firms[:20]:  # 显示前20家
                strategy_info = all_strategies[firm_id]
                f.write(f"  {firm_id}: 研发比例 {strategy_info['research_share']:.1%}\n")
            if len(encouraged_firms) > 20:
                f.write(f"  ... 以及其他 {len(encouraged_firms)-20} 家\n")

            f.write(f"\n抑制创新企业 ({len(suppressed_firms)} 家):\n")
            for firm_id in suppressed_firms[:20]:
                f.write(f"  {firm_id}: 研发比例 0%\n")
            if len(suppressed_firms) > 20:
                f.write(f"  ... 以及其他 {len(suppressed_firms)-20} 家\n")

            f.write("\n")

            # 7. 创新事件历史（使用FirmInnovationEvent对象）
            innovation_events = await economic_center.query_all_firm_innovation_events.remote()
            month_events = [e for e in innovation_events if e.month == month]

            if month_events:
                f.write("【本月创新事件详情】\n")
                f.write("-"*80 + "\n")
                f.write(f"共 {len(month_events)} 个创新事件\n\n")

                for i, event in enumerate(month_events[:24], 1):  # 只显示前20个
                    f.write(f"事件 {i}:\n")
                    f.write(f"  企业: {event.company_id}\n")
                    f.write(f"  类型: {event.innovation_type or 'N/A'}\n")
                    if event.old_value is not None and event.new_value is not None:
                        f.write(f"  变化: {event.old_value:.2f} → {event.new_value:.2f}\n")
                    if event.price_change is not None:
                        f.write(f"  价格变化: {event.price_change:.4f}\n")
                    if event.attribute_change is not None:
                        f.write(f"  属性变化: {event.attribute_change:.4f}\n")
                    f.write("\n")

                if len(month_events) > 20:
                    f.write(f"... 以及其他 {len(month_events)-20} 个事件\n")
            else:
                f.write("【本月创新事件】\n")
                f.write("-"*80 + "\n")
                f.write("本月无创新事件记录\n")

            f.write("\n")

            # 8. 结尾
            f.write("="*80 + "\n")
            f.write("报告生成完毕\n")
            f.write("="*80 + "\n")

        print(f"✅ 创新月度报告已保存: {filename}")

    async def _get_all_firm_strategies(
        self, economic_center, firms: List[Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取所有企业的创新策略

        Args:
            economic_center: 经济中心对象
            firms: 企业列表

        Returns:
            Dict: {firm_id: {"strategy": str, "research_share": float}}
        """
        strategies = {}

        if firms:
            # 从企业列表获取所有firm_id，然后查询策略
            for firm in firms:
                firm_id = firm.company_id
                # 使用新的query_firm_innovation_config获取FirmInnovationConfig对象
                config = await economic_center.query_firm_innovation_config.remote(firm_id)
                strategies[firm_id] = {
                    "strategy": config.innovation_strategy,
                    "research_share": config.fund_share
                }

        return strategies

    async def export_summary_report(
        self,
        economic_center,
        total_months: int,
        config: Any
    ):
        """
        导出整体汇总报告（所有月份）

        Args:
            economic_center: 经济中心对象
            total_months: 总月份数
            config: 配置对象
        """
        filename = f"{self.output_dir}/innovation_summary_all_months.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            # 1. 标题
            f.write("="*80 + "\n")
            f.write(f"创新系统完整报告 (月份 1-{total_months})\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # 2. 配置参数
            f.write("【系统配置】\n")
            f.write("-"*80 + "\n")
            f.write(f"创新提升倍数 γ: {config.innovation_gamma}\n")
            f.write(f"创新强度 λ: {config.innovation_lambda}\n")
            f.write(f"研发边际递减 β: {config.innovation_concavity_beta}\n")
            f.write(f"基础研发比例: {config.innovation_research_share:.1%}\n")
            f.write("\n")

            # 3. 所有创新事件（使用FirmInnovationEvent对象）
            all_events = await economic_center.query_all_firm_innovation_events.remote()

            f.write("【创新事件总览】\n")
            f.write("-"*80 + "\n")
            f.write(f"创新事件总数: {len(all_events)}\n")

            # 按月份统计
            events_by_month = {}
            for event in all_events:
                m = event.month
                events_by_month[m] = events_by_month.get(m, 0) + 1

            f.write(f"\n各月事件分布:\n")
            for month in range(1, total_months + 1):
                count = events_by_month.get(month, 0)
                f.write(f"  第 {month} 月: {count} 个事件\n")

            f.write("\n")

            # 5. 结尾
            f.write("="*80 + "\n")
            f.write("汇总报告生成完毕\n")
            f.write("="*80 + "\n")

        print(f"✅ 创新汇总报告已保存: {filename}")


# 示例用法
if __name__ == "__main__":
    print("创新数据导出模块已加载")
