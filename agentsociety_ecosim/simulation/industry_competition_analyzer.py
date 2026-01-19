#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业竞争分析模块
每月分析12个行业中两家竞争企业的销售份额，输出饼状图和详细JSON报告
"""

import os
import json
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from agentsociety_ecosim.utils.log_utils import setup_global_logger

logger = setup_global_logger(name="industry_competition", log_dir="logs", level="INFO")


class IndustryCompetitionAnalyzer:
    """
    行业竞争分析器

    功能：
    1. 按行业（daily_cate）分组企业
    2. 计算每家企业的月度销售数据
    3. 生成饼状图展示市场份额
    4. 输出详细的JSON报告
    """

    def __init__(self, output_dir: str = "outputs/industry_competition", economic_center=None, use_timestamp: bool = True):
        """
        初始化分析器

        Args:
            output_dir: 输出目录基础路径
            economic_center: 经济中心对象 (用于查询创新策略)
            use_timestamp: 是否在输出目录中添加时间戳(默认True,避免覆盖)
        """
        # 如果启用时间戳,在目录名中添加时间戳
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"{output_dir}_{timestamp}"
        else:
            self.output_dir = output_dir

        self.industry_mapping = {}  # {industry_name: [firm_id1, firm_id2]}
        self.monthly_reports = []  # 所有月度报告的历史记录
        self.economic_center = economic_center  # 保存经济中心引用

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.output_dir}/json", exist_ok=True)

        logger.info(f"行业竞争分析器初始化完成，输出目录: {self.output_dir}")

    def register_industry_firms(self, firms: List[Any]):
        """
        注册行业-企业映射关系

        Args:
            firms: 企业列表
        """
        industry_firms = defaultdict(list)

        for firm in firms:
            # 获取企业的行业分类
            industry = firm.main_business
            industry_firms[industry].append({
                'firm_id': firm.company_id,
                'firm_name': firm.company_name,
                'firm': firm
            })

        # 只保留有2家企业的行业（竞争市场）
        self.industry_mapping = {
            industry: firms_list
            for industry, firms_list in industry_firms.items()
            if len(firms_list) == 2
        }

        logger.info(f"✅ 注册了 {len(self.industry_mapping)} 个行业的竞争关系")
        for industry, firms_list in self.industry_mapping.items():
            firm_ids = [f['firm_id'] for f in firms_list]
            logger.info(f"   📦 {industry}: {firm_ids}")


    async def analyze_monthly_competition(self, economic_center, month: int, production_stats: Dict[str, Any] = None):
        """
        分析指定月份的行业竞争情况

        Args:
            economic_center: 经济中心对象（Ray remote actor）
            month: 月份
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 开始分析第 {month} 月的行业竞争情况")
        logger.info(f"{'='*60}")

        # 收集销售数据（经济中心是Ray actor，需要用remote调用）
        sales_data = await economic_center.collect_sales_statistics.remote(month)

        # 可选：从生产统计获取劳动与创新数据
        firm_labor_efficiency = {}
        firm_research_labor = {}
        firm_innovation_arrival_rate = {}
        firm_innovation_arrivals = {}

        if isinstance(production_stats, dict):
            firm_labor_efficiency = production_stats.get('firm_labor_efficiency', {}) or {}
            firm_research_labor = production_stats.get('firm_research_labor', {}) or {}
            firm_innovation_arrival_rate = production_stats.get('firm_innovation_arrival_rate', {}) or {}
            firm_innovation_arrivals = production_stats.get('firm_innovation_arrivals', {}) or {}
        else:
            # 若调用方未传入，则尝试从经济中心查询最近缓存的本月生产统计
            try:
                ps = await economic_center.query_production_stats_by_month.remote(month)
                if isinstance(ps, dict):
                    firm_labor_efficiency = ps.get('firm_labor_efficiency', {}) or {}
                    firm_research_labor = ps.get('firm_research_labor', {}) or {}
                    firm_innovation_arrival_rate = ps.get('firm_innovation_arrival_rate', {}) or {}
                    firm_innovation_arrivals = ps.get('firm_innovation_arrivals', {}) or {}
            except Exception:
                pass

        # 按行业汇总数据
        industry_reports = {}

        for industry, firms_list in self.industry_mapping.items():
            firm1 = firms_list[0]
            firm2 = firms_list[1]

            # 计算每家企业的销售数据
            firm1_stats = self._calculate_firm_sales(firm1['firm_id'], sales_data)
            firm2_stats = self._calculate_firm_sales(firm2['firm_id'], sales_data)

            # 获取企业财务数据
            firm1_financials = await economic_center.query_firm_monthly_financials.remote(
                firm1['firm_id'], month
            )
            firm2_financials = await economic_center.query_firm_monthly_financials.remote(
                firm2['firm_id'], month
            )

            # 获取企业生产数据
            firm1_production = await economic_center.query_firm_production_stats.remote(
                firm1['firm_id'], month
            )
            firm2_production = await economic_center.query_firm_production_stats.remote(
                firm2['firm_id'], month
            )

            # 获取企业创新策略（从FirmInnovationConfig对象获取）
            firm1_innovation_config = await economic_center.query_firm_innovation_config.remote(firm1['firm_id'])
            firm2_innovation_config = await economic_center.query_firm_innovation_config.remote(firm2['firm_id'])

            # 计算市场份额（只考虑家庭购买）
            total_quantity = firm1_stats['household_quantity'] + firm2_stats['household_quantity']
            total_revenue = firm1_stats['household_revenue'] + firm2_stats['household_revenue']

            if total_quantity > 0:
                firm1_quantity_share = (firm1_stats['household_quantity'] / total_quantity) * 100
                firm2_quantity_share = (firm2_stats['household_quantity'] / total_quantity) * 100
            else:
                firm1_quantity_share = 0.0
                firm2_quantity_share = 0.0

            if total_revenue > 0:
                firm1_revenue_share = (firm1_stats['household_revenue'] / total_revenue) * 100
                firm2_revenue_share = (firm2_stats['household_revenue'] / total_revenue) * 100
            else:
                firm1_revenue_share = 0.0
                firm2_revenue_share = 0.0

            # 构建行业报告（只考虑家庭购买）
            industry_report = {
                "industry": industry,
                "month": month,
                "timestamp": datetime.now().isoformat(),
                "total_market_quantity": total_quantity,  # 只包含家庭购买数量
                "total_market_revenue": total_revenue,  # 只包含家庭购买收入
                "firms": [
                    {
                        "firm_id": firm1['firm_id'],
                        "firm_name": firm1['firm_name'],
                        "sales_quantity": firm1_stats['household_quantity'],  # 只显示家庭购买数量
                        "sales_revenue": firm1_stats['household_revenue'],  # 只显示家庭购买收入
                        "quantity_share_pct": firm1_quantity_share,
                        "revenue_share_pct": firm1_revenue_share,
                        "household_sales": firm1_stats['household_quantity'],
                        "inherent_market_sales": firm1_stats['inherent_market_quantity'],
                        "product_count": firm1_stats['product_count'],
                        "product_details": firm1_stats['product_details'],
                        "financials": firm1_financials,
                        "production": firm1_production,
                        "innovation_strategy": firm1_innovation_config.innovation_strategy,
                        "labor": (firm_labor_efficiency.get(firm1['firm_id'], {})) | {
                            # 附加研究有效劳动力（如可用）
                            "research_effective_labor": firm_research_labor.get(firm1['firm_id'], 0.0)
                        },
                        "innovation": {
                            "arrival_rate": firm_innovation_arrival_rate.get(firm1['firm_id'], 0.0),
                            "arrivals": firm_innovation_arrivals.get(firm1['firm_id'], 0)
                        }
                    },
                    {
                        "firm_id": firm2['firm_id'],
                        "firm_name": firm2['firm_name'],
                        "sales_quantity": firm2_stats['household_quantity'],  # 只显示家庭购买数量
                        "sales_revenue": firm2_stats['household_revenue'],  # 只显示家庭购买收入
                        "quantity_share_pct": firm2_quantity_share,
                        "revenue_share_pct": firm2_revenue_share,
                        "household_sales": firm2_stats['household_quantity'],
                        "inherent_market_sales": firm2_stats['inherent_market_quantity'],
                        "product_count": firm2_stats['product_count'],
                        "product_details": firm2_stats['product_details'],
                        "financials": firm2_financials,
                        "production": firm2_production,
                        "innovation_strategy": firm2_innovation_config.innovation_strategy,
                        "labor": (firm_labor_efficiency.get(firm2['firm_id'], {})) | {
                            "research_effective_labor": firm_research_labor.get(firm2['firm_id'], 0.0)
                        },
                        "innovation": {
                            "arrival_rate": firm_innovation_arrival_rate.get(firm2['firm_id'], 0.0),
                            "arrivals": firm_innovation_arrivals.get(firm2['firm_id'], 0)
                        }
                    }
                ]
            }

            industry_reports[industry] = industry_report

            # 输出控制台摘要（只考虑家庭购买）
            logger.info(f"\n🏭 【{industry}】（仅家庭购买）")
            logger.info(f"   家庭购买销量: {total_quantity:.1f} | 家庭购买收入: ${total_revenue:.2f}")
            logger.info(f"   {firm1['firm_id']}: 销量份额 {firm1_quantity_share:.1f}% | 收入份额 {firm1_revenue_share:.1f}%")
            logger.info(f"   {firm2['firm_id']}: 销量份额 {firm2_quantity_share:.1f}% | 收入份额 {firm2_revenue_share:.1f}%")

        # 保存JSON报告
        self._save_json_report(industry_reports, month)

        # 生成饼状图
        self._generate_pie_charts(industry_reports, month)

        # 保存到历史记录
        self.monthly_reports.append({
            "month": month,
            "reports": industry_reports
        })

        logger.info(f"\n✅ 第 {month} 月行业竞争分析完成")
        logger.info(f"{'='*60}\n")

    def _calculate_firm_sales(self, firm_id: str, sales_data: Dict) -> Dict[str, Any]:
        """
        计算单个企业的销售统计数据（只考虑家庭购买）

        Args:
            firm_id: 企业ID
            sales_data: 销售数据字典 {(product_id, seller_id): {...}}

        Returns:
            企业销售统计
        """
        total_quantity = 0.0
        total_revenue = 0.0
        household_quantity = 0.0
        household_revenue = 0.0  # 只计算家庭购买的收入
        inherent_market_quantity = 0.0
        product_details = []

        for (product_id, seller_id), stats in sales_data.items():
            if seller_id == firm_id:
                quantity = stats.get('quantity_sold', 0.0)
                revenue = stats.get('revenue', 0.0)
                hh_quantity = stats.get('household_quantity', 0.0)
                inherent_quantity = stats.get('inherent_market_quantity', 0.0)

                total_quantity += quantity
                total_revenue += revenue
                household_quantity += hh_quantity
                inherent_market_quantity += inherent_quantity

                # 计算家庭购买的收入：如果总销量>0，按比例分配收入
                if quantity > 0:
                    unit_price = revenue / quantity
                    hh_revenue = hh_quantity * unit_price
                    household_revenue += hh_revenue
                elif hh_quantity > 0:
                    # 如果只有家庭购买，收入全部算作家庭购买
                    household_revenue += revenue

                product_details.append({
                    "product_id": product_id,
                    "quantity_sold": quantity,
                    "revenue": revenue,
                    "household_quantity": hh_quantity,
                    "inherent_market_quantity": inherent_quantity,
                    "demand_level": stats.get('demand_level', 'normal')
                })

        return {
            "total_quantity": total_quantity,
            "total_revenue": total_revenue,
            "household_quantity": household_quantity,
            "household_revenue": household_revenue,  # 新增：家庭购买收入
            "inherent_market_quantity": inherent_market_quantity,
            "product_count": len(product_details),
            "product_details": product_details
        }

    def _save_json_report(self, industry_reports: Dict[str, Any], month: int):
        """
        保存JSON报告

        Args:
            industry_reports: 行业报告字典
            month: 月份
        """
        json_path = f"{self.output_dir}/json/month_{month:02d}_industry_competition.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(industry_reports, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 已保存JSON报告: {json_path}")

    def _generate_pie_charts(self, industry_reports: Dict[str, Any], month: int):
        """
        生成饼状图展示市场份额

        Args:
            industry_reports: 行业报告字典
            month: 月份
        """
        num_industries = len(industry_reports)
        if num_industries == 0:
            logger.warning("没有行业数据，跳过饼状图生成")
            return

        # 创建子图布局（3列布局）
        cols = 3
        rows = (num_industries + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        # 扁平化axes以便迭代
        axes_flat = [ax for row in axes for ax in row]

        for idx, (industry, report) in enumerate(industry_reports.items()):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            firm1 = report['firms'][0]
            firm2 = report['firms'][1]

            # 使用收入份额作为饼图数据
            sizes = [firm1['revenue_share_pct'], firm2['revenue_share_pct']]
            labels = [
                f"{firm1['firm_id']}\n{firm1['revenue_share_pct']:.1f}%",
                f"{firm2['firm_id']}\n{firm2['revenue_share_pct']:.1f}%"
            ]

            # 颜色方案：firm1 蓝色、firm2 绿色；抑制创新为红色
            firm1_strategy = firm1.get('innovation_strategy', 'suppressed')
            firm2_strategy = firm2.get('innovation_strategy', 'suppressed')
            colors = []
            # firm1 颜色
            if firm1_strategy == 'suppressed':
                colors.append('#D7191C')  # 红色
            else:
                colors.append('#1F77B4')  # 蓝色
            # firm2 颜色
            if firm2_strategy == 'suppressed':
                colors.append('#D7191C')  # 红色
            else:
                colors.append('#2CA02C')  # 绿色

            explode = (0.05, 0.05)  # 突出显示

            # 如果没有销售数据，显示空饼图
            if sum(sizes) == 0:
                sizes = [1, 1]
                labels = [f"{firm1['firm_id']}\nNo Sales", f"{firm2['firm_id']}\nNo Sales"]
                colors = ['#CCCCCC', '#AAAAAA']

            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1.5)
            )
            # 添加图例（右上角），颜色与切片一致
            try:
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor=colors[0], edgecolor='white', label=f"{firm1['firm_id']}") ,
                    Patch(facecolor=colors[1], edgecolor='white', label=f"{firm2['firm_id']}")
                ]
                ax.legend(handles=legend_handles, loc='upper right', frameon=True)
            except Exception:
                pass
            ax.set_title(f"{industry}\nHousehold Revenue: ${report['total_market_revenue']:.0f}",
                        fontsize=12, fontweight='bold')

        # 隐藏多余的子图
        for idx in range(num_industries, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle(f"Month {month} - Industry Competition Market Share Analysis (Household Purchases Only, by Revenue)\nDark Green/Light Green=Encouraged Innovation (Creative Destruction Theory) | Red=Suppressed Innovation",
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # 保存图片
        chart_path = f"{self.output_dir}/charts/month_{month:02d}_market_share.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 已生成饼状图: {chart_path}")

        # 额外生成单独的行业图表（方便查看细节）
        for industry, report in industry_reports.items():
            self._generate_single_industry_chart(industry, report, month)

    def _generate_single_industry_chart(self, industry: str, report: Dict[str, Any], month: int):
        """
        为单个行业生成详细图表

        Args:
            industry: 行业名称
            report: 行业报告
            month: 月份
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        firm1 = report['firms'][0]
        firm2 = report['firms'][1]

        # 颜色方案：firm1 蓝色、firm2 绿色；抑制创新为红色
        firm1_strategy = firm1.get('innovation_strategy', 'suppressed') if (firm := firm1) else 'suppressed'
        firm2_strategy = firm2.get('innovation_strategy', 'suppressed')
        colors = []
        # firm1 颜色
        if firm1_strategy == 'suppressed':
            colors.append('#D7191C')  # 红色
        else:
            colors.append('#1F77B4')  # 蓝色
        # firm2 颜色
        if firm2_strategy == 'suppressed':
            colors.append('#D7191C')  # 红色
        else:
            colors.append('#2CA02C')  # 绿色

        # 左图：收入份额饼图
        revenue_sizes = [firm1['revenue_share_pct'], firm2['revenue_share_pct']]
        revenue_labels = [firm1['firm_id'], firm2['firm_id']]

        if sum(revenue_sizes) > 0:
            ax1.pie(
                revenue_sizes,
                labels=revenue_labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90,
                explode=(0.05, 0.05),
                wedgeprops=dict(edgecolor='white', linewidth=1.5)
            )
            ax1.set_title(f"Revenue Share (Household Only)\nTotal: ${report['total_market_revenue']:.2f}", fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Sales Data', ha='center', va='center', fontsize=14)
            ax1.set_title("Revenue Share", fontweight='bold')

        # 右图：销量份额饼图
        quantity_sizes = [firm1['quantity_share_pct'], firm2['quantity_share_pct']]

        if sum(quantity_sizes) > 0:
            ax2.pie(
                quantity_sizes,
                labels=revenue_labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90,
                explode=(0.05, 0.05),
                wedgeprops=dict(edgecolor='white', linewidth=1.5)
            )
            ax2.set_title(f"Quantity Share (Household Only)\nTotal: {report['total_market_quantity']:.1f}", fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Sales Data', ha='center', va='center', fontsize=14)
            ax2.set_title("Quantity Share", fontweight='bold')
        # 为单个行业图添加整图图例（右上角）
        try:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=colors[0], edgecolor='white', label=f"{firm1['firm_id']}") ,
                Patch(facecolor=colors[1], edgecolor='white', label=f"{firm2['firm_id']}")
            ]
            fig.legend(handles=legend_handles, loc='upper right', frameon=True)
        except Exception:
            pass

        plt.suptitle(f"{industry} - Month {month} Competition Analysis\nDark Green/Light Green=Encouraged Innovation (Creative Destruction Theory) | Red=Suppressed Innovation",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图片（使用安全的文件名）
        safe_industry_name = industry.replace('/', '_').replace(' ', '_')
        chart_path = f"{self.output_dir}/charts/month_{month:02d}_{safe_industry_name}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self):
        """
        生成整体汇总报告（所有月份）
        """
        if not self.monthly_reports:
            logger.warning("没有月度数据，无法生成汇总报告")
            return

        summary_path = f"{self.output_dir}/json/summary_all_months.json"

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.monthly_reports, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 已生成汇总报告: {summary_path}")

        # 注意：_generate_trend_charts 现在是异步方法，需要单独调用
        logger.info("提示：请使用 await generate_trend_charts_async() 生成趋势图")

    async def generate_trend_charts_async(self, economic_center=None):
        """
        生成市场份额趋势图（跨月份）- 异步版本

        Args:
            economic_center: 经济中心对象（用于查询创新事件）
        """
        if len(self.monthly_reports) < 2:
            logger.info("月份数据不足，跳过趋势图生成")
            return

        # 获取所有创新事件
        innovation_events = []
        if economic_center:
            try:
                innovation_events = await economic_center.query_all_firm_innovation_events.remote()
            except Exception as e:
                logger.warning(f"无法获取创新事件数据: {e}")
                innovation_events = []

        # 为每个行业生成趋势图
        industries = list(self.industry_mapping.keys())

        for industry in industries:
            months = []
            firm1_shares = []
            firm2_shares = []
            firm1_id = None
            firm2_id = None
            firm1_color = None
            firm2_color = None

            for monthly_data in self.monthly_reports:
                month = monthly_data['month']
                reports = monthly_data['reports']

                if industry in reports:
                    report = reports[industry]
                    firm1 = report['firms'][0]
                    firm2 = report['firms'][1]

                    if firm1_id is None:
                        firm1_id = firm1['firm_id']
                        firm2_id = firm2['firm_id']
                        # 颜色方案：firm1 蓝色、firm2 绿色；抑制创新为红色
                        firm1_strategy = firm1.get('innovation_strategy', 'suppressed')
                        firm2_strategy = firm2.get('innovation_strategy', 'suppressed')

                        firm1_color = '#1F77B4' if firm1_strategy != 'suppressed' else '#D7191C'
                        firm2_color = '#2CA02C' if firm2_strategy != 'suppressed' else '#D7191C'

                    months.append(month)
                    firm1_shares.append(firm1['revenue_share_pct'])
                    firm2_shares.append(firm2['revenue_share_pct'])

            if len(months) < 2:
                continue

            # 绘制趋势图
            plt.figure(figsize=(12, 7))

            # 绘制市场份额曲线
            plt.plot(months, firm1_shares, marker='o', label=f"{firm1_id}",
                    linewidth=2, color=firm1_color, markersize=8)
            plt.plot(months, firm2_shares, marker='s', label=f"{firm2_id}",
                    linewidth=2, color=firm2_color, markersize=8)

            # 标注创新事件
            if innovation_events:
                # 筛选出该行业两家企业的创新事件
                firm_ids = [firm1_id, firm2_id]
                
                # 先按(company_id, month)聚合，统计每个月份每个公司的创新事件数量
                # 使用set去重，因为同一个创新到达可能产生多个事件（如price+labor_productivity_factor）
                # 但我们只关心实际创新到达的次数
                innovation_counts = {}  # {(company_id, month): count}
                
                for event in innovation_events:
                    event_company_id = event.company_id
                    event_month = event.month
                    innovation_type = event.innovation_type
                    
                    # 只统计有创新类型的事件，且属于该行业的企业
                    if event_company_id in firm_ids and innovation_type and event_month in months:
                        key = (event_company_id, event_month)
                        # 每个有效的创新事件计数为1
                        # 注意：由于同一个innovation_arrivals可能产生多个事件（如price和labor_productivity_factor），
                        # 我们通过统计不同的innovation_type来更准确地反映创新次数
                        if key not in innovation_counts:
                            innovation_counts[key] = 0
                        innovation_counts[key] += 1
                
                # 遍历聚合后的结果，每个(company_id, month)只标注一次
                for (event_company_id, event_month), count in innovation_counts.items():
                    try:
                        month_idx = months.index(event_month)

                        # 根据企业ID确定y坐标
                        if event_company_id == firm1_id:
                            y_pos = firm1_shares[month_idx]
                            color = firm1_color
                        else:
                            y_pos = firm2_shares[month_idx]
                            color = firm2_color

                        # 添加垂直标注线
                        plt.axvline(x=event_month, color=color, linestyle='--',
                                  alpha=0.5, linewidth=1.5)

                        # 添加标注文本
                        plt.annotate(f'Innovation×{count}',
                                   xy=(event_month, y_pos),
                                   xytext=(0, 15), textcoords='offset points',
                                   ha='center', fontsize=9,
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor=color, alpha=0.3, edgecolor=color),
                                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                    except (ValueError, IndexError):
                        continue

            plt.xlabel('Month', fontsize=12, fontweight='bold')
            plt.ylabel('Market Share (%)', fontsize=12, fontweight='bold')
            plt.title(f"{industry} - Market Share Trend (Household Purchases Only)\nBlue / Green=Encouraged Innovation (Creative Destruction Theory) | Red=Suppressed Innovation",
                     fontsize=14, fontweight='bold')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3, linestyle=':')
            plt.ylim(0, 100)

            # 设置x轴为整数月份
            if months:
                plt.xticks(months)

            # 保存图片
            safe_industry_name = industry.replace('/', '_').replace(' ', '_')
            trend_path = f"{self.output_dir}/charts/trend_{safe_industry_name}.png"
            plt.savefig(trend_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 已生成趋势图: {trend_path}")

    def _generate_trend_charts(self):
        """
        生成市场份额趋势图（跨月份）- 保留同步版本用于向后兼容
        """
        if len(self.monthly_reports) < 2:
            logger.info("月份数据不足，跳过趋势图生成")
            return

        # 为每个行业生成趋势图
        industries = list(self.industry_mapping.keys())

        for industry in industries:
            months = []
            firm1_shares = []
            firm2_shares = []
            firm1_id = None
            firm2_id = None
            firm1_color = None
            firm2_color = None

            for monthly_data in self.monthly_reports:
                month = monthly_data['month']
                reports = monthly_data['reports']

                if industry in reports:
                    report = reports[industry]
                    firm1 = report['firms'][0]
                    firm2 = report['firms'][1]

                    if firm1_id is None:
                        firm1_id = firm1['firm_id']
                        firm2_id = firm2['firm_id']
                        # 颜色方案：firm1 蓝色、firm2 绿色；抑制创新为红色
                        firm1_strategy = firm1.get('innovation_strategy', 'suppressed')
                        firm2_strategy = firm2.get('innovation_strategy', 'suppressed')

                        firm1_color = '#1F77B4' if firm1_strategy != 'suppressed' else '#D7191C'
                        firm2_color = '#2CA02C' if firm2_strategy != 'suppressed' else '#D7191C'

                    months.append(month)
                    firm1_shares.append(firm1['revenue_share_pct'])
                    firm2_shares.append(firm2['revenue_share_pct'])

            if len(months) < 2:
                continue

            # 绘制趋势图
            plt.figure(figsize=(10, 6))
            plt.plot(months, firm1_shares, marker='o', label=firm1_id, linewidth=2, color=firm1_color)
            plt.plot(months, firm2_shares, marker='s', label=firm2_id, linewidth=2, color=firm2_color)

            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Market Share (%)', fontsize=12)
            plt.title(f"{industry} - Market Share Trend (Household Purchases Only)\nGreen=Encouraged Innovation | Red=Suppressed Innovation",
                     fontsize=14, fontweight='bold')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)

            # 保存图片
            safe_industry_name = industry.replace('/', '_').replace(' ', '_')
            trend_path = f"{self.output_dir}/charts/trend_{safe_industry_name}.png"
            plt.savefig(trend_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 已生成趋势图: {trend_path}")


# 示例用法（供参考）
if __name__ == "__main__":
    analyzer = IndustryCompetitionAnalyzer()
    print("行业竞争分析器已初始化")
