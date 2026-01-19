"""Monthly Statistics Visualization Module"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import numpy as np
from typing import Dict, List, Any
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class MonthlyVisualization:
    """Monthly Statistics Visualization Class"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.charts_dir = f"output/{experiment_name}/charts/"
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def plot_unemployment_trend(self, monthly_unemployment_stats: Dict[int, Dict]):
        """1. Plot unemployment trend"""
        try:
            months = sorted(monthly_unemployment_stats.keys())
            unemployment_rates = [monthly_unemployment_stats[m]['unemployment_rate'] * 100 for m in months]
            unemployed_counts = [monthly_unemployment_stats[m]['total_unemployed'] for m in months]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Unemployment rate trend
            ax1.plot(months, unemployment_rates, marker='o', linewidth=2, color='#e74c3c')
            ax1.fill_between(months, unemployment_rates, alpha=0.3, color='#e74c3c')
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Unemployment Rate (%)', fontsize=12)
            ax1.set_title('Monthly Unemployment Rate Trend', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)
            
            # Unemployed count trend
            ax2.bar(months, unemployed_counts, color='#3498db', alpha=0.7)
            ax2.set_xlabel('Month', fontsize=12)
            ax2.set_ylabel('Number of Unemployed', fontsize=12)
            ax2.set_title('Monthly Unemployed Count', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}unemployment_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Unemployment trend chart saved")
            
        except Exception as e:
            print(f"❌ Unemployment trend chart failed: {e}")
    
    def plot_firm_revenue_distribution(self, monthly_firm_revenue: Dict[int, Dict]):
        """2. Plot firm revenue distribution"""
        try:
            # Use last month data
            if not monthly_firm_revenue:
                print("⚠️ No firm revenue data")
                return
            
            last_month = max(monthly_firm_revenue.keys())
            revenues = monthly_firm_revenue[last_month]
            
            if not revenues:
                print("⚠️ No firm revenue data for last month")
                return
            
            revenue_values = [v['revenue'] for v in revenues.values() if v['revenue'] > 0]
            profit_values = [v['profit'] for v in revenues.values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Revenue distribution histogram
            ax1.hist(revenue_values, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Revenue ($)', fontsize=12)
            ax1.set_ylabel('Number of Firms', fontsize=12)
            ax1.set_title(f'Firm Revenue Distribution (Month {last_month})', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Profit distribution histogram
            ax2.hist(profit_values, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Profit ($)', fontsize=12)
            ax2.set_ylabel('Number of Firms', fontsize=12)
            ax2.set_title(f'Firm Profit Distribution (Month {last_month})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even Line')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}firm_revenue_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Firm revenue distribution chart saved")
            
        except Exception as e:
            print(f"❌ Firm revenue distribution chart failed: {e}")
    
    def plot_product_purchase_rate(self, monthly_product_sales: Dict[int, Dict], 
                                   monthly_product_inventory: Dict[int, Dict]):
        """2. Plot product purchase rate (sales rate)"""
        try:
            if not monthly_product_sales or not monthly_product_inventory:
                print("⚠️ No product sales or inventory data")
                return
            
            last_month = max(monthly_product_sales.keys())
            sales = monthly_product_sales.get(last_month, {})
            inventory = monthly_product_inventory.get(last_month, {})
            
            # Calculate purchase rate
            purchase_rates = []
            for product_id, sale_data in sales.items():
                inv_data = inventory.get(product_id, {})
                if inv_data and inv_data.get('quantity', 0) > 0:
                    sold = sale_data.get('total_quantity', 0)
                    total = sold + inv_data.get('quantity', 0)
                    rate = (sold / total) * 100 if total > 0 else 0
                    purchase_rates.append(rate)
            
            if not purchase_rates:
                print("⚠️ Cannot calculate purchase rate")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(purchase_rates, bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Purchase Rate (%)', fontsize=12)
            ax.set_ylabel('Number of Products', fontsize=12)
            ax.set_title(f'Product Purchase Rate Distribution (Month {last_month})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axvline(x=np.mean(purchase_rates), color='red', linestyle='--', 
                      linewidth=2, label=f'Average Rate: {np.mean(purchase_rates):.1f}%')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}product_purchase_rate.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Product purchase rate chart saved")
            
        except Exception as e:
            print(f"❌ Product purchase rate chart failed: {e}")
    
    def plot_product_inventory_trend(self, monthly_product_inventory: Dict[int, Dict]):
        """3. Plot product inventory trend"""
        try:
            months = sorted(monthly_product_inventory.keys())
            
            # Calculate total inventory per month
            total_inventory_by_month = []
            for month in months:
                total = sum(p['quantity'] for p in monthly_product_inventory[month].values())
                total_inventory_by_month.append(total)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(months, total_inventory_by_month, marker='o', linewidth=2, color='#16a085')
            ax.fill_between(months, total_inventory_by_month, alpha=0.3, color='#16a085')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Total Inventory Quantity', fontsize=12)
            ax.set_title('Monthly Total Inventory Trend', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}product_inventory_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Product inventory trend chart saved")
            
        except Exception as e:
            print(f"❌ Product inventory trend chart failed: {e}")
    
    def plot_product_price_trend(self, monthly_product_prices: Dict[int, Dict]):
        """4. Plot product price trend"""
        try:
            months = sorted(monthly_product_prices.keys())
            
            # Calculate average price per month
            avg_prices = []
            for month in months:
                prices = [p['price'] for p in monthly_product_prices[month].values() if p['price'] > 0]
                avg_price = np.mean(prices) if prices else 0
                avg_prices.append(avg_price)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(months, avg_prices, marker='o', linewidth=2, color='#e67e22')
            ax.fill_between(months, avg_prices, alpha=0.3, color='#e67e22')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Average Price ($)', fontsize=12)
            ax.set_title('Monthly Average Product Price Trend', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}product_price_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Product price trend chart saved")
            
        except Exception as e:
            print(f"❌ Product price trend chart failed: {e}")
    
    def plot_purchase_quantity_distribution(self, monthly_product_sales: Dict[int, Dict], monthly_product_inventory: Dict[int, Dict] = None):
        """5. Plot top 10 products sales and restock trend over time"""
        try:
            months = sorted(monthly_product_sales.keys())
            
            # 统计所有商品的总销量，找出Top 10
            product_total_sales = {}
            for month in months:
                for product_id, data in monthly_product_sales[month].items():
                    if product_id not in product_total_sales:
                        product_total_sales[product_id] = {
                            'name': data.get('name', 'Unknown'),
                            'total_quantity': 0
                        }
                    product_total_sales[product_id]['total_quantity'] += data.get('total_quantity', 0)
            
            # 按总销量排序，取Top 10
            top_products = sorted(product_total_sales.items(), 
                                key=lambda x: x[1]['total_quantity'], 
                                reverse=True)[:10]
            
            if not top_products:
                print("⚠️ No product sales data")
                return
            
            # 为每个Top 10商品收集月度销量和补货量
            product_data = {}
            for product_id, product_info in top_products:
                monthly_sales = []
                monthly_restock = []
                
                for i, month in enumerate(months):
                    # 收集销量
                    sales = 0
                    if product_id in monthly_product_sales[month]:
                        sales = monthly_product_sales[month][product_id].get('total_quantity', 0)
                    monthly_sales.append(sales)
                    
                    # 计算补货量（如果有库存数据）
                    if monthly_product_inventory:
                        end_inventory = 0
                        if product_id in monthly_product_inventory.get(month, {}):
                            end_inventory = monthly_product_inventory[month][product_id].get('quantity', 0)
                        
                        start_inventory = 0
                        if i > 0:
                            prev_month = months[i-1]
                            if product_id in monthly_product_inventory.get(prev_month, {}):
                                start_inventory = monthly_product_inventory[prev_month][product_id].get('quantity', 0)
                        
                        restock = sales + (end_inventory - start_inventory)
                        monthly_restock.append(max(0, restock))
                    else:
                        monthly_restock.append(0)
                
                product_data[product_id] = {
                    'name': product_info['name'],
                    'sales': monthly_sales,
                    'restock': monthly_restock
                }
            
            # 创建图表 - 2行5列展示Top 10商品
            fig, axes = plt.subplots(2, 5, figsize=(24, 10))
            axes = axes.flatten()
            
            for idx, (product_id, data) in enumerate(product_data.items()):
                ax = axes[idx]
                product_name = data['name']
                sales = data['sales']
                restock = data['restock']
                
                # 🔧 只保留ASCII字符，避免乱码
                try:
                    ascii_name = ''.join(c for c in product_name if ord(c) < 128)
                    if len(ascii_name.strip()) > 10:
                        product_name = ascii_name.strip()[:50]
                    else:
                        product_name = f"Product {product_id[:8]}"
                except:
                    product_name = f"Product {idx+1}"
                
                if len(product_name) > 50:
                    product_name = product_name[:47] + '...'
                
                # 绘制销量趋势（蓝色实线）
                ax.plot(months, sales, marker='o', linewidth=2.5, 
                       markersize=6, color='#2E86DE', alpha=0.9, label='Sales', zorder=3)
                ax.fill_between(months, sales, alpha=0.2, color='#2E86DE')
                
                # 绘制补货趋势（橙色虚线）
                if monthly_product_inventory and any(r > 0 for r in restock):
                    ax.plot(months, restock, marker='s', linewidth=2, 
                           markersize=5, color='#EE5A24', alpha=0.8, 
                           linestyle='--', label='Restock', zorder=2)
                
                ax.set_xlabel('Month', fontsize=10)
                ax.set_ylabel('Quantity', fontsize=10)
                ax.set_title(f'#{idx+1}: {product_name}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='upper left')
                
                # 标注总销量和平均补货
                total_sales = sum(sales)
                avg_restock = np.mean(restock) if any(r > 0 for r in restock) else 0
                info_text = f'Sales: {total_sales:.0f}'
                if avg_restock > 0:
                    info_text += f'\nAvg Restock: {avg_restock:.1f}'
                
                ax.text(0.95, 0.95, info_text, 
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.suptitle('Top 10 Best-Selling Products - Sales vs Restock', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"{self.charts_dir}top_products_sales_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Top products sales and restock trend chart saved")
            
        except Exception as e:
            print(f"❌ Top products sales trend chart failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_supply_demand_curve(self, monthly_supply_demand: Dict[int, Dict]):
        """6. Plot supply-demand curve"""
        try:
            if not monthly_supply_demand:
                print("⚠️ No supply-demand data")
                return
            
            last_month = max(monthly_supply_demand.keys())
            supply_demand = monthly_supply_demand[last_month]
            
            # 同时过滤supply和demand都大于0的数据点，确保x和y长度相同
            supplies = []
            demands = []
            for sd in supply_demand.values():
                if sd['supply'] > 0 and sd['demand'] > 0:
                    supplies.append(sd['supply'])
                    demands.append(sd['demand'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Supply vs Demand scatter plot
            if supplies and demands:
                ax1.scatter(supplies, demands, alpha=0.6, s=50, c='#3498db')
                max_val = max(max(supplies), max(demands))
                ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equilibrium Line')
                ax1.set_xlabel('Supply Quantity', fontsize=12)
                ax1.set_ylabel('Demand Quantity', fontsize=12)
                ax1.set_title(f'Supply-Demand Relationship (Month {last_month})', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No valid supply-demand data', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_xlabel('Supply Quantity', fontsize=12)
                ax1.set_ylabel('Demand Quantity', fontsize=12)
                ax1.set_title(f'Supply-Demand Relationship (Month {last_month})', fontsize=14, fontweight='bold')
            
            # Supply-Demand ratio distribution
            ratios = [sd['supply_demand_ratio'] for sd in supply_demand.values() 
                     if sd['supply_demand_ratio'] != float('inf') and sd['supply_demand_ratio'] < 10]
            if ratios:
                ax2.hist(ratios, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
                ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Equilibrium Point')
                ax2.set_xlabel('Supply-Demand Ratio (Supply/Demand)', fontsize=12)
                ax2.set_ylabel('Number of Products', fontsize=12)
                ax2.set_title(f'Supply-Demand Ratio Distribution (Month {last_month})', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}supply_demand_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Supply-demand curve chart saved")
            
        except Exception as e:
            print(f"❌ Supply-demand curve chart failed: {e}")
    
    def plot_product_sales_ranking(self, monthly_product_sales: Dict[int, Dict]):
        """8. Plot product sales ranking (Long-tail distribution)"""
        try:
            # 使用最后一个月的数据
            if not monthly_product_sales:
                print("⚠️ No product sales data")
                return
            
            last_month = max(monthly_product_sales.keys())
            sales_data = monthly_product_sales[last_month]
            
            # 提取所有商品的销量
            product_sales = []
            for product_id, data in sales_data.items():
                product_sales.append({
                    'name': data.get('name', 'Unknown'),
                    'quantity': data.get('total_quantity', 0)
                })
            
            # 按销量从高到低排序
            product_sales.sort(key=lambda x: x['quantity'], reverse=True)
            
            if not product_sales:
                print("⚠️ No product sales to plot")
                return
            
            # 提取数据
            quantities = [p['quantity'] for p in product_sales]
            ranks = list(range(1, len(quantities) + 1))
            
            # 计算累积销量占比（帕累托分析）
            total_quantity = sum(quantities)
            cumulative_pct = np.cumsum(quantities) / total_quantity * 100
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 子图1: 销量排名曲线（长尾分布）
            ax1.plot(ranks, quantities, linewidth=2, color='#3498db', alpha=0.8)
            ax1.fill_between(ranks, quantities, alpha=0.3, color='#3498db')
            ax1.set_xlabel('Product Rank', fontsize=12)
            ax1.set_ylabel('Sales Quantity', fontsize=12)
            ax1.set_title(f'Product Sales Ranking - Long Tail Distribution (Month {last_month})', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # 对数刻度更好地展示长尾
            
            # 标注Top产品
            top_n = min(5, len(product_sales))
            for i in range(top_n):
                ax1.annotate(f'#{i+1}\n{quantities[i]:.0f}', 
                           xy=(i+1, quantities[i]),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=9, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            
            # 子图2: 帕累托图（累积占比）
            ax2_bar = ax2.bar(ranks, quantities, color='#3498db', alpha=0.6, width=1.0)
            ax2.set_xlabel('Product Rank', fontsize=12)
            ax2.set_ylabel('Sales Quantity', fontsize=12, color='#3498db')
            ax2.tick_params(axis='y', labelcolor='#3498db')
            
            # 添加累积占比曲线
            ax2_line = ax2.twinx()
            ax2_line.plot(ranks, cumulative_pct, color='#e74c3c', linewidth=3, marker='o', 
                         markersize=2, label='Cumulative %')
            ax2_line.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                           label='80% Line')
            ax2_line.set_ylabel('Cumulative Sales %', fontsize=12, color='#e74c3c')
            ax2_line.tick_params(axis='y', labelcolor='#e74c3c')
            ax2_line.set_ylim(0, 105)
            ax2_line.legend(loc='lower right')
            
            # 找到80%分界点
            idx_80 = np.argmax(cumulative_pct >= 80)
            if idx_80 > 0:
                pct_80 = (idx_80 + 1) / len(product_sales) * 100
                ax2.axvline(x=idx_80+1, color='red', linestyle='--', linewidth=2, alpha=0.5)
                ax2.text(idx_80+1, ax2.get_ylim()[1]*0.5, 
                        f'Top {idx_80+1} products\n({pct_80:.1f}%)\n= 80% sales',
                        fontsize=10, ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            
            ax2.set_title(f'Pareto Analysis - 80/20 Rule (Month {last_month})', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}product_sales_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Product sales ranking chart saved")
            print(f"   📊 Total products: {len(product_sales)}")
            print(f"   📊 Top {idx_80+1} products ({pct_80:.1f}%) account for 80% of sales")
            
        except Exception as e:
            print(f"❌ Product sales ranking chart failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_product_sales_ranking_multi_months(self, monthly_product_sales: Dict[int, Dict], target_months: List[int] = [1, 4, 7, 10]):
        """8b. Plot product sales ranking for multiple months (Pareto comparison)"""
        try:
            if not monthly_product_sales:
                print("⚠️ No product sales data")
                return
            
            # 筛选可用的目标月份
            available_months = sorted(monthly_product_sales.keys())
            plot_months = [m for m in target_months if m in available_months]
            
            if not plot_months:
                print(f"⚠️ None of target months {target_months} available in data")
                return
            
            # 创建2x2的子图布局
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            for idx, month in enumerate(plot_months[:4]):  # 最多画4个月
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                sales_data = monthly_product_sales[month]
                
                # 提取所有商品的销量
                product_sales = []
                for product_id, data in sales_data.items():
                    product_sales.append({
                        'name': data.get('name', 'Unknown'),
                        'quantity': data.get('total_quantity', 0)
                    })
                
                # 按销量从高到低排序
                product_sales.sort(key=lambda x: x['quantity'], reverse=True)
                
                if not product_sales:
                    ax.text(0.5, 0.5, f'No data for Month {month}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    continue
                
                # 提取数据
                quantities = [p['quantity'] for p in product_sales]
                ranks = list(range(1, len(quantities) + 1))
                
                # 计算累积销量占比（帕累托分析）
                total_quantity = sum(quantities)
                cumulative_pct = np.cumsum(quantities) / total_quantity * 100
                
                # 绘制帕累托图（柱状图 + 累积曲线）
                ax_bar = ax
                ax_bar.bar(ranks, quantities, color=colors[idx], alpha=0.6, width=1.0)
                ax_bar.set_xlabel('Product Rank', fontsize=11)
                ax_bar.set_ylabel('Sales Quantity', fontsize=11, color=colors[idx])
                ax_bar.tick_params(axis='y', labelcolor=colors[idx])
                
                # 添加累积占比曲线（双Y轴）
                ax_line = ax_bar.twinx()
                ax_line.plot(ranks, cumulative_pct, color='darkred', linewidth=2.5, 
                           marker='o', markersize=3, label='Cumulative %')
                ax_line.axhline(y=80, color='red', linestyle='--', linewidth=2, 
                              alpha=0.6, label='80% Line')
                ax_line.set_ylabel('Cumulative Sales %', fontsize=11, color='darkred')
                ax_line.tick_params(axis='y', labelcolor='darkred')
                ax_line.set_ylim(0, 105)
                
                # 找到80%分界点
                idx_80 = np.argmax(cumulative_pct >= 80)
                if idx_80 > 0:
                    pct_80 = (idx_80 + 1) / len(product_sales) * 100
                    ax_bar.axvline(x=idx_80+1, color='red', linestyle='--', 
                                  linewidth=2, alpha=0.5)
                    
                    # 标注80%分界点
                    y_pos = ax_bar.get_ylim()[1] * 0.6
                    ax_bar.text(idx_80+1, y_pos, 
                              f'{idx_80+1} products\n({pct_80:.1f}%)\n= 80% sales',
                              fontsize=9, ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor='yellow', alpha=0.8))
                    
                    # 在标题中显示帕累托比例
                    ax_bar.set_title(f'Month {month} - Pareto: {pct_80:.1f}%/80%', 
                                   fontsize=13, fontweight='bold')
                else:
                    ax_bar.set_title(f'Month {month} - Sales Ranking', 
                                   fontsize=13, fontweight='bold')
                
                ax_bar.grid(True, alpha=0.3, axis='y')
                ax_line.legend(fontsize=9, loc='lower right')
                
                # 添加统计信息
                info_text = f'Total Products: {len(product_sales)}\nTotal Sales: {total_quantity:.0f}'
                ax_bar.text(0.02, 0.98, info_text, 
                          transform=ax_bar.transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='left',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 隐藏未使用的子图
            for idx in range(len(plot_months), 4):
                axes[idx].axis('off')
            
            plt.suptitle('Product Sales Pareto Distribution - Multi-Month Comparison', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            plt.savefig(f"{self.charts_dir}product_sales_ranking_multi_months.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Multi-month product sales ranking chart saved")
            print(f"   📊 Plotted months: {plot_months}")
            
        except Exception as e:
            print(f"❌ Multi-month product sales ranking chart failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_firm_operation_rate(self, monthly_firm_operation_rate: Dict[int, Dict]):
        """9. Plot firm operation rate"""
        try:
            months = sorted(monthly_firm_operation_rate.keys())
            
            # Average operation rate per month
            avg_operation_rates = []
            for month in months:
                rates = [f['operation_rate'] for f in monthly_firm_operation_rate[month].values()]
                avg_rate = np.mean(rates) * 100 if rates else 0
                avg_operation_rates.append(avg_rate)
            
            # Last month operation rate distribution
            last_month = months[-1]
            operation_rates = [f['operation_rate'] * 100 
                             for f in monthly_firm_operation_rate[last_month].values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Operation rate trend
            ax1.plot(months, avg_operation_rates, marker='o', linewidth=2, color='#e74c3c')
            ax1.fill_between(months, avg_operation_rates, alpha=0.3, color='#e74c3c')
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Average Operation Rate (%)', fontsize=12)
            ax1.set_title('Monthly Average Firm Operation Rate Trend', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Operation rate distribution
            ax2.hist(operation_rates, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Operation Rate (%)', fontsize=12)
            ax2.set_ylabel('Number of Firms', fontsize=12)
            ax2.set_title(f'Firm Operation Rate Distribution (Month {last_month})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axvline(x=np.mean(operation_rates), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(operation_rates):.1f}%')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}firm_operation_rate.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Firm operation rate chart saved")
            
        except Exception as e:
            print(f"❌ Firm operation rate chart failed: {e}")
    
    def plot_top_products_restock_trend(self, monthly_product_sales: Dict[int, Dict], monthly_product_inventory: Dict[int, Dict]):
        """9. Plot top 10 products restock trend (sales + inventory changes)"""
        try:
            months = sorted(monthly_product_sales.keys())
            
            # 统计所有商品的总销量，找出Top 10
            product_total_sales = {}
            for month in months:
                for product_id, data in monthly_product_sales[month].items():
                    if product_id not in product_total_sales:
                        product_total_sales[product_id] = {
                            'name': data.get('name', 'Unknown'),
                            'total_quantity': 0
                        }
                    product_total_sales[product_id]['total_quantity'] += data.get('total_quantity', 0)
            
            # 按总销量排序，取Top 10
            top_products = sorted(product_total_sales.items(), 
                                key=lambda x: x[1]['total_quantity'], 
                                reverse=True)[:10]
            
            if not top_products:
                print("⚠️ No product sales data for restock analysis")
                return
            
            # 为每个Top 10商品计算月度补货量
            product_restock_data = {}
            for product_id, product_info in top_products:
                monthly_restock = []
                for i, month in enumerate(months):
                    # 计算补货量 = 本月销量 + (期末库存 - 期初库存)
                    sales = 0
                    if product_id in monthly_product_sales[month]:
                        sales = monthly_product_sales[month][product_id].get('total_quantity', 0)
                    
                    end_inventory = 0
                    if product_id in monthly_product_inventory.get(month, {}):
                        end_inventory = monthly_product_inventory[month][product_id].get('quantity', 0)
                    
                    # 期初库存 = 上月期末库存
                    start_inventory = 0
                    if i > 0:
                        prev_month = months[i-1]
                        if product_id in monthly_product_inventory.get(prev_month, {}):
                            start_inventory = monthly_product_inventory[prev_month][product_id].get('quantity', 0)
                    
                    # 补货量 = 销量 + (期末 - 期初)
                    restock = sales + (end_inventory - start_inventory)
                    monthly_restock.append(max(0, restock))  # 确保非负
                
                product_restock_data[product_id] = {
                    'name': product_info['name'],
                    'restock': monthly_restock
                }
            
            # 创建图表 - 2行5列展示Top 10商品的补货趋势
            fig, axes = plt.subplots(2, 5, figsize=(24, 10))
            axes = axes.flatten()
            
            colors = plt.cm.Paired(np.linspace(0, 1, 10))
            
            for idx, (product_id, data) in enumerate(product_restock_data.items()):
                ax = axes[idx]
                product_name = data['name']
                restock = data['restock']
                
                # 只保留ASCII字符，避免乱码
                try:
                    ascii_name = ''.join(c for c in product_name if ord(c) < 128)
                    if len(ascii_name.strip()) > 10:
                        product_name = ascii_name.strip()[:50]
                    else:
                        product_name = f"Product {product_id[:8]}"
                except:
                    product_name = f"Product {idx+1}"
                
                if len(product_name) > 50:
                    product_name = product_name[:47] + '...'
                
                # 绘制补货趋势
                ax.bar(months, restock, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                ax.plot(months, restock, marker='o', linewidth=2, markersize=5, 
                       color='darkblue', alpha=0.8, label='Restock Trend')
                
                ax.set_xlabel('Month', fontsize=10)
                ax.set_ylabel('Restock Quantity', fontsize=10)
                ax.set_title(f'#{idx+1}: {product_name}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=8)
                
                # 标注平均补货量
                avg_restock = np.mean(restock)
                ax.axhline(y=avg_restock, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(0.95, 0.95, f'Avg: {avg_restock:.1f}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.suptitle('Top 10 Best-Selling Products - Monthly Restock Quantity', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"{self.charts_dir}top_products_restock_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Top products restock trend chart saved")
            
        except Exception as e:
            print(f"❌ Top products restock trend chart failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_annual_firm_profit_distribution(self, monthly_firm_revenue: Dict[int, Dict]):
        """10. Plot annual firm profit distribution (cumulative profit over all months)"""
        try:
            if not monthly_firm_revenue:
                print("⚠️ No firm revenue data")
                return
            
            # 收集所有企业的全年利润数据
            firm_annual_profits = {}
            months = sorted(monthly_firm_revenue.keys())
            
            # 计算每个企业的全年累计利润
            for month in months:
                for firm_id, firm_data in monthly_firm_revenue[month].items():
                    if firm_id not in firm_annual_profits:
                        firm_annual_profits[firm_id] = 0.0
                    firm_annual_profits[firm_id] += firm_data.get('profit', 0.0)
            
            if not firm_annual_profits:
                print("⚠️ No firm profit data available")
                return
            
            # 提取利润数据
            annual_profit_values = list(firm_annual_profits.values())
            profitable_firms = [p for p in annual_profit_values if p > 0]
            loss_firms = [p for p in annual_profit_values if p < 0]
            break_even_firms = [p for p in annual_profit_values if p == 0]
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Annual Firm Profit Distribution Analysis', fontsize=16, fontweight='bold')
            
            # 1. 全年利润分布直方图
            ax1.hist(annual_profit_values, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even Line')
            ax1.axvline(x=np.mean(annual_profit_values), color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(annual_profit_values):,.0f}')
            ax1.set_xlabel('Annual Profit ($)', fontsize=12)
            ax1.set_ylabel('Number of Firms', fontsize=12)
            ax1.set_title('Annual Profit Distribution (All Firms)', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend()
            ax1.ticklabel_format(style='plain', axis='x')
            
            # 2. 盈利/亏损企业分类统计
            categories = ['Profitable', 'Loss-making', 'Break-even']
            counts = [len(profitable_firms), len(loss_firms), len(break_even_firms)]
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            
            bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Number of Firms', fontsize=12)
            ax2.set_title('Firm Profitability Classification', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # 3. 利润范围分布
            profit_ranges = {
                'Large Profit (>$50K)': len([p for p in annual_profit_values if p > 50000]),
                'Medium Profit ($10K-$50K)': len([p for p in annual_profit_values if 10000 <= p <= 50000]),
                'Small Profit ($0-$10K)': len([p for p in annual_profit_values if 0 < p < 10000]),
                'Small Loss ($0 to -$10K)': len([p for p in annual_profit_values if -10000 <= p < 0]),
                'Medium Loss (-$10K to -$50K)': len([p for p in annual_profit_values if -50000 <= p < -10000]),
                'Large Loss (<-$50K)': len([p for p in annual_profit_values if p < -50000])
            }
            
            range_labels = list(profit_ranges.keys())
            range_counts = list(profit_ranges.values())
            range_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
            
            bars = ax3.bar(range_labels, range_counts, color=range_colors, alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Number of Firms', fontsize=12)
            ax3.set_title('Profit Range Distribution', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45)
            
            # 在柱状图上添加数值标签
            for bar, count in zip(bars, range_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # 4. 利润趋势分析（按月份显示平均利润）
            monthly_avg_profits = []
            monthly_profitable_ratios = []
            
            for month in months:
                month_profits = [firm_data.get('profit', 0) for firm_data in monthly_firm_revenue[month].values()]
                avg_profit = np.mean(month_profits) if month_profits else 0
                profitable_count = len([p for p in month_profits if p > 0])
                profitable_ratio = (profitable_count / len(month_profits)) * 100 if month_profits else 0
                
                monthly_avg_profits.append(avg_profit)
                monthly_profitable_ratios.append(profitable_ratio)
            
            # 双Y轴显示平均利润和盈利企业比例
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(months, monthly_avg_profits, marker='o', linewidth=2, 
                           color='#3498db', label='Average Monthly Profit')
            ax4.fill_between(months, monthly_avg_profits, alpha=0.3, color='#3498db')
            ax4.set_xlabel('Month', fontsize=12)
            ax4.set_ylabel('Average Monthly Profit ($)', fontsize=12, color='#3498db')
            ax4.tick_params(axis='y', labelcolor='#3498db')
            ax4.grid(True, alpha=0.3)
            
            line2 = ax4_twin.plot(months, monthly_profitable_ratios, marker='s', linewidth=2, 
                                color='#e74c3c', label='Profitable Firms %')
            ax4_twin.set_ylabel('Profitable Firms (%)', fontsize=12, color='#e74c3c')
            ax4_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax4_twin.set_ylim(0, 100)
            
            # 添加零利润线
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
            ax4.set_title('Monthly Profit Trends', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{self.charts_dir}annual_firm_profit_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印统计信息
            total_firms = len(annual_profit_values)
            profitable_count = len(profitable_firms)
            loss_count = len(loss_firms)
            break_even_count = len(break_even_firms)
            
            print(f"✅ Annual firm profit distribution chart saved")
            print(f"   📊 Total Firms: {total_firms}")
            print(f"   📈 Profitable Firms: {profitable_count} ({profitable_count/total_firms*100:.1f}%)")
            print(f"   📉 Loss-making Firms: {loss_count} ({loss_count/total_firms*100:.1f}%)")
            print(f"   ⚖️ Break-even Firms: {break_even_count} ({break_even_count/total_firms*100:.1f}%)")
            print(f"   💰 Average Annual Profit: ${np.mean(annual_profit_values):,.2f}")
            print(f"   📊 Median Annual Profit: ${np.median(annual_profit_values):,.2f}")
            
        except Exception as e:
            print(f"❌ Annual firm profit distribution chart failed: {e}")
            import traceback
            traceback.print_exc()