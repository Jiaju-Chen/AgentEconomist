# TODO: 构建整体的运行逻辑

import ray
ray.shutdown()  # Ensure Ray is not already running
ray.init(runtime_env={"env_vars": {"RAY_ADDRESS": "1"}})

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from agentsociety_ecosim.utils.log_utils import setup_global_logger

logger = setup_global_logger(name="ecosim", log_dir="logs", level="INFO")
  
import asyncio
import random
from typing import List
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from tqdm import tqdm
# 导入模拟相关模块
from agentsociety_ecosim.agent.firm import Firm
from agentsociety_ecosim.agent.government import Government
from agentsociety_ecosim.agent.household import Household
from agentsociety_ecosim.agent.bank import Bank
from agentsociety_ecosim.center.ecocenter import EconomicCenter
from agentsociety_ecosim.center.jobmarket import LaborMarket
from agentsociety_ecosim.center.assetmarket import ProductMarket
from agentsociety_ecosim.logger import get_logger
from agentsociety_ecosim.utils.data_loader import *

# 为 MCP 服务器设置：如果环境变量 MCP_MODE 存在，强制使用 CPU
if os.getenv('MCP_MODE'):
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用环境变量或相对路径获取模型路径
model_path = os.getenv("MODEL_PATH")
if not model_path:
    # 默认使用相对路径（从当前文件向上找到项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # 从 simulation/ 到 agentsociety-ecosim/
    model_path = os.path.join(project_root, "model", "all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)

client = QdrantClient(url="http://localhost:6333")
collection_name = "all_products"

if not client.get_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

std_job = load_jobs()
job_dis = load_job_dis()


async def limited_find_jobs(semaphore, household:Household):
    async with semaphore:
        return await household.find_jobs()

async def limited_init(semaphore, h):
    async with semaphore:
        await h.initialize()

async def run_simulation(
    economic_center:EconomicCenter,
    product_market:ProductMarket,
    labor_market:LaborMarket,
    government:Government,
    households:List[Household],
    firms:List[Firm],
    bank=None,  # 新增银行参数
    num_iterations=10,
):
    logger.info("===== Starting Economic Simulation =====")
    # print("===== Starting Economic Simulation =====")
    for iter in tqdm(range(num_iterations)):

        logger.info(f"Month {iter + 1}/{num_iterations}")
        # print(f"Month {iter + 1}/{num_iterations}")
        # 企业发布工作
        for firm in firms:
            await firm.define_job_openings(job_dis, std_job, labor_market)
        logger.info(f"Month {iter + 1}: Firms have posted jobs.")
        # print(f"Month {iter + 1}: Firms have posted jobs.")
        # 家庭找工作
        # random.shuffle(households)
        logger.info(f"Month {iter + 1}: Households are finding jobs.")
        # print(f"Month {iter + 1}: Households are finding jobs.")
        # 随机选取10个家庭
        households = households[:10] 

        # 控制最大同时并发任务数，比如 200
        semaphore = asyncio.Semaphore(200)

        tasks = [limited_find_jobs(semaphore, h) for h in households]

        all_matched_jobs = await asyncio.gather(*tasks)

        # 统计匹配到的工作数量
        total_matched_jobs = sum(len(jobs) for jobs in all_matched_jobs)
        logger.info(f"Month {iter + 1}: Total matched jobs: {total_matched_jobs}")
        # print(f"Month {iter + 1}: Total matched jobs: {total_matched_jobs}")
        # 对齐工作
        align_jobs_tasks = []
        for household, matched_jobs in zip(households, all_matched_jobs):
            if matched_jobs:
                count += 1
                logger.info(f"Household {household.household_id} found {len(matched_jobs)} jobs.")
                logger.info(f"Household {household.household_id} found {len(matched_jobs)} jobs.")
                # print(f"Household {household.household_id} found {len(matched_jobs)} jobs.")
                for job in matched_jobs:
                    align_jobs_tasks.append(labor_market.align_job.remote(household.household_id, job))
        all_align = await asyncio.gather(*align_jobs_tasks)

        # 统计对齐的工作数量
        count = 0
        for align in all_align:
            if align:
                count += 1
        logger.info(f"Month {iter + 1}: Total {count} jobs aligned with households.")
        # print(f"Month {iter + 1}: Total {count} jobs aligned with households.")
        # 劳动力市场更新工作
        await labor_market.update_jobs.remote()

        # 家庭消费
        logger.info(f"Iteration {iter + 1}: Households are consuming products.")
        logger.info(f"Month {iter + 1}: Households are consuming products.")
        # print(f"Month {iter + 1}: Households are consuming products.")
        for household in households:
            household.set_current_month(iter + 1)
            await household.consume(product_market, bank)  # 传递银行参数

        # 处理工资 
        logger.info(f"Month {iter + 1}: Processing wages for all jobs.")
        await labor_market.process_wages.remote(economic_center, iter + 1)
        
        # 月末处理：银行利息发放和税收再分配
        current_month = iter + 1
        
        # 1. 银行发放利息
        if bank:
            print(f"\n💰 ===== 第 {current_month} 月银行利息发放 =====")
            logger.info(f"Month {current_month}: Calculating and paying bank interest...")
            
            # 获取银行摘要（发放前）
            bank_summary_before = await bank.get_bank_summary.remote()
            print(f"📊 发放前银行状态:")
            print(f"   活跃储蓄账户: {bank_summary_before['active_accounts']} 个")
            print(f"   总存款: ${bank_summary_before['total_deposits']:.2f}")
            print(f"   累计已付利息: ${bank_summary_before['total_interest_paid']:.2f}")
            
            total_interest = await bank.calculate_and_pay_monthly_interest.remote(current_month)
            
            # 获取银行摘要（发放后）
            bank_summary_after = await bank.get_bank_summary.remote()
            print(f"✅ 第 {current_month} 月利息发放完成:")
            print(f"   本月发放利息: ${total_interest:.2f}")
            print(f"   新总存款: ${bank_summary_after['total_deposits']:.2f}")
            print(f"   累计已付利息: ${bank_summary_after['total_interest_paid']:.2f}")
            print(f"   平均账户余额: ${bank_summary_after['average_balance']:.2f}")
            
            logger.info(f"Month {current_month}: Bank paid total interest: ${total_interest:.2f}")
        
        # 2. 税收再分配
        print(f"\n🏛️ ===== 第 {current_month} 月税收再分配 =====")
        logger.info(f"Month {current_month}: Redistributing tax revenue...")
        
        redistribution_result = await economic_center.redistribute_monthly_taxes.remote(current_month)
        
        print(f"📊 税收再分配详情:")
        tax_breakdown = redistribution_result.get('tax_breakdown', {})
        print(f"   消费税收入: ${tax_breakdown.get('consume_tax', 0):.2f}")
        print(f"   个人所得税收入: ${tax_breakdown.get('labor_tax', 0):.2f}")
        print(f"   企业所得税收入: ${tax_breakdown.get('corporate_tax', 0):.2f}")
        print(f"   税收总额: ${redistribution_result.get('total_tax_collected', 0):.2f}")
        print(f"✅ 再分配完成:")
        print(f"   受益劳动者数量: {redistribution_result.get('recipients', 0)} 个")
        print(f"   人均分配金额: ${redistribution_result.get('per_person', 0):.2f}")
        print(f"   总再分配金额: ${redistribution_result.get('total_redistributed', 0):.2f}")
        
        logger.info(f"Month {current_month}: Tax redistribution completed - ${redistribution_result.get('total_redistributed', 0):.2f} distributed to {redistribution_result.get('recipients', 0)} workers")

    for household in households:
        total_income, total_spent = await economic_center.compute_household_settlement.remote(household.household_id)
        final_wealth = await household.get_balance_ref()
        logger.info(f"Household {household.household_id} - Total Income: {total_income}, Total Spent: {total_spent}, Initial Wealth: {household.initial_wealth}, Final Wealth: {final_wealth}")
        # print(f"Household {household.household_id} - Total Income: {total_income}, Total Spent: {total_spent}, Initial Wealth: {household.initial_wealth}, Final Wealth: {final_wealth}")
    for household in households:
        # 获取每月支出数据
        monthly_income, monthly_expense = await economic_center.compute_household_monthly_stats.remote(household.household_id)
        # monthly_expense Dict[month, float]
        for month, spent in monthly_expense.items():
            logger.info(f"Household {household.household_id} - Month {month}: Total Spent: ${spent:.2f}")

    # 输出银行统计信息
    if bank:
        logger.info("===== Bank Summary =====")
        bank_summary = await bank.get_bank_summary.remote()
        logger.info(f"Bank Statistics:")
        logger.info(f"  Total Accounts: {bank_summary['total_accounts']}")
        logger.info(f"  Active Accounts: {bank_summary['active_accounts']}")
        logger.info(f"  Total Deposits: ${bank_summary['total_deposits']:.2f}")
        logger.info(f"  Total Interest Paid: ${bank_summary['total_interest_paid']:.2f}")
        logger.info(f"  Average Balance: ${bank_summary['average_balance']:.2f}")

    logger.info("===== Simulation Completed =====")
    # print("===== Simulation Completed =====")
async def main():


    # Initialize Economic Center, Product Market, Labor Market, and Government
    economic_center = EconomicCenter.remote()    
    product_market = ProductMarket.remote()
    labor_market = LaborMarket.remote()

    government = Government.remote(
        government_id="gov_1",
        initial_budget=1000000.0,
        economic_center=economic_center
    )
    await government.initialize.remote()
    
    # 初始化银行
    bank = Bank.remote(
        bank_id="central_bank",
        initial_capital=10000000.0,  # 银行初始资本1000万
        economic_center=economic_center
    )
    await bank.initialize.remote()
    logger.info("Bank initialized with capital $10,000,000")

    # Create Households according to laborhour
    products = load_products()
    firm2product = load_product_map()

    households = []
    households_dict = load_households()
    for key, values in households_dict.items():
        household_id = key
        labor_hours = load_lh(household_id, values)
    households_dict = load_households()
    for key, values in households_dict.items():
        household_id = key
        labor_hours = load_lh(household_id, values)
        household = Household(
            household_id=household_id,
            household_id=household_id,
            economic_center=economic_center,
            labor_hour=labor_hours,
            labormarket=labor_market,
            product_market=product_market,
        )
        households.append(household)
    await asyncio.gather(*[limited_init(asyncio.Semaphore(200), h) for h in households])
    logger.info(f"Initialized {len(households)} households with labor hours.")
    # print(f"Initialized {len(households)} households with labor hours.")
    firms_df = load_firms_df()

    records = firms_df.to_dict(orient='records')
    firms = []
    for record in records:
        kwargs = Firm.parse_dicts(record)
        firm = Firm(**kwargs, economic_center=economic_center, product_market=product_market)
        firms.append(firm)
    await asyncio.gather(*[limited_init(asyncio.Semaphore(200), f) for f in firms])
    logger.info(f"Initialized {len(firms)} firms.")
    # print(f"Initialized {len(firms)} firms.")
    for firm in firms:
        load_products_firm(firm, products, firm2product, economic_center, product_market, model, tokenizer, client)
    logger.info("All firms and products initialized.")
    # print("All firms and products initialized.")
    await run_simulation(
        economic_center=economic_center,
        product_market=product_market,
        labor_market=labor_market,
        government=government,
        households=households,
        firms=firms,
        bank=bank,  # 传递银行参数
        num_iterations=12
    )
    try:
        await asyncio.sleep(99999)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting...")

    finally:
        print("Shutting down Ray...")
        ray.shutdown()




if __name__ == "__main__":
    # Run the simulation
    asyncio.run(main())