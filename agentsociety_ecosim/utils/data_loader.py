# 添加当前文件夹到 Python 路径
import sys
import os
from pathlib import Path

# 确保工作目录在 agentsociety_ecosim/
_CURRENT_FILE = Path(__file__).resolve()  # data_loader.py 的绝对路径
_UTILS_DIR = _CURRENT_FILE.parent  # utils/
_ECOSIM_DIR = _UTILS_DIR.parent  # agentsociety_ecosim/
os.chdir(_ECOSIM_DIR)  # 切换到 agentsociety_ecosim/ 目录

sys.path.append('.')
# 添加上级文件夹到 Python 路径
sys.path.append('..')
import pandas as pd
import ast
import numpy as np
import torch
from agentsociety_ecosim.center.model import Product, LaborHour
from agentsociety_ecosim.utils.product_attribute_loader import inject_product_attributes
from agentsociety_ecosim.utils.embedding import embedding
from qdrant_client.models import PointStruct
from uuid import uuid5, NAMESPACE_DNS, uuid4
from pandas import DataFrame
from agentsociety_ecosim.utils.log_utils import setup_global_logger

logger = setup_global_logger(__name__)

def load_cluster()->DataFrame:
    data = pd.read_csv('data/product/marketing_sample_for_walmart_com-product_details__20200101_20200331__30k_data.csv')


    filtered_data = data[data['Brand'].notna()].copy()
    filtered_data['Brand'] = filtered_data['Brand'].str.lower()
    filtered_data = filtered_data.drop(['Item Number', 'Available','Postal Code','Package Size','Sale Price', 'Gtin'], axis=1)
    filtered_data = filtered_data[filtered_data['Category'].notna()]
    filtered_data['Category'] = filtered_data['Category'].str.lower()

    df = pd.DataFrame(filtered_data['Category'])
    df['path'] = df['Category'].str.split(r'\s*\|\s*')
    df['level1'] = df['path'].apply(lambda x: x[0] if len(x) > 0 else '')
    df['level2'] = df['path'].apply(lambda x: x[1] if len(x) > 1 else '')
    df['leaf']   = df['path'].apply(lambda x: x[-1])  

    filtered_data.drop(['Category'], axis=1, inplace=True)
    filtered_data = pd.concat([filtered_data, df[['level1', 'level2', 'leaf']]], axis=1)

    company_cluster = pd.read_csv('data/product/company_cluster.csv', header=None, names=['company_id'])
    filtered_data['company_id'] = company_cluster['company_id'].values
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['company_id'] = 'firm' + filtered_data['company_id'].astype(str)
    # 过滤掉价格为0或负数的商品
    filtered_data = filtered_data[filtered_data['List Price'] > 0]
    filtered_data = filtered_data.sort_values(by=['level1', 'Product Name'], ascending=[True, True])
    return filtered_data

def load_processed_products():
    df = pd.read_csv('data/product/processed_products.csv')
    # 过滤掉价格为0或负数的商品
    df = df[df['List Price'] > 0].copy()
    df['List Price'] = df['List Price'].astype(float)
    return df

def load_name():
    name_df = pd.read_csv('data/product/company_names.csv', header=None, names=['name'])
    return name_df

def load_jobs_with_hmean():
    jobs = pd.read_excel('data/job/onet/Occupation Data.xlsx')
    jobs = jobs[jobs['O*NET-SOC Code'].str.endswith('00')].copy()
    jobs['O*NET-SOC Code'] = jobs['O*NET-SOC Code'].str.split('.').str[0]
   
    job_codes = jobs['O*NET-SOC Code'].unique()               

    salary = pd.read_excel('data/job/oews/all_data_M_2024.xlsx', usecols=['OCC_TITLE', 'OCC_CODE', 'H_MEAN', 'A_MEAN', 'PRIM_STATE' ])
    salary = salary[salary['PRIM_STATE'] == 'NY'].copy()
    for idx, row in salary.iterrows():
        if row['H_MEAN'] == '*' or row['H_MEAN'] == '#':
            try:
                # 保留两位小数
                row['H_MEAN'] = round(row['A_MEAN'] / 2080, 2)
                salary.at[idx, 'H_MEAN'] = row['H_MEAN']
            except Exception as e:
                salary.drop(idx, inplace=True)  
    salary["H_MEAN"] = pd.to_numeric(salary["H_MEAN"], errors="coerce")
    salary_code = salary['OCC_CODE'].unique()
    common_jobs = set(job_codes) & set(salary_code)
    filtered_salary = salary[salary['OCC_CODE'].isin(common_jobs)]
    avg_salary = (
        filtered_salary
        .groupby("OCC_CODE", as_index=False)['H_MEAN']
        .mean()
        .rename(columns={"H_MEAN": "Average_Wage"})
    )
    avg_salary['Average_Wage'] = avg_salary['Average_Wage'].round(2)
    jobs = jobs.merge(avg_salary, left_on='O*NET-SOC Code', right_on='OCC_CODE', how='left')
    jobs = jobs.drop(columns=['OCC_CODE'])
    jobs = jobs.dropna(subset=["Average_Wage"])
    return jobs

def load_jobs ():
    jobs = pd.read_csv('data/job/jobs_with_skills_abilities_IM_merged.csv')
    jobs['skills'] = jobs['skills'].apply(ast.literal_eval)
    jobs['abilities'] = jobs['abilities'].apply(ast.literal_eval)
    jobs['Average_Wage'] = jobs['Average_Wage'].astype(float)
    jobs = jobs.dropna(subset=['Average_Wage'])
    return jobs

jobs = load_jobs()

def load_labor_hours():
    jobs = load_jobs()
    all_labor_hours = []
    for idx, row in jobs.iterrows():
        job_id = row['O*NET-SOC Code']
        skill_dict = {}
        ability_dict = {}
        skill_profile = row['skills']
        ability_profile = row['abilities']
        for skill, dic in skill_profile.items():
            skill_name = skill
            mean = dic['mean']
            std = dic['std']
            skill_value = np.random.normal(loc=mean, scale=std)
            skill_dict[skill_name] = skill_value
        for ability, dic in ability_profile.items():
            ability_name = ability
            mean = dic['mean']
            std = dic['std']
            ability_value = np.random.normal(loc=mean, scale=std)
            ability_dict[ability_name] = ability_value
        labor_hour = LaborHour.create(
            agent_id=str(uuid4()),
            skill_profile=skill_dict,
            ability_profile=ability_dict,
            template=job_id,
            total_hours=40,
        )
        all_labor_hours.append(labor_hour)
    return all_labor_hours

def load_firms_df():
    firms_df = pd.read_csv('data/firm/daily_necessities_firms_with_category.csv')
    return firms_df

def load_job_dis():
    job_dis = pd.read_csv('data/industry_soc_distribution.csv')
    return job_dis

def load_products():
    df = pd.read_csv('data/product/processed_products.csv')
    # 过滤掉价格为0或负数的商品
    df = df[df['List Price'] > 0].copy()
    return df

def load_product_map():
    return pd.read_csv('data/firm2product.csv')

async def load_products_firm(firm, products, map, amount_config, economic_center, product_market, model, tokenizer):
    """
    为企业加载商品（不再直接操作 Qdrant，由 ProductMarket 统一管理）
    """
    id = firm.company_id
    product_firm = []
    logger.info(f"[ProductLoader] 开始注册企业 {id} 的商品，待处理 {len(products)} 条记录")
    for idx, row in products.iterrows():
        if row['daily_cate'] == 'Meat and Seafood' or row['daily_cate'] == 'Dairy Products' or row['daily_cate'] == 'Confectionery and Snacks' or row['daily_cate'] == 'Grains and Bakery' or row['daily_cate'] == 'Beverages':
            amount = amount_config['food_amount']
        else:
            amount = amount_config['non_food_amount']
        if row['List Price'] > 0:
            product_kwargs = dict(
                name=row['Product Name'],
                amount=amount,  
                price=row['List Price'],
                owner_id=id,
                classification=row['daily_cate'],
                brand=row['Brand'],
                product_id=row['Uniq Id'],
                description=row['Description']
            )
            product_kwargs = inject_product_attributes(product_kwargs, product_kwargs.get("product_id"))
            product = Product.create(**product_kwargs)
            product_firm.append(product)
            await economic_center.register_product.remote(id, product)
            await product_market.publish_product.remote(product)
            # logger.info(
            #     f"[ProductLoader] 已注册商品 {product.product_id} ({product.name}) "
            #     f"| 分类:{product.classification} 价格:{product.price} 属性已附加:{bool(product.attributes)}"
            # )
    logger.info(f"[ProductLoader] 企业 {id} 商品注册完毕，总计 {len(product_firm)} 条有效商品")
    
    # 🚀 批量加载到 Qdrant（通过 ProductMarket Actor）
    if product_firm:
        await product_market.batch_load_to_qdrant.remote(product_firm)


def load_product_to_qdrant(model, tokenizer, client, product_list):
    """
    批量加载商品向量到 Qdrant（使用批量 embedding 加速）
    """
    collection_name = "part_products"
    
    # 🚀 批量处理：先收集所有文本
    texts = []
    for product in product_list:
        text = ' '.join([product.name, product.brand, product.description or '', product.classification])
        texts.append(text)
    
    # 🚀 批量计算所有向量（加速 5-10 倍）
    from agentsociety_ecosim.utils.embedding import batch_embedding
    vectors = batch_embedding(texts, tokenizer, model, batch_size=32)
    
    # 构建 Qdrant points
    points = []
    for product, vector in zip(product_list, vectors):
        payload = {
            "name": product.name,
            "Uniq Id": product.product_id,
            "description": product.description,
            "classification": product.classification,
            "price": product.price,
            "owner_id": product.owner_id,
            "description": product.description or ""  # 确保 description 不为 None
        }
        
        # 🔥 使用复合ID确保竞争模式下同一商品的不同供应商都能存储
        # Qdrant只接受整数或UUID，所以将复合字符串转换为UUID5（确定性UUID）
        composite_string = f"{product.product_id}@{product.owner_id}"
        unique_id = str(uuid5(NAMESPACE_DNS, composite_string))
        points.append(PointStruct(id=unique_id, vector=vector, payload=payload))
    
    # 批量插入 Qdrant
    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"[Qdrant] 批量插入 {len(points)} 个商品向量")


def allocate_products(products, firms_df, random_state):
    companies = firms_df

    # 规范类别名
    if 'daily_cate' in products.columns:
        products['daily_cate'] = products['daily_cate'].astype(str).str.strip()
    if 'industry_category' in companies.columns:
        companies['industry_category'] = companies['industry_category'].astype(str).str.strip()

    # 仅统计有商品的类别，并设总公司目标为两倍此数量（记录用）
    categories = sorted(products['daily_cate'].dropna().unique().tolist())
    total_target_companies = 2 * len(categories)

    rng = np.random.default_rng(random_state)
    new_maps = []

    for c in categories:
        prod_c = products.loc[products['daily_cate'] == c, :]
        if prod_c.empty:
            continue

        comp_c = companies.loc[companies['industry_category'] == c, :]
        # 按用户前置保证：每类至少2家公司
        if len(comp_c) < 2:
            raise ValueError(f"类别 {c} 可用公司少于2家，无法按每类2家公司分配")

        chosen_companies = comp_c.sample(n=2, random_state=random_state)

        # 打乱并平均切块
        prod_ids = prod_c['Uniq Id'].sample(frac=1.0, random_state=random_state).tolist()
        chunks = np.array_split(prod_ids, 2)

        comp_ids = chosen_companies['factset_entity_id'].tolist()
        for comp_id, chunk in zip(comp_ids, chunks):
            if len(chunk) == 0:
                continue
            tmp = pd.DataFrame({'company_id': comp_id, 'product_id': list(chunk)})
            new_maps.append(tmp)

    new_map = pd.concat(new_maps, ignore_index=True) if new_maps else pd.DataFrame(columns=['company_id','product_id'])
    new_map.to_csv("data/company_product_map_rescaled.csv", index=False)
    return new_map

def allocate_products_competitive(products, firms_df, random_state):
    """
    竞争性商品分配：同一类别的企业销售相同的商品（创新破坏理论）
    
    与 allocate_products 的区别：
    - allocate_products: 将商品平均分配给各企业（不同企业卖不同商品）
    - allocate_products_competitive: 所有企业销售相同商品（企业间竞争市场份额）
    
    Args:
        products: 商品数据框
        firms_df: 企业数据框
        random_state: 随机种子
    
    Returns:
        新的映射表 (company_id, product_id)，同一类别的所有企业共享所有商品
    """
    companies = firms_df

    # 规范类别名
    if 'daily_cate' in products.columns:
        products['daily_cate'] = products['daily_cate'].astype(str).str.strip()
    if 'industry_category' in companies.columns:
        companies['industry_category'] = companies['industry_category'].astype(str).str.strip()

    # 仅统计有商品的类别
    categories = sorted(products['daily_cate'].dropna().unique().tolist())
    total_target_companies = 2 * len(categories)

    rng = np.random.default_rng(random_state)
    new_maps = []

    for c in categories:
        prod_c = products.loc[products['daily_cate'] == c, :]
        if prod_c.empty:
            continue

        comp_c = companies.loc[companies['industry_category'] == c, :]
        # 按用户前置保证：每类至少2家公司
        if len(comp_c) < 2:
            raise ValueError(f"类别 {c} 可用公司少于2家，无法按每类2家公司分配")

        chosen_companies = comp_c.sample(n=2, random_state=random_state)

        # 🔥 关键区别：所有企业获得相同的商品列表（不分块）
        prod_ids = prod_c['Uniq Id'].tolist()
        
        # 每家企业都分配到该类别的所有商品
        comp_ids = chosen_companies['factset_entity_id'].tolist()
        for comp_id in comp_ids:
            if len(prod_ids) == 0:
                continue
            tmp = pd.DataFrame({'company_id': comp_id, 'product_id': prod_ids})
            new_maps.append(tmp)

    new_map = pd.concat(new_maps, ignore_index=True) if new_maps else pd.DataFrame(columns=['company_id','product_id'])
    new_map.to_csv("data/company_product_map_competitive.csv", index=False)
    return new_map

def load_households():
    household_data = pd.read_csv('data/household/psid_2019_updated.csv')
    # household_data = household_data[household_data['household_id'] < 500]

    household_data['household_id'] = household_data['household_id'].astype(int).astype(str)
    
    households = {}
    for idx, row in household_data.iterrows():
        household_id = row['household_id']
        head_job = row['head_job1_soc_code']
        spouse_job = row['spouse_job1_soc_code']
        households[household_id] = {
            "head_job": head_job,
            "spouse_job": spouse_job
        }
    return households

def load_lh(id, value):
    '''
    根据head_job和spouse_job生成对应的laborhour
    '''
    jobs_id = jobs['O*NET-SOC Code'].unique().tolist()
    head_job = value['head_job']
    spouse_job = value['spouse_job']
    lh = []
    if head_job in jobs_id:
        head_skill, head_ability = create_profile(head_job)
        lh.append(LaborHour.create(
            agent_id=id,
            skill_profile=head_skill,
            ability_profile=head_ability,
            template=head_job,
            total_hours=40,
            lh_type='head'
        ))
    # if spouse_job != '00-0000':
    if spouse_job in jobs_id:
        spouse_skill, spouse_ability = create_profile(spouse_job)
        lh.append(LaborHour.create(
            agent_id=id,
            skill_profile=spouse_skill,
            ability_profile=spouse_ability,
            template=spouse_job,
            total_hours=40,
            lh_type='spouse'
        ))

    return lh

def create_profile(job):
    skill_dict = {}
    ability_dict = {}
    std_job_info = jobs[jobs['O*NET-SOC Code'] == job]
    skill_profile = std_job_info.iloc[0]['skills']
    ability_profile = std_job_info.iloc[0]['abilities']
    for skill, dic in skill_profile.items():
        skill_name = skill
        mean = dic['mean']
        std = dic['std']
        skill_value = np.random.normal(loc=mean, scale=std)
        skill_dict[skill_name] = skill_value
    for ability, dic in ability_profile.items():
        ability_name = ability
        mean = dic['mean']
        std = dic['std']
        ability_value = np.random.normal(loc=mean, scale=std)
        ability_dict[ability_name] = ability_value

    return skill_dict, ability_dict


def match_pro_firm(product_id):
    mapping = load_product_map()
    firm_id = mapping[mapping['product_id'] == product_id]['firm_id'].values
    return firm_id[0]



if __name__ == "__main__":
    # Example usage
    df = load_jobs_with_hmean()
    print(df.head())