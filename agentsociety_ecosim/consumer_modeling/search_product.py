#!/usr/bin/env python
# coding: utf-8
import pandas as pd

def load_and_process_product_data(path, keep_levels=False):
    """
    Loads, processes, and returns the product data.
    If keep_levels=True，则保留 level1/level2/leaf 列。
    """
    data = pd.read_csv(path)

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

    # Apply the category mapping
    filtered_data['budget_category'] = filtered_data['level1'].apply(map_walmart_category)

    company_cluster = pd.read_csv('consumer_modeling/company_cluster.csv', header=None, names=['company_id'])
    filtered_data['company_id'] = company_cluster['company_id'].values
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['company_id'] = 'firm' + filtered_data['company_id'].astype(str)
    # 过滤掉价格为0或负数的商品
    filtered_data = filtered_data[filtered_data['List Price'] > 0]
    
    # Add this line to print the column names
    print(filtered_data.columns)
    if not keep_levels:
        filtered_data = filtered_data.drop(['level1', 'level2', 'leaf'], axis=1)
    return filtered_data

def load_product_data(path):
    data = load_and_process_product_data(path)
    return data

def find_level2(level1,df):
    """
    Find the level2 category based on level1.
    """
    level2 = df[df['level1'] == level1]['level2'].unique()
    return level2

def find_leaf(level2, df):
    """
    Find the leaf category based on level2.
    """
    leaf = df[df['level2'] == level2]['leaf'].unique()
    return leaf

def find_product(budget_category, df):
    """
    Find the products based on budget_category.
    """
    products = df[df['budget_category'] == budget_category]
    return products

def map_walmart_category(walmart_category):
    """
    Maps Walmart product categories to budget categories.
    """
    category_mapping = {
        'health': "med",
        'premium beauty': "dress",
        'sports & outdoors': "eec",
        'baby': "daily",
        'food': "food",
        'clothing': "dress",
        'home': "house",
        'personal care': "daily",
        'toys': "eec",
        'pets': "daily",
        'auto & tires': "trco",
        'household essentials': "daily",
        'patio & garden': "house",
        'beauty': "dress",
        'home improvement': "house",
        'shop by brand': "other",
        'feature': "other",
        'party & occasions': "daily",
        'electronics': "trco",
        'industrial & scientific': "other",
        'seasonal': "other",
        'books': "eec",
        'walmart for business': "other",
        'office supplies': "daily",
        'video games': "eec",
        'arts crafts & sewing': "eec",
        'shop by movie': "eec",
        'jewelry': "dress",
        'music': "eec",
        'shop by video game': "eec",
        'character shop': "eec",
        'arts, crafts & sewing': "eec",
        'cell phones': "trco",
        'collectibles': "eec",
# In[ ]:'musical instruments': "eec"
    }
    return category_mapping.get(walmart_category, "other")  # Default to "other" if not found

def print_product_tree(df, max_leaf=5):
    """
    打印商品的树状结构（level1 -> level2 -> leaf），每个分支下最多展示 max_leaf 个叶子类别。
    """
    # 检查必要的列是否存在
    required_cols = ['level1', 'level2', 'leaf']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in DataFrame, skip print_product_tree.")
            return
    
    level1s = df['level1'].dropna().unique()
    for l1 in level1s:
        print(f"{l1}")
        level2s = df[df['level1'] == l1]['level2'].dropna().unique()
        for l2 in level2s:
            print(f"  └─{l2}")
            leaves = df[(df['level1'] == l1) & (df['level2'] == l2)]['leaf'].dropna().unique()
            for leaf in leaves[:max_leaf]:
                print(f"      └─{leaf}")
            if len(leaves) > max_leaf:
                print(f"      └─...({len(leaves)-max_leaf} more)")

