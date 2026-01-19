import pandas as pd
import numpy as np

def _alloc_targets_per_category(prod_cnt: pd.Series, total_target: int, min_per_cat: int = 100):
    prod_cnt = prod_cnt[prod_cnt > 0].copy()
    cats = prod_cnt.index

    base = np.floor(prod_cnt * (total_target / max(prod_cnt.sum(), 1))).astype(int)
    base = base.clip(lower=min_per_cat)
    base = np.minimum(base, prod_cnt)

    diff = total_target - base.sum()

    if diff != 0:
        room_up = (prod_cnt - base).clip(lower=0)
        room_dn = (base - min_per_cat).clip(lower=0)

        if diff > 0:
            order = prod_cnt.sort_values(ascending=False).index.tolist()
            for c in order:
                if diff == 0: break
                add = min(room_up[c], diff)
                base[c] += add
                diff -= add
        else:
            order = prod_cnt.sort_values(ascending=True).index.tolist()
            for c in order:
                if diff == 0: break
                dec = min(room_dn[c], -diff)
                base[c] -= dec
                diff += dec

        if diff != 0:
            for c in cats:
                if diff == 0: break
                if diff > 0 and base[c] < prod_cnt[c]:
                    base[c] += 1
                    diff -= 1
                elif diff < 0 and base[c] > min_per_cat:
                    base[c] -= 1
                    diff += 1

    return base.astype(int)

def _stratified_sample_by_price(df_cat: pd.DataFrame, k: int, price_col: str, random_state: int = 42):
    df_cat = df_cat.copy()
    if price_col not in df_cat.columns or df_cat[price_col].isna().all():
        return df_cat.sample(n=k, random_state=random_state)

    df_price = df_cat.dropna(subset=[price_col])
    df_nan = df_cat[df_cat[price_col].isna()]
    n_nan = int(round(len(df_nan) / len(df_cat) * k)) if len(df_cat) > 0 else 0
    n_nan = min(n_nan, len(df_nan))
    n_price = k - n_nan

    if n_price <= 0:
        return df_nan.sample(n=k, random_state=random_state)

    q = df_price[price_col].quantile([0.0,0.2,0.4,0.6,0.8,1.0]).values
    q = np.unique(q)
    if len(q) <= 2:
        sample_price = df_price.sample(n=n_price, random_state=random_state)
    else:
        bins = q
        while len(np.unique(bins)) < len(bins):
            bins = np.array([b + 1e-9*i for i,b in enumerate(bins)])
        df_price = df_price.copy()
        df_price["_bin"] = pd.cut(df_price[price_col], bins=bins, include_lowest=True)
        bin_counts = df_price["_bin"].value_counts().sort_index()
        alloc = np.floor(bin_counts / bin_counts.sum() * n_price).astype(int)
        remain = n_price - alloc.sum()
        if remain > 0:
            extra_bins = bin_counts.sort_values(ascending=False).index.tolist()
            for b in extra_bins:
                if remain == 0: break
                alloc[b] += 1
                remain -= 1

        parts = []
        rng = np.random.default_rng(random_state)
        for b, num in alloc.items():
            pool = df_price[df_price["_bin"] == b]
            if num > 0 and len(pool) > 0:
                num = min(num, len(pool))
                parts.append(pool.sample(n=num, random_state=int(rng.integers(0, 10_000_000))))
        sample_price = pd.concat(parts, ignore_index=True) if parts else df_price.sample(n=n_price, random_state=random_state)

    sample_nan = df_nan.sample(n=n_nan, random_state=random_state) if n_nan > 0 else df_nan.iloc[0:0]
    return pd.concat([sample_price, sample_nan], ignore_index=True)

def reduce_products_and_update_map(
    products: pd.DataFrame,
    new_map: pd.DataFrame,
    households: int,
    category_col: str = "daily_cate",
    price_col: str = "price",
    min_per_cat: int = 100,
    multiplier: int = 12,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    total_target = int(max(1, households * multiplier))

    # 用 Uniq Id 替代 product_id
    used_pids = set(new_map['product_id'].unique())
    prod_pool = products[products['Uniq Id'].isin(used_pids)].copy()

    total_target = min(total_target, len(prod_pool))

    cnt = prod_pool.groupby(category_col).size().rename('n_prod')
    targets = _alloc_targets_per_category(cnt, total_target, min_per_cat=min_per_cat)

    chosen_parts = []
    for c, k in targets.items():
        if k <= 0: continue
        df_cat = prod_pool[prod_pool[category_col] == c]
        if len(df_cat) == 0: continue
        k = min(k, len(df_cat))
        sample_c = _stratified_sample_by_price(
            df_cat, k=k, price_col=price_col,
            random_state=int(rng.integers(0, 10_000_000))
        )
        chosen_parts.append(sample_c[['Uniq Id', category_col, price_col]] if price_col in df_cat.columns else sample_c[['Uniq Id', category_col]])

    reduced_products = pd.concat(chosen_parts, ignore_index=True) if chosen_parts else prod_pool.iloc[0:0]
    chosen_pids = set(reduced_products['Uniq Id'])

    new_map_reduced = new_map[new_map['product_id'].isin(chosen_pids)].copy()

    missing_pids = chosen_pids - set(new_map_reduced['product_id'])
    if missing_pids:
        nm_tmp = new_map.merge(products[['Uniq Id', category_col]], left_on='product_id', right_on='Uniq Id', how='left')
        comp_cat_counts = nm_tmp.groupby(['company_id', category_col]).size().rename('cnt').reset_index()
        idx = comp_cat_counts.sort_values(['company_id','cnt'], ascending=[True,False]).groupby('company_id').head(1)
        comp_main_cat = dict(zip(idx['company_id'], idx[category_col]))

        map_reduced_cat = new_map_reduced.merge(products[['Uniq Id', category_col]], left_on='product_id', right_on='Uniq Id', how='left')
        companies_in_cat = (
            map_reduced_cat.groupby('company_id')[category_col].agg(lambda s: s.value_counts().idxmax() if len(s.dropna()) else np.nan)
                           .dropna()
        )
        cat_to_companies = {}
        for cid, c in companies_in_cat.items():
            cat_to_companies.setdefault(c, []).append(cid)

        cur_load = new_map_reduced.groupby('company_id').size().rename('load')
        all_companies = set(new_map['company_id'].unique())
        cur_load = cur_load.reindex(list(all_companies), fill_value=0)

        prod_to_cat = dict(zip(products['Uniq Id'], products[category_col]))

        added_rows = []
        for pid in missing_pids:
            c = prod_to_cat.get(pid, None)
            if c is None: continue
            cands = cat_to_companies.get(c, [])
            if not cands:
                cands = [cid for cid, mc in comp_main_cat.items() if mc == c]
            if not cands: continue
            loads = cur_load.loc[cands]
            target_cid = loads.idxmin()
            added_rows.append({'company_id': target_cid, 'product_id': pid})
            cur_load.loc[target_cid] += 1

        if added_rows:
            new_map_reduced = pd.concat([new_map_reduced, pd.DataFrame(added_rows)], ignore_index=True)

    report = (
        new_map_reduced.merge(products[['Uniq Id', category_col]], left_on='product_id', right_on='Uniq Id', how='left')
                       .groupby(category_col)
                       .agg(companies_used=('company_id','nunique'),
                            products_assigned=('product_id','nunique'))
                       .assign(avg_products_per_company=lambda d: d['products_assigned'] / d['companies_used'])
                       .sort_values('products_assigned', ascending=False)
    )
    new_map_reduced.to_csv("data/company_product_map_rescaled.csv", index=False)
    return reduced_products, new_map_reduced, report

