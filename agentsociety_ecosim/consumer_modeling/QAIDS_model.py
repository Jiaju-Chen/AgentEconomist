import numpy as np

def predict_q_aids(expenses, next_budget, categories=None):
    """
    使用简化 QAIDS 模型，根据过去若干年各类支出预测下一年各类预算分配。
    
    参数:
    - expenses: array-like, shape (n_years, n_categories)，过去 n 年的各类别支出
    - next_budget: float，下年总预算
    - categories: list of str, 可选，各类别名称。默认按顺序 ["food","dress","house","daily","med","trco","eec","other"]
    
    返回:
    - dict: {category: 预测金额, ...}
    """
    arr = np.asarray(expenses, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expenses 应为二维数组，形状 (n_years, n_categories)")
    
    n_years, n_cat = arr.shape
    if categories is None:
        if n_cat != 8:
            # 如果类别数量非 8 且未提供名称，则用索引命名
            categories = [f"cat{i}" for i in range(n_cat)]
        else:
            categories = ["food", "dress", "house", "daily", "med", "trco", "eec", "other"]
    elif len(categories) != n_cat:
        raise ValueError("categories 列表长度应与 expenses 的列数匹配")
    
    totals = arr.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("每年总支出应大于 0")
    
    ln_tot = np.log(totals)
    ln_next = np.log(next_budget)
    
    # 对每个类别拟合 share ~ a*(ln_tot)^2 + b*ln_tot + c
    coeffs = [np.polyfit(ln_tot, arr[:, i] / totals, 2) for i in range(n_cat)]
    
    # 预测下一年各类别预算份额
    raw = np.array([np.polyval(coeffs[i], ln_next) for i in range(n_cat)])
    # 保证非负
    raw = np.clip(raw, 0, None)
    # 归一化为份额
    if raw.sum() > 0:
        shares = raw / raw.sum()
    else:
        shares = np.ones(n_cat) / n_cat
    
    preds = shares * next_budget
    # 保留一位小数
    preds = np.round(preds, 1)
    return {k: float(v) for k, v in zip(categories, preds)}

