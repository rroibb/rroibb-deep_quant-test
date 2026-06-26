""""
滚动相关系数矩阵 — 股票间关系特征
输入: 多股票收盘价 DataFrame (行=日期, 列=股票代码)
输出: 每只股票对其他股票的平均/最大/最小相关系数、连通性等特征
"""
import numpy as np
import pandas as pd


def compute_rolling_correlation(price_df, window=20, min_periods=None):
    """
    计算滚动窗口内股票间的相关系数矩阵，提取每个节点(股票)的关系特征。

    参数:
        price_df: pd.DataFrame, index=日期, columns=股票代码, values=收盘价
        window:   滚动窗口大小 (默认20个交易日)
        min_periods: 最小有效天数 (默认 window//2)
    返回:
        pd.DataFrame, MultiIndex columns=(股票代码, 特征名)
        特征包括:
            - avg_corr:    与其余股票的平均相关系数 (行业关联度)
            - max_corr:    最大相关系数 (最紧密的联动关系)
            - min_corr:    最小相关系数 (最独立)
            - corr_std:    相关系数标准差 (关系多样性)
            - corr_skew:   相关系数偏度 (偏向正向还是负向关联)
            - eigen_ratio: 最大特征值占比 (系统性风险暴露)
            - centrality:  图中心度 = avg_corr / corr_std (影响力和独立性的比值)
    """
    if min_periods is None:
        min_periods = max(10, window // 2)

    returns = price_df.pct_change().dropna(how='all')
    stocks = returns.columns.tolist()
    n_stocks = len(stocks)
    dates = returns.index

    if n_stocks < 2:
        raise ValueError("至少需要2只股票")

    # 预分配结果
    multi_index = pd.MultiIndex.from_product(
        [stocks, ['avg_corr', 'max_corr', 'min_corr', 'corr_std',
                  'corr_skew', 'eigen_ratio', 'centrality']],
        names=['stock', 'feature']
    )
    result = pd.DataFrame(np.nan, index=dates, columns=multi_index)

    # 预计算所有日期的相关系数矩阵 (缓存)
    # 使用 rolling + manual corr 以避免计算全量
    for t in range(window - 1, len(dates)):
        end = t + 1
        start = end - window
        window_ret = returns.iloc[start:end]

        # 过滤有效样本
        valid_mask = window_ret.count() >= min_periods
        valid_stocks = valid_mask[valid_mask].index.tolist()
        if len(valid_stocks) < 2:
            continue

        valid_ret = window_ret[valid_stocks].dropna(axis=1, thresh=min_periods)
        if valid_ret.shape[1] < 2:
            continue

        corr = valid_ret.corr().values
        n_valid = corr.shape[0]
        current_date = dates[t]
        valid_cols = valid_ret.columns.tolist()

        # 对每只股票提取特征
        for i, stock in enumerate(valid_cols):
            row = np.delete(corr[i], i)  # 去掉自相关(总是1)

            avg = row.mean()
            max_v = row.max()
            min_v = row.min()
            std_v = row.std()
            skew_v = ((row - avg) ** 3).mean() / (std_v ** 3 + 1e-8)

            # 特征值分解 (1阶近似)
            eigen = np.linalg.eigvalsh(corr)
            eigen_ratio = eigen[-1] / (eigen.sum() + 1e-8)

            centrality = avg / (std_v + 1e-8)

            result.loc[current_date, (stock, 'avg_corr')] = avg
            result.loc[current_date, (stock, 'max_corr')] = max_v
            result.loc[current_date, (stock, 'min_corr')] = min_v
            result.loc[current_date, (stock, 'corr_std')] = std_v
            result.loc[current_date, (stock, 'corr_skew')] = skew_v
            result.loc[current_date, (stock, 'eigen_ratio')] = eigen_ratio
            result.loc[current_date, (stock, 'centrality')] = centrality

    return result


def rolling_corr_to_panel_features(price_df, panel_df, window=20):
    """
    便捷函数: 将相关系数特征合并到策略面板中。
    
    panel_df: 现有策略面板 (MultiIndex [date, code]), 必须包含 'close' 列
    返回: panel_df 增加 7 列相关系数特征
    """
    pivot = price_df.pivot_table(
        index=price_df.index if hasattr(price_df, 'index') else price_df.columns,
        values='close', columns='code', aggfunc='last'
    ) if isinstance(price_df, pd.DataFrame) and 'code' in price_df.columns else price_df

    if isinstance(price_df, pd.DataFrame) and 'code' in price_df.columns:
        pivot = price_df.pivot_table(
            index=price_df.index.get_level_values('date')
            if hasattr(price_df.index, 'get_level_values') else price_df.index,
            values='close', columns='code', aggfunc='last'
        )
    else:
        pivot = price_df.copy()

    corr_features = compute_rolling_correlation(pivot, window=window)

    panel_out = panel_df.copy()
    for stock in corr_features.columns.get_level_values(0).unique():
        for feat in ['avg_corr', 'max_corr', 'min_corr', 'corr_std',
                     'corr_skew', 'eigen_ratio', 'centrality']:
            col = (stock, feat)
            if col in corr_features.columns:
                series = corr_features.loc[:, col]
                for date in series.index:
                    if (date, stock) in panel_out.index:
                        panel_out.loc[(date, stock), f'corr_{feat}'] = series.loc[date]

    return panel_out