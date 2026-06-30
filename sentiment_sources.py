import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = './sentiment_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# 东方财富综合情绪：所有股票一日快照
_CACHED_COMMENT_DF = None
_CACHED_COMMENT_DATE = None


def _ts_code_to_ak(ticker):
    code = ticker.split('.')[0]
    return code


def fetch_eastmoney_comment_all():
    global _CACHED_COMMENT_DF, _CACHED_COMMENT_DATE
    today_str = datetime.now().strftime('%Y-%m-%d')
    cache_file = os.path.join(CACHE_DIR, f'eastmoney_comment_{today_str}.parquet')

    if _CACHED_COMMENT_DF is not None and _CACHED_COMMENT_DATE == today_str:
        return _CACHED_COMMENT_DF

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                _CACHED_COMMENT_DF = df
                _CACHED_COMMENT_DATE = today_str
                return df
        except:
            pass

    try:
        import akshare as ak
        df = ak.stock_comment_em()
        col_map = {
            df.columns[0]: 'rank', df.columns[1]: 'code', df.columns[2]: 'name',
            df.columns[3]: 'price', df.columns[4]: 'change_pct', df.columns[5]: 'turnover_rate',
            df.columns[6]: 'pe', df.columns[7]: 'main_power_cost',
            df.columns[8]: 'inst_participation', df.columns[9]: 'composite_score',
            df.columns[10]: 'rise', df.columns[11]: 'rank_position',
            df.columns[12]: 'attention_index', df.columns[13]: 'date',
        }
        df = df.rename(columns=col_map)
        df = df.drop_duplicates(subset=['code'])
        df['code'] = df['code'].astype(str).str.zfill(6)
        for col in ['composite_score', 'attention_index', 'inst_participation']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50.0)
        df['date'] = df['date'].astype(str)
        for col in ['price', 'change_pct', 'turnover_rate', 'pe', 'main_power_cost', 'rise', 'rank_position']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_parquet(cache_file, index=False)
        _CACHED_COMMENT_DF = df
        _CACHED_COMMENT_DATE = today_str
        return df
    except Exception as e:
        print(f"  [sentiment] 东方财富情绪获取失败: {e}")
        return pd.DataFrame()


def _fetch_detail_single(ticker, func_name, cache_key):
    ak_code = _ts_code_to_ak(ticker)
    cache_file = os.path.join(CACHE_DIR, f'{cache_key}_{ak_code}.parquet')

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty and len(df) >= 10:
                return df
        except:
            pass

    try:
        import akshare as ak
        fn = getattr(ak, func_name)
        df = fn(symbol=ak_code)
        if df is not None and not df.empty and len(df) > 5:
            df = df.copy()
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col]).astype(str)
            for i in range(1, len(df.columns)):
                df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce').fillna(0.0)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_parquet(cache_file)
        return df
    except Exception as e:
        print(f"    [sentiment] {func_name}/{ak_code}: {e}")
        return pd.DataFrame()


def fetch_stock_sentiment_detail(ticker, days=20):
    result = {}

    for func_name, col_name, cache_key, val_idx in [
        ('stock_comment_detail_scrd_desire_em', 'buy_desire', 'desire', 2),
        ('stock_comment_detail_zhpj_lspf_em', 'composite_score', 'score', 1),
        ('stock_comment_detail_scrd_focus_em', 'attention_index', 'focus', 1),
        ('stock_comment_detail_zlkp_jgcyd_em', 'inst_participation', 'inst', 1),
    ]:
        df = _fetch_detail_single(ticker, func_name, cache_key)
        if df is not None and not df.empty and len(df.columns) > val_idx:
            date_col = df.columns[0]
            val_col = df.columns[val_idx]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                result[col_name] = df[val_col].astype(float).tail(days)
            except Exception:
                continue

    if not result:
        return pd.DataFrame()

    combined = pd.DataFrame(result)
    combined.index.name = 'trade_date'
    return combined


def fetch_all_sentiment_features(tickers, days=20):
    all_data = {}
    print(f"\n获取 {len(tickers)} 只股票的情绪数据...")

    # 先获取全局快照（最新）
    snapshot = fetch_eastmoney_comment_all()

    for i, ticker in enumerate(tickers, 1):
        code_short = _ts_code_to_ak(ticker)
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ")
        detail_df = fetch_stock_sentiment_detail(ticker, days)

        if detail_df is not None and not detail_df.empty:
            all_data[ticker] = detail_df
            print(f"{len(detail_df)}条")
        elif snapshot is not None and not snapshot.empty:
            # fallback: 用全局快照的单日值补全
            row = snapshot[snapshot['code'] == code_short]
            if not row.empty:
                today = datetime.now().strftime('%Y-%m-%d')
                single = pd.DataFrame({
                    'composite_score': [row['composite_score'].values[0]],
                    'attention_index': [row['attention_index'].values[0]],
                    'inst_participation': [row['inst_participation'].values[0]],
                }, index=pd.DatetimeIndex([today], name='trade_date'))
                all_data[ticker] = single
                print("快照")
            else:
                print("无数据")
        else:
            print("无数据")

    return all_data


def build_sentiment_panel(tickers, days=20):
    """返回 MultiIndex (date, code) 格式的情感面板，可直接 merge 到 df_panel"""
    raw = fetch_all_sentiment_features(tickers, days)
    records = []
    for ticker, df in raw.items():
        if df.empty:
            continue
        for date, row in df.iterrows():
            date = pd.Timestamp(date)
            records.append({
                'date': date,
                'code': ticker,
                **{f'sentiment_{k}': v for k, v in row.to_dict().items()}
            })
    if not records:
        return pd.DataFrame()
    panel = pd.DataFrame(records)
    sent_cols = [c for c in panel.columns if c.startswith('sentiment_')]
    panel = panel.set_index(['date', 'code']).sort_index()
    panel = panel[sent_cols]
    return panel
