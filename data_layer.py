import os
import time
import numpy as np
import pandas as pd
import tushare as ts
from collections import deque
from datetime import datetime
from config import TUSHARE_TOKEN, SECTOR_MAP, STOCK_POOL

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


class RateLimiter:
    def __init__(self, max_requests_per_minute=50):
        self.max_requests = max_requests_per_minute
        self.requests = deque()

    def wait_if_needed(self):
        now = time.time()
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + 60 - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                return self.wait_if_needed()
        self.requests.append(time.time())


limiter = RateLimiter()


def fetch_stock_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            limiter.wait_if_needed()
            df = pro.daily(
                ts_code=ticker,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='trade_date,open,high,low,close,vol,amount,pre_close'
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.sort_values('trade_date', ascending=True)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df.rename(columns={'vol': 'volume'}, inplace=True)
            df['pre_close'] = df['pre_close'].fillna(df['close'].shift(1))
            return df
        except Exception as e:
            if '频率' in str(e) or 'limit' in str(e).lower():
                wait_time = 65 + attempt * 15
                time.sleep(wait_time)
            else:
                return pd.DataFrame()
    return pd.DataFrame()


CACHE_DIR = './a_stock_cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_with_cache(ticker, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 60:
                return df
        except:
            try:
                os.remove(cache_file)
            except:
                pass
    df = fetch_stock_data(ticker, start_date, end_date)
    if df is not None and not df.empty and len(df) > 60:
        try:
            df.to_parquet(cache_file)
        except:
            pass
    return df


def fetch_all_stocks(start_date, end_date):
    all_data = {}
    print(f"\n获取 {len(STOCK_POOL)} 只股票数据...")
    for i, ticker in enumerate(STOCK_POOL, 1):
        print(f"  [{i}/{len(STOCK_POOL)}] {ticker} {SECTOR_MAP.get(ticker,'')}...", end=" ")
        df = fetch_with_cache(ticker, start_date, end_date)
        if not df.empty:
            all_data[ticker] = df
            print(f"{len(df)}条")
        else:
            print("跳过")
    print(f"  成功: {len(all_data)}/{len(STOCK_POOL)}")
    return all_data


def fetch_market_index(start_date, end_date):
    ticker = '000001.SH'
    print(f"获取市场指数 {ticker}...")
    return fetch_with_cache(ticker, start_date, end_date)


class SequenceDataset:
    def __init__(self, df_panel, seq_len=60, feature_cols=None, target_col='Future_20d_Ret'):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.samples = []
        self.idx_to_dt_code = []
        self._build(df_panel)

    def _build(self, df_panel):
        for code in df_panel.index.get_level_values(1).unique():
            stock_data = df_panel.xs(code, level=1).sort_index()
            values = stock_data[self.feature_cols].values
            targets = stock_data[self.target_col].values
            dates = stock_data.index
            for i in range(len(values) - self.seq_len):
                x = values[i:i + self.seq_len]
                y = targets[i + self.seq_len - 1]
                if not np.any(np.isnan(x)) and not np.isnan(y):
                    self.samples.append((x.astype(np.float32), float(y)))
                    self.idx_to_dt_code.append((dates[i + self.seq_len - 1], code))
        print(f"  构建序列数据集: {len(self.samples)} 样本 (seq_len={self.seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        import torch
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
