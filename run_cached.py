"""Full end-to-end pipeline using only cached parquet data (no tushare token needed)"""
import os, sys, warnings, glob
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCK_POOL, OUTPUT_DIR, DEVICE
from features import engineer_features, TECHNICAL_FEATURES_DL, merge_sentiment_features
from trainer import train_all_deep_models, train_xgboost_models
from backtest import DeepQuantBacktester


def load_cached_stocks():
    cache_dir = './a_stock_cache'
    stock_data = {}
    print("\nReading cached parquet files...")
    for ticker in STOCK_POOL:
        matches = glob.glob(os.path.join(cache_dir, f"{ticker}_*.parquet"))
        if not matches:
            continue
        try:
            df = pd.read_parquet(matches[0])
            df.index = pd.to_datetime(df.index)
            if 'vol' in df.columns:
                df = df.rename(columns={'vol': 'volume'})
            if 'pre_close' not in df.columns:
                df['pre_close'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            stock_data[ticker] = engineer_features(df)
            stock_data[ticker]['code'] = ticker
        except Exception as e:
            pass
    return stock_data


def build_panel(stock_data):
    df_list = []
    for ticker, df in stock_data.items():
        temp = df.copy().reset_index()
        temp['code'] = ticker
        df_list.append(temp)
    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    return panel


def main():
    print("=" * 60)
    print("  Full Pipeline - Cached Data Mode")
    print(f"  Device: {DEVICE}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    stock_data = load_cached_stocks()
    print(f"Loaded {len(stock_data)} stocks from cache")

    panel = build_panel(stock_data)
    print(f"Panel shape: {panel.shape}")

    active_codes = panel.index.get_level_values(1).unique().tolist()
    print(f"Active codes: {active_codes}")

    panel = merge_sentiment_features(panel, active_codes)
    print(f"After sentiment merge: {panel.shape}")

    # Synthetic market index from panel mean returns
    daily_ret = panel.groupby(level=0)['Simple_Return'].mean()
    market_df = pd.DataFrame(index=daily_ret.index)
    market_df['close'] = (1 + daily_ret).cumprod()
    print(f"Market index: {len(market_df)} days")

    print("\n" + "="*60)
    print("  Step 1: Train Deep Learning Models")
    print("="*60)
    models, scaler, input_size = train_all_deep_models(panel, market_df, seq_len=30)
    if not models:
        print("DL training failed, exiting")
        return

    print("\n" + "="*60)
    print("  Step 2: Train XGBoost Models")
    print("="*60)
    xgb_models, xgb_scalers, feat_imp = train_xgboost_models(panel, market_df)

    print("\n" + "="*60)
    print("  Step 3: Run Backtest")
    print("="*60)
    bt = DeepQuantBacktester(
        df_panel=panel,
        market_df=market_df,
        models_dict=models,
        scaler_dl=scaler,
        xgb_models=xgb_models,
        xgb_scalers=xgb_scalers,
    )
    daily_ret = bt.run()
    if daily_ret is None or daily_ret.empty:
        print("Backtest returned no results")
        return

    total_n = daily_ret['Strategy_Ret'].sum()
    ann_n = daily_ret['Strategy_Ret'].mean() * 240
    ann_v = daily_ret['Strategy_Ret'].std() * np.sqrt(240)
    sharpe = (ann_n - 0.02) / ann_v if ann_v > 1e-8 else 0
    cum_max = daily_ret['Cum_Strategy'].cummax()
    mdd = (cum_max - daily_ret['Cum_Strategy']).max()

    print(f"\n{'='*60}")
    print("  Performance Summary")
    print(f"{'='*60}")
    print(f"  Total Return (net):   {total_n*100:>8.2f}%")
    print(f"  Annualized Return:    {ann_n*100:>8.2f}%")
    print(f"  Sharpe Ratio:         {sharpe:>8.4f}")
    print(f"  Max Drawdown:         {mdd*100:>8.2f}%")
    print(f"{'='*60}")

    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
