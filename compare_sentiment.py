"""
情感特征效果对比（离线轻量版）
方案：用 akshare 获取 + 工程情感特征 → 训练 DL 模型 → 对比有无情感的验证 IC
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (engineer_features, merge_sentiment_features, TECHNICAL_FEATURES_DL,
                      SENTIMENT_FEATURES, TECHNICAL_FEATURES)
from trainer import prepare_dl_panel, train_model
from data_layer import SequenceDataset
from config import STOCK_POOL, BATCH_SIZE, OUTPUT_DIR
from torch.utils.data import DataLoader


def load_local_cache():
    cache_dir = './a_stock_cache'
    if not os.path.isdir(cache_dir):
        print(f"缓存目录 {cache_dir} 不存在")
        return []
    files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    if not files:
        print("无缓存数据")
        return []
    print(f"找到 {len(files)} 个缓存文件")
    return files


def build_panel_from_cache():
    stock_data = {}
    from config import SECTOR_MAP

    cache_dir = './a_stock_cache'
    print("\n从缓存 parquet 读取...")
    import glob
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
            stock_data[ticker] = df
        except Exception as e:
            pass

    print(f"有效标的: {len(stock_data)} 只")
    if len(stock_data) < 5:
        print(f"数据不足 ({len(stock_data)} < 10)，退出")
        return pd.DataFrame()

    # 特征工程
    for ticker in list(stock_data.keys()):
        stock_data[ticker] = engineer_features(stock_data[ticker])
        stock_data[ticker]['code'] = ticker

    df_list = []
    for ticker, df in stock_data.items():
        temp = df.copy().reset_index()
        temp['code'] = ticker
        df_list.append(temp)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"面板: {panel.shape}")

    return panel


def compare_ic_with_without(panel):
    """对比有无情感特征的 DL 模型验证 IC"""

    from models import LSTMStockPredictor, TransformerStockPredictor

    active_codes = panel.index.get_level_values(1).unique().tolist()
    panel_with = merge_sentiment_features(panel.copy(), active_codes)
    sent_avail = [c for c in SENTIMENT_FEATURES if c in panel_with.columns]
    panel_without = panel_with.drop(columns=sent_avail, errors='ignore')

    print(f"\n情感特征: {sent_avail}")
    print(f"DL特征维数: 含情感={len(TECHNICAL_FEATURES_DL)+len(sent_avail)}, 无情感={len(TECHNICAL_FEATURES_DL)}")

    panel_with['Future_20d_Ret'] = panel_with.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1)
    panel_without['Future_20d_Ret'] = panel_without.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1)

    results = []

    for label, p, feat_list in [
        ('w/ Sentiment', panel_with, TECHNICAL_FEATURES_DL + sent_avail),
        ('w/o Sentiment', panel_without, TECHNICAL_FEATURES_DL)
    ]:
        print(f"\n{'='*60}")
        print(f"  >> {label}  (feat_dim={len(feat_list)})")
        print(f"{'='*60}")

        p_dl, scaler = prepare_dl_panel(p, feat_list)
        all_dates = sorted(p_dl.index.get_level_values(0).unique())
        split_idx = int(len(all_dates) * 0.8)
        train_dates = all_dates[:split_idx]
        val_dates = all_dates[split_idx:]

        train_panel = p_dl[p_dl.index.get_level_values(0).isin(train_dates)]
        val_panel = p_dl[p_dl.index.get_level_values(0).isin(val_dates)]

        train_ds = SequenceDataset(train_panel, 30, feat_list)
        val_ds = SequenceDataset(val_panel, 30, feat_list)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("数据集为空，跳过")
            continue

        tr = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
        vl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

        input_dim = len(feat_list)

        for model_cls, mname in [(LSTMStockPredictor, 'LSTM'),
                                  (TransformerStockPredictor, 'Transformer')]:
            print(f"\n  训练 {mname}...")
            m = model_cls(input_size=input_dim)
            epochs = 30 if mname == 'LSTM' else 20
            m, hist, ic = train_model(m, tr, vl, epochs=epochs, model_name=f'{mname}_{label}')
            print(f"  OK {mname} val_IC = {ic:.4f}")
            results.append({'label': label, 'model': mname, 'ic': ic})

    return results


def main():
    print("=" * 60)
    print("  情感特征效果对比 (离线轻量版)")
    print("=" * 60)

    panel = build_panel_from_cache()
    if panel.empty:
        return

    results = compare_ic_with_without(panel)

    if not results:
        print("\n无对比结果")
        return

    df_r = pd.DataFrame(results)

    print(f"\n\n{'='*60}")
    print("  IC 对比汇总")
    print(f"{'='*60}")
    print(f"{'模型':<20} {'有情感':<15} {'无情感':<15} {'差值':<15}")
    print("-" * 65)
    for model_name in ['LSTM', 'Transformer']:
        row = df_r[df_r['model'] == model_name]
        if len(row) < 2:
            continue
        r_with = float(row[row['label'] == 'w/ Sentiment']['ic'].values[0])
        r_wo = float(row[row['label'] == 'w/o Sentiment']['ic'].values[0])
        diff = r_with - r_wo
        arrow = "^" if diff > 0 else "v"
        print(f"{model_name:<20} {r_with:<15.4f} {r_wo:<15.4f} {diff:+.4f} {arrow}")

    print(f"\n{'='*60}")
    avg_with = df_r[df_r['label'] == 'w/ Sentiment']['ic'].mean()
    avg_wo = df_r[df_r['label'] == 'w/o Sentiment']['ic'].mean()
    print(f"平均IC: 有情感={avg_with:.4f}  无情感={avg_wo:.4f}  diff={abs(avg_with-avg_wo):.4f} {'>' if avg_with > avg_wo else '<'}0")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
