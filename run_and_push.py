"""
deep_quant 全流程执行 + 飞书推送
一次运行：数据加载 -> 模型训练 -> 回测对比 -> 可视化 -> 飞书推送
"""
import sys, os, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import STOCK_POOL, OUTPUT_DIR, DEVICE
from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
from trainer import train_xgboost_models, prepare_dl_panel, train_model
from backtest import DeepQuantBacktester
from feishu_pusher import FeishuPusher
from visualization import plot_all, plot_dashboard
from datetime import datetime

CACHE_DIR = os.path.join(BASE, 'a_stock_cache')


def load_data():
    """加载缓存数据并构建面板"""
    print("=" * 60)
    print("  [1/6] 加载数据...")
    print("=" * 60)

    loaded = {}
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.parquet'):
            continue
        ticker = fname.split('_')[0]
        if ticker not in STOCK_POOL:
            continue
        try:
            df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
            if len(df) > 60:
                df = engineer_features(df)
                df['code'] = ticker
                loaded[ticker] = df
        except Exception as e:
            print(f"  [跳过] {ticker}: {e}")

    print(f"  加载: {len(loaded)}/{len(STOCK_POOL)} 只股票")

    df_list = []
    for tick, df in loaded.items():
        temp = df.copy().reset_index()
        temp['code'] = tick
        df_list.append(temp)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"  面板: {panel.shape}")

    # 构建等权市场基准
    all_returns = []
    for ticker, df in loaded.items():
        ret = df['close'].pct_change().rename(ticker)
        all_returns.append(ret)
    ret_panel = pd.concat(all_returns, axis=1)
    equal_weight_ret = ret_panel.mean(axis=1)
    market_df = pd.DataFrame(index=equal_weight_ret.index)
    market_df['close'] = (1 + equal_weight_ret).cumprod()
    market_df['open'] = market_df['close']
    market_df['high'] = market_df['close']
    market_df['low'] = market_df['close']
    market_df['pre_close'] = market_df['close'].shift(1).fillna(market_df['close'].iloc[0])
    market_df['volume'] = 0
    market_df['amount'] = 0
    market_df['Simple_Return'] = equal_weight_ret
    print(f"  基准指数: {market_df['close'].iloc[-1] / market_df['close'].iloc[0] - 1:+.2%}")

    return panel, loaded, market_df


def train_models(panel, market_df):
    """训练所有模型 (DL + XGBoost)"""
    print(f"\n{'=' * 60}")
    print("  [2/6] 训练深度学习模型...")
    print(f"{'=' * 60}")

    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1
    )

    panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)

    all_dates = sorted(panel_dl.index.get_level_values(0).unique())
    recent_dates = all_dates[-500:]
    split_idx = int(len(recent_dates) * 0.8)
    train_dates = recent_dates[:split_idx]
    val_dates = recent_dates[split_idx:]
    print(f"  训练: {train_dates[0].date()} ~ {train_dates[-1].date()} ({len(train_dates)}天)")
    print(f"  验证: {val_dates[0].date()} ~ {val_dates[-1].date()} ({len(val_dates)}天)")

    from torch.utils.data import DataLoader
    from data_layer import SequenceDataset
    from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
    import torch

    train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
    val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
    train_ds = SequenceDataset(train_p, 30, TECHNICAL_FEATURES_DL[:12])
    val_ds = SequenceDataset(val_p, 30, TECHNICAL_FEATURES_DL[:12])

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  [错误] 数据集为空")
        return None, None, None, None, None

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    dl_models = {}

    print("  训练 LSTM...")
    lstm = LSTMStockPredictor(input_size=12, hidden_size=64, num_layers=1, bidirectional=False).to(DEVICE)
    lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=5, model_name='LSTM')
    dl_models['lstm'] = lstm

    print("  训练 Transformer...")
    trans = TransformerStockPredictor(input_size=12, d_model=64, nhead=2, num_encoder_layers=2).to(DEVICE)
    trans, _, _ = train_model(trans, train_loader, val_loader, epochs=5, model_name='Transformer')
    dl_models['transformer'] = trans

    print("  训练 CNN...")
    cnn = CNNChartPatternRecognizer(in_channels=12).to(DEVICE)
    cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=5, model_name='CNN')
    dl_models['cnn'] = cnn

    print(f"\n{'=' * 60}")
    print("  [3/6] 训练 XGBoost...")
    print(f"{'=' * 60}")
    xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

    print("  [4/6] 加载融合模型...")
    from models.fusion import MultiModalFusionModel
    fusion_model = MultiModalFusionModel()

    return dl_models, scaler, xgb_models, xgb_scalers, fusion_model, val_dates


def run_backtest(panel, market_df, all_models, scaler, xgb_models, xgb_scalers, val_dates):
    """运行3模式回测对比"""
    print(f"\n{'=' * 60}")
    print("  [5/6] 回测对比...")
    print(f"{'=' * 60}")

    val_start, val_end = val_dates[0], val_dates[-1]
    all_dates = sorted(panel.index.get_level_values(0).unique())
    warmup_idx = max(0, all_dates.index(val_start) - 30)
    warmup_start = all_dates[warmup_idx]
    panel_subset = panel.loc[warmup_start:val_end]
    market_subset = market_df.loc[val_start:val_end]
    print(f"  回测区间: {val_start.date()} ~ {val_end.date()}, {len(market_subset)}天")
    print(f"  DL预热数据: {warmup_start.date()} ~ {val_start.date()}, {(all_dates.index(val_start)-warmup_idx)}天")

    bt = DeepQuantBacktester(panel_subset, market_subset, all_models, scaler,
                             xgb_models=xgb_models, xgb_scalers=xgb_scalers,
                             backtest_start=val_start)

    results = {}
    for mode_name, use_f, use_x, use_d in [
        ('Multimodal Fusion (DL+XGB)', True, True, True),
        ('Deep Learning Only (LSTM+Transformer+CNN)', False, False, True),
        ('XGBoost Only', False, True, False),
    ]:
        print(f"\n  {mode_name}")
        daily = bt.run(use_fusion=use_f, use_xgb=use_x, use_dl=use_d)
        results[mode_name] = daily

    return results


def push_to_feishu(results, chart_paths, webhook_url, secret='', val_dates=None):
    """推送中文卡片到飞书"""
    print(f"\n{'=' * 60}")
    print("  [6/6] 推送中文回测报告到飞书...")
    print(f"{'=' * 60}")

    pusher = FeishuPusher(webhook_url=webhook_url, secret=secret)

    period = ''
    if val_dates and len(val_dates) >= 2:
        period = f"{val_dates[0].date()} ~ {val_dates[-1].date()} (100个交易日)"

    pusher.send_backtest_report(
        results=results,
        pool_name='电子科技50股',
        benchmark_name='电子科技50等权指数',
        period=period,
        generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )

    print(f"  飞书推送完成!")


def main():
    print("=" * 70)
    print("  deep_quant 全流程量化策略执行")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    panel, loaded, market_df = load_data()
    if len(loaded) < 30:
        print(f"标的不足: {len(loaded)}")
        return

    # 2-4. 训练模型
    ret = train_models(panel, market_df)
    if ret[0] is None:
        return
    dl_models, scaler, xgb_models, xgb_scalers, fusion_model, val_dates = ret
    all_models = {**dl_models, 'fusion': fusion_model}

    # 5. 回测
    results = run_backtest(panel, market_df, all_models, scaler, xgb_models, xgb_scalers, val_dates)

    # 6. 可视化
    chart_paths = plot_all(results, OUTPUT_DIR)

    # 7. 保存数据
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, daily in results.items():
        fname = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'n')
        daily.to_csv(os.path.join(OUTPUT_DIR, f'{fname}_{ts}.csv'), encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    bench = next(iter(results.values()))['Cum_Benchmark']
    print(f"  基准累计收益: {bench.iloc[-1]*100:.2f}%")
    for name, daily in results.items():
        print(f"  {name}: {daily['Cum_Strategy'].iloc[-1]*100:.2f}%")
    print(f"  图表: {len(chart_paths)} 张")
    print(f"  数据已保存至: {OUTPUT_DIR}")

    # 8. 飞书推送 (中文卡片)
    webhook_url = os.environ.get('FEISHU_WEBHOOK_URL', '')
    webhook_secret = os.environ.get('FEISHU_WEBHOOK_SECRET', '')
    if webhook_url:
        push_to_feishu(results, chart_paths, webhook_url, webhook_secret, val_dates)
    else:
        print(f"\n  [飞书] 未设置 FEISHU_WEBHOOK_URL 环境变量，跳过推送")

    print(f"\n{'=' * 70}")
    print("  全流程执行完成!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()