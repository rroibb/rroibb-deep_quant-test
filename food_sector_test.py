"""
食品饮料板块 策略测试
数据获取 -> 训练 -> 回测 -> 可视化 -> 飞书推送
"""
import sys, os, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
from trainer import train_xgboost_models, prepare_dl_panel, train_model
from backtest import DeepQuantBacktester
from feishu_pusher import FeishuPusher
from visualization import plot_all
from datetime import datetime
import torch
from config import DEVICE

# ==================== 食品饮料板块 股票池 ====================
FOOD_SECTOR_MAP = {
    # 白酒
    '600519.SH': '贵州茅台', '000858.SZ': '五粮液', '002304.SZ': '洋河股份',
    '000568.SZ': '泸州老窖', '600809.SH': '山西汾酒', '000860.SZ': '顺鑫农业',
    # 啤酒
    '600600.SH': '青岛啤酒', '600132.SH': '重庆啤酒', '000729.SZ': '燕京啤酒',
    # 乳制品
    '600887.SH': '伊利股份', '600882.SH': '妙可蓝多',
    # 调味品
    '603288.SH': '海天味业', '603027.SH': '千禾味业', '002507.SZ': '涪陵榨菜',
    # 肉制品
    '000895.SZ': '双汇发展', '002714.SZ': '牧原股份',
    # 休闲食品
    '002557.SZ': '洽洽食品', '603345.SH': '安井食品', '300783.SZ': '三只松鼠',
    '002847.SZ': '盐津铺子', '603517.SH': '绝味食品',
    # 保健品/酵母
    '300146.SZ': '汤臣倍健', '600298.SH': '安琪酵母',
}

FOOD_POOL = list(FOOD_SECTOR_MAP.keys())
OUTPUT_DIR = os.path.join(BASE, 'output')
FOOD_CACHE = os.path.join(BASE, 'a_stock_cache')

def fetch_food_data():
    """从 tushare 获取食品板块数据"""
    print("=" * 60)
    print("  [0/6] 获取食品饮料板块数据...")
    print("=" * 60)

    os.makedirs(FOOD_CACHE, exist_ok=True)

    existing = set()
    for f in os.listdir(FOOD_CACHE):
        if f.endswith('.parquet'):
            existing.add(f.split('_')[0])

    to_fetch = [t for t in FOOD_POOL if t not in existing]
    if not to_fetch:
        print(f"  所有 {len(FOOD_POOL)} 只股票已缓存，跳过下载")
        return True

    print(f"  需下载: {len(to_fetch)}/{len(FOOD_POOL)} 只 (已缓存: {len(existing)})")

    try:
        import tushare as ts
        ts.set_token('61e150819162b1cfc17b7ffa16607391e075b231d4f2fb5733b59868')
        pro = ts.pro_api()
    except Exception as e:
        print(f"  tushare连接失败: {e}")
        return False

    success = 0
    for ticker in to_fetch:
        try:
            name = FOOD_SECTOR_MAP.get(ticker, '')
            print(f"  下载 {ticker} ({name})...", end=' ')
            df = pro.daily(ts_code=ticker, start_date='20210622', end_date='20260622')
            if df is None or len(df) < 100:
                print(f"数据不足({len(df) if df is not None else 0}行)")
                continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]
            # tushare pro.daily 字段映射
            rename_map = {'vol': 'volume', 'pct_chg': 'pct_change'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            if 'pre_close' not in df.columns:
                df['pre_close'] = df['close'].shift(1)
            if 'trade_date' not in df.columns:
                df['trade_date'] = pd.to_datetime(df.index) if 'date' not in df.columns else pd.to_datetime(df['date'])
            else:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.loc[df['pre_close'].isna(), 'pre_close'] = df.loc[df['pre_close'].isna(), 'close']
            fname = os.path.join(FOOD_CACHE, f'{ticker}_2021-06-22_2026-06-22.parquet')
            df.to_parquet(fname, index=False)
            success += 1
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"FAIL: {e}")

    print(f"  成功: {success}/{len(to_fetch)}")
    return success > 0


def run_food_pipeline():
    print("=" * 70)
    print("  食品饮料板块 量化策略测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {DEVICE}")
    print(f"  股票池: {len(FOOD_POOL)} 只食品饮料股")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 0: 获取数据
    if not fetch_food_data():
        return

    # Step 1: 加载数据
    print(f"\n{'=' * 60}")
    print("  [1/6] 加载数据...")
    print(f"{'=' * 60}")
    loaded = {}
    for fname in os.listdir(FOOD_CACHE):
        if not fname.endswith('.parquet'):
            continue
        ticker = fname.split('_')[0]
        if ticker not in FOOD_POOL:
            continue
        try:
            df = pd.read_parquet(os.path.join(FOOD_CACHE, fname))
            if len(df) > 60:
                df = engineer_features(df)
                df['code'] = ticker
                loaded[ticker] = df
        except Exception as e:
            print(f"  [跳过] {ticker}: {e}")

    print(f"  加载: {len(loaded)}/{len(FOOD_POOL)} 只股票")

    df_list = []
    for tick, df in loaded.items():
        df = df.copy()
        if 'trade_date' in df.columns:
            df = df.set_index('trade_date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        temp = df.copy().reset_index()
        temp['code'] = tick
        df_list.append(temp)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"  面板: {panel.shape}")

    # 构建等权基准
    all_returns = []
    for ticker, df in loaded.items():
        df_temp = df.copy()
        if 'trade_date' in df_temp.columns:
            df_temp = df_temp.set_index('trade_date')
        df_temp.index = pd.to_datetime(df_temp.index)
        ret = df_temp['close'].pct_change().rename(ticker)
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
    print(f"  基准: {market_df['close'].iloc[-1]/market_df['close'].iloc[0]-1:+.2%}")

    # Step 2-4: 训练
    print(f"\n{'=' * 60}")
    print("  [2/6] 训练深度学习模型...")
    print(f"{'=' * 60}")

    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)
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

    train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
    val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
    train_ds = SequenceDataset(train_p, 30, TECHNICAL_FEATURES_DL[:12])
    val_ds = SequenceDataset(val_p, 30, TECHNICAL_FEATURES_DL[:12])

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  [错误] 数据集为空")
        return

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    dl_models = {}
    print("  训练 LSTM...")
    lstm = LSTMStockPredictor(input_size=12, hidden_size=64, num_layers=1, bidirectional=False).to(DEVICE)
    lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=5, model_name='LSTM_Food')
    dl_models['lstm'] = lstm

    print("  训练 Transformer...")
    trans = TransformerStockPredictor(input_size=12, d_model=64, nhead=2, num_encoder_layers=2).to(DEVICE)
    trans, _, _ = train_model(trans, train_loader, val_loader, epochs=5, model_name='Transformer_Food')
    dl_models['transformer'] = trans

    print("  训练 CNN...")
    cnn = CNNChartPatternRecognizer(in_channels=12).to(DEVICE)
    cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=5, model_name='CNN_Food')
    dl_models['cnn'] = cnn

    print(f"\n{'=' * 60}")
    print("  [3/6] 训练 XGBoost...")
    print(f"{'=' * 60}")
    xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

    print("  [4/6] 加载辅助模型...")
    from models.fusion import MultiModalFusionModel
    all_models = {**dl_models, 'fusion': fusion_model}

    # Step 5: 回测
    print(f"\n{'=' * 60}")
    print("  [5/6] 回测对比...")
    print(f"{'=' * 60}")
    val_start, val_end = val_dates[0], val_dates[-1]
    panel_subset = panel.loc[val_start:val_end]
    market_subset = market_df.loc[val_start:val_end]
    print(f"  区间: {val_start.date()} ~ {val_end.date()}, {len(market_subset)}天")

    bt = DeepQuantBacktester(panel_subset, market_subset, all_models, scaler,
                             xgb_models=xgb_models, xgb_scalers=xgb_scalers)

    results = {}
    for mode_name, use_f, use_x, use_d in [
        ('Multimodal Fusion (DL+XGB+NLP+LLM)', True, True, True),
        ('Deep Learning Only (LSTM+Transformer+CNN)', False, False, True),
        ('XGBoost Only', False, True, False),
    ]:
        print(f"\n  {mode_name}")
        daily = bt.run(use_fusion=use_f, use_xgb=use_x, use_dl=use_d)
        results[mode_name] = daily

    # Step 6: 可视化 + 保存
    chart_paths = plot_all(results, OUTPUT_DIR)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, daily in results.items():
        fname = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'n')
        daily.to_csv(os.path.join(OUTPUT_DIR, f'food_{fname}_{ts_str}.csv'), encoding='utf-8-sig')

    # 打印汇总
    bench = next(iter(results.values()))['Cum_Benchmark']
    print(f"\n{'=' * 60}")
    print(f"  [食品饮料板块] 回测结果")
    print(f"{'=' * 60}")
    print(f"  基准: {bench.iloc[-1]*100:.2f}%")
    for name, daily in results.items():
        print(f"  {name}: {daily['Cum_Strategy'].iloc[-1]*100:.2f}%")

    # Step 7: 飞书推送
    print(f"\n{'=' * 60}")
    print("  [6/6] 推送到飞书...")
    print(f"{'=' * 60}")

    pusher = FeishuPusher(
        webhook_url=os.environ.get('FEISHU_WEBHOOK_URL', ''),
        secret=os.environ.get('FEISHU_WEBHOOK_SECRET', '')
    )
    period = f"{val_start.date()} ~ {val_end.date()} ({len(market_subset)}天)"

    pusher.send_backtest_report(
        results=results,
        benchmark_name='食品饮料等权指数',
        pool_name='食品饮料23股',
        period=period,
        generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )

    print(f"\n{'=' * 70}")
    print("  食品饮料板块 测试完成!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run_food_pipeline()