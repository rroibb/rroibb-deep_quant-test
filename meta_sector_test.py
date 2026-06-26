"""
稀土+钨+有色金属板块 策略测试
"""
import sys, os, numpy as np, pandas as pd, warnings
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

# ==================== 稀土+钨+有色 股票池 ====================
METAL_SECTOR_MAP = {
    # 稀土
    '600111.SH': '北方稀土', '600259.SH': '广晟有色', '600392.SH': '盛和资源',
    '000831.SZ': '中国稀土',
    # 钨
    '000657.SZ': '中钨高新', '002842.SZ': '翔鹭钨业', '600549.SH': '厦门钨业',
    '002378.SZ': '章源钨业',
    # 铜
    '000630.SZ': '铜陵有色', '600362.SH': '江西铜业', '601899.SH': '紫金矿业',
    # 黄金
    '600489.SH': '中金黄金', '600547.SH': '山东黄金', '600988.SH': '赤峰黄金',
    # 钴/镍/锡
    '000960.SZ': '锡业股份', '603799.SH': '华友钴业', '603993.SH': '洛阳钼业',
    # 铝
    '601600.SH': '中国铝业',
    # 锂
    '002460.SZ': '赣锋锂业', '002466.SZ': '天齐锂业',
    # 白银
    '000603.SZ': '盛达资源',
}

METAL_POOL = list(METAL_SECTOR_MAP.keys())
OUTPUT_DIR = os.path.join(BASE, 'output')
METAL_CACHE = os.path.join(BASE, 'a_stock_cache')


def fetch_metal_data():
    print("=" * 60)
    print("  [0/6] 获取稀土+有色金属板块数据...")
    print("=" * 60)
    os.makedirs(METAL_CACHE, exist_ok=True)
    existing = {f.split('_')[0] for f in os.listdir(METAL_CACHE) if f.endswith('.parquet')}
    to_fetch = [t for t in METAL_POOL if t not in existing]
    if not to_fetch:
        print(f"  所有 {len(METAL_POOL)} 只股票已缓存")
        return True
    print(f"  需下载: {len(to_fetch)}/{len(METAL_POOL)} 只")
    try:
        import tushare as ts
        pro = ts.pro_api()
    except Exception as e:
        print(f"  tushare 连接失败: {e}")
        return False
    success = 0
    for ticker in to_fetch:
        try:
            name = METAL_SECTOR_MAP.get(ticker, '')
            print(f"  下载 {ticker} ({name})...", end=' ')
            df = pro.daily(ts_code=ticker, start_date='20210622', end_date='20260622')
            if df is None or len(df) < 100:
                print(f"数据不足")
                continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]
            rename_map = {'vol': 'volume'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            if 'pre_close' not in df.columns:
                df['pre_close'] = df['close'].shift(1)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.loc[df['pre_close'].isna(), 'pre_close'] = df['pre_close'].fillna(df['close'])
            fname = os.path.join(METAL_CACHE, f'{ticker}_2021-06-22_2026-06-22.parquet')
            df.to_parquet(fname, index=False)
            success += 1
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"FAIL: {e}")
    print(f"  成功: {success}/{len(to_fetch)}")
    return True


def run_metal_pipeline():
    print("=" * 70)
    print("  稀土+钨+有色金属 量化策略测试")
    print(f"  时间: {datetime.now():%Y-%m-%d %H:%M:%S}  设备: {DEVICE}")
    print(f"  股票池: {len(METAL_POOL)} 只 (稀土{4}·钨{4}·铜{3}·金{3}·钴锡{3}·锂{2}·铝{1}·银{1})")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fetch_metal_data()

    # 加载
    print("\n" + "=" * 60)
    print("  [1/6] 加载数据 + 构建基准...")
    print("=" * 60)
    loaded = {}
    for fname in os.listdir(METAL_CACHE):
        if not fname.endswith('.parquet'): continue
        ticker = fname.split('_')[0]
        if ticker not in METAL_POOL: continue
        try:
            df = pd.read_parquet(os.path.join(METAL_CACHE, fname))
            if len(df) > 60:
                df = engineer_features(df)
                df['code'] = ticker
                loaded[ticker] = df
        except Exception as e:
            print(f"  [跳过] {ticker}: {e}")
    print(f"  加载: {len(loaded)}/{len(METAL_POOL)} 只股票")

    df_list = []
    for tick, df in loaded.items():
        temp = df.copy().set_index('trade_date') if 'trade_date' in df.columns else df.copy()
        temp.index = pd.to_datetime(temp.index)
        temp = temp.sort_index().reset_index()
        temp['code'] = tick
        df_list.append(temp)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"  面板: {panel.shape}")

    # 等权基准
    all_returns = []
    for ticker, df in loaded.items():
        df_t = df.copy()
        if 'trade_date' in df_t.columns:
            df_t = df_t.set_index('trade_date')
        df_t.index = pd.to_datetime(df_t.index)
        all_returns.append(df_t['close'].pct_change().rename(ticker))
    ret_panel = pd.concat(all_returns, axis=1)
    eq_ret = ret_panel.mean(axis=1)
    market_df = pd.DataFrame(index=eq_ret.index)
    market_df['close'] = (1 + eq_ret).cumprod()
    market_df['Simple_Return'] = eq_ret
    market_df['open'] = market_df['high'] = market_df['low'] = market_df['close']
    market_df['pre_close'] = market_df['close'].shift(1).fillna(market_df['close'].iloc[0])
    market_df['volume'] = market_df['amount'] = 0
    print(f"  基准累计: {market_df['close'].iloc[-1]/market_df['close'].iloc[0]-1:+.2%}")

    # 训练
    print("\n" + "=" * 60)
    print("  [2/6] 训练深度学习模型...")
    print("=" * 60)
    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)
    panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)
    all_dates = sorted(panel_dl.index.get_level_values(0).unique())
    recent_dates = all_dates[-500:]
    sp = int(len(recent_dates)*0.8)
    train_dates, val_dates = recent_dates[:sp], recent_dates[sp:]
    print(f"  训练: {train_dates[0].date()}~{train_dates[-1].date()} ({len(train_dates)}天)")
    print(f"  验证: {val_dates[0].date()}~{val_dates[-1].date()} ({len(val_dates)}天)")

    from torch.utils.data import DataLoader
    from data_layer import SequenceDataset
    from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer

    train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
    val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
    train_ds = SequenceDataset(train_p, 30, TECHNICAL_FEATURES_DL[:12])
    val_ds = SequenceDataset(val_p, 30, TECHNICAL_FEATURES_DL[:12])
    train_loader = DataLoader(train_ds, 128, shuffle=True)
    val_loader = DataLoader(val_ds, 128, shuffle=False)

    dl_models = {}
    for name, model_cls in [('LSTM', LSTMStockPredictor), ('Transformer', TransformerStockPredictor), ('CNN', CNNChartPatternRecognizer)]:
        print(f"  训练 {name}...")
        if name == 'LSTM':
            m = model_cls(12, 64, 1, 0).to(DEVICE)
        elif name == 'Transformer':
            m = model_cls(12, 64, 2, 2).to(DEVICE)
        else:
            m = model_cls(12).to(DEVICE)
        m, _, _ = train_model(m, train_loader, val_loader, 5, model_name=f'{name}_Metal')
        dl_models[name.lower()] = m

    print(f"\n{'='*60}\n  [3/6] 训练 XGBoost...\n{'='*60}")
    xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

    from models.fusion import MultiModalFusionModel
    from models.nlp_sentiment import NLPSentimentAnalyzer
    from models.llm_analyzer import LLMAnalyzer
    all_models = {**dl_models, 'fusion': MultiModalFusionModel(),
                  'nlp': NLPSentimentAnalyzer(), 'llm': LLMAnalyzer(use_mock=True)}

    # 回测
    print(f"\n{'='*60}\n  [5/6] 回测对比...\n{'='*60}")
    v0, v1 = val_dates[0], val_dates[-1]
    panel_subset = panel.loc[v0:v1]
    market_subset = market_df.loc[v0:v1]
    print(f"  区间: {v0.date()}~{v1.date()}, {len(market_subset)}天")

    bt = DeepQuantBacktester(panel_subset, market_subset, all_models, scaler,
                             xgb_models=xgb_models, xgb_scalers=xgb_scalers)
    results = {}
    for mode_name, uf, ux, ud in [
        ('Multimodal Fusion (DL+XGB+NLP+LLM)', True, True, True),
        ('Deep Learning Only (LSTM+Transformer+CNN)', False, False, True),
        ('XGBoost Only', False, True, False),
    ]:
        print(f"\n  {mode_name}")
        daily = bt.run(use_fusion=uf, use_xgb=ux, use_dl=ud)
        results[mode_name] = daily

    # 可视化 + 保存
    chart_paths = plot_all(results, OUTPUT_DIR)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, daily in results.items():
        fname = name.replace(' ','_').replace('(','').replace(')','').replace('+','n')
        daily.to_csv(os.path.join(OUTPUT_DIR, f'metal_{fname}_{ts}.csv'), encoding='utf-8-sig')

    bench = next(iter(results.values()))['Cum_Benchmark']
    print(f"\n{'='*60}\n  [稀土+有色金属] 回测结果\n{'='*60}")
    print(f"  基准: {bench.iloc[-1]*100:.2f}%")
    for name, daily in results.items():
        print(f"  {name}: {daily['Cum_Strategy'].iloc[-1]*100:.2f}%")

    # 飞书推送
    print(f"\n{'='*60}\n  [6/6] 推送飞书...\n{'='*60}")
    pusher = FeishuPusher(
        webhook_url=os.environ.get('FEISHU_WEBHOOK_URL', ''),
        secret=os.environ.get('FEISHU_WEBHOOK_SECRET', ''))
    period = f"{v0.date()}~{v1.date()} ({len(market_subset)}天)"
    pusher.send_backtest_report(results=results,
        benchmark_name='稀土有色金属等权指数',
        pool_name='稀土有色20股',
        period=period, generated=datetime.now().strftime('%Y-%m-%d %H:%M'))

    print(f"\n{'='*70}\n  稀土+有色金属 测试完成!\n{'='*70}")


if __name__ == '__main__':
    run_metal_pipeline()