"""
新能源+电动车产业链 策略测试 (~30只)
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

# ==================== 新能源+电动车 股票池 ====================
NEV_SECTOR_MAP = {
    # 整车
    '002594.SZ': '比亚迪',    '000625.SZ': '长安汽车',  '601633.SH': '长城汽车',
    '600104.SH': '上汽集团',  '601238.SH': '广汽集团',  '600733.SH': '北汽蓝谷',
    '000800.SZ': '一汽解放',  '601127.SH': '赛力斯',
    # 电池
    '300750.SZ': '宁德时代',  '300014.SZ': '亿纬锂能',  '002074.SZ': '国轩高科',
    '300207.SZ': '欣旺达',    '300438.SZ': '鹏辉能源',
    # 电池材料 (锂/钴/镍/电解液/隔膜)
    '002709.SZ': '天赐材料',  '300769.SZ': '德方纳米',  '300073.SZ': '当升科技',
    '603799.SH': '华友钴业',  '603659.SH': '璞泰来',    '688005.SH': '容百科技',
    '002460.SZ': '赣锋锂业',  '002466.SZ': '天齐锂业',
    # 光伏
    '601012.SH': '隆基绿能',  '600438.SH': '通威股份',  '002459.SZ': '晶澳科技',
    '300274.SZ': '阳光电源',  '688599.SH': '天合光能',  '603806.SH': '福斯特',
    # 风电
    '002202.SZ': '金风科技',  '601615.SH': '明阳智能',
    # 电驱/充电
    '300124.SZ': '汇川技术',  '002850.SZ': '科达利',    '300001.SZ': '特锐德',
}

NEV_POOL = list(NEV_SECTOR_MAP.keys())
OUTPUT_DIR = os.path.join(BASE, 'output')
NEV_CACHE = os.path.join(BASE, 'a_stock_cache')


def fetch_nev_data():
    print("=" * 60)
    print("  [0/6] 获取新能源+电动车板块数据...")
    print("=" * 60)
    os.makedirs(NEV_CACHE, exist_ok=True)
    existing = {f.split('_')[0] for f in os.listdir(NEV_CACHE) if f.endswith('.parquet')}
    to_fetch = [t for t in NEV_POOL if t not in existing]
    if not to_fetch:
        print(f"  所有 {len(NEV_POOL)} 只已缓存")
        return True
    print(f"  需下载: {len(to_fetch)}/{len(NEV_POOL)} 只")
    try:
        import tushare as ts
        pro = ts.pro_api()
    except Exception as e:
        print(f"  tushare 连接失败: {e}")
        return False
    success = 0
    for ticker in to_fetch:
        try:
            name = NEV_SECTOR_MAP.get(ticker, '')
            print(f"  下载 {ticker} ({name})...", end=' ')
            df = pro.daily(ts_code=ticker, start_date='20210622', end_date='20260622')
            if df is None or len(df) < 100:
                print(f"数据不足"); continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'vol': 'volume'})
            if 'pre_close' not in df.columns:
                df['pre_close'] = df['close'].shift(1)
            df.loc[df['pre_close'].isna(), 'pre_close'] = df['close']
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.to_parquet(os.path.join(NEV_CACHE, f'{ticker}_2021-06-22_2026-06-22.parquet'), index=False)
            success += 1; print(f"OK ({len(df)})")
        except Exception as e:
            print(f"FAIL: {e}")
    print(f"  成功: {success}/{len(to_fetch)}")
    return True


def run_nev_pipeline():
    print("=" * 70)
    print("  新能源+电动车 量化策略测试")
    print(f"  时间: {datetime.now():%Y-%m-%d %H:%M:%S}  设备: {DEVICE}")
    print(f"  股票池: {len(NEV_POOL)}只 (整车{8}·电池{5}·材料{8}·光伏{6}·风电{2}·电驱{3})")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fetch_nev_data()

    # 加载
    print(f"\n{'='*60}\n  [1/6] 加载数据+构建基准...\n{'='*60}")
    loaded = {}
    for fname in os.listdir(NEV_CACHE):
        if not fname.endswith('.parquet'): continue
        ticker = fname.split('_')[0]
        if ticker not in NEV_POOL: continue
        try:
            df = pd.read_parquet(os.path.join(NEV_CACHE, fname))
            if len(df) > 60:
                df = engineer_features(df)
                df['code'] = ticker
                loaded[ticker] = df
        except Exception as e:
            print(f"  [跳过] {ticker}: {e}")
    print(f"  加载: {len(loaded)}/{len(NEV_POOL)}只")

    df_list = []
    for tick, df in loaded.items():
        temp = df.copy()
        if 'trade_date' in temp.columns:
            temp = temp.set_index('trade_date')
        temp.index = pd.to_datetime(temp.index)
        temp = temp.sort_index().reset_index()
        temp['code'] = tick; df_list.append(temp)

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
    eq_ret = pd.concat(all_returns, axis=1).mean(axis=1)
    market_df = pd.DataFrame(index=eq_ret.index)
    market_df['close'] = (1 + eq_ret).cumprod()
    market_df['Simple_Return'] = eq_ret
    market_df['open'] = market_df['high'] = market_df['low'] = market_df['close']
    market_df['pre_close'] = market_df['close'].shift(1).fillna(market_df['close'].iloc[0])
    market_df['volume'] = market_df['amount'] = 0
    print(f"  基准累计: {market_df['close'].iloc[-1]/market_df['close'].iloc[0]-1:+.2%}")

    # 训练DL
    print(f"\n{'='*60}\n  [2/6] 训练深度学习...\n{'='*60}")
    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)
    panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)
    all_dates = sorted(panel_dl.index.get_level_values(0).unique())
    recent = all_dates[-500:]
    sp = int(len(recent)*0.8)
    train_dates, val_dates = recent[:sp], recent[sp:]
    print(f"  训练: {train_dates[0].date()}~{train_dates[-1].date()} ({len(train_dates)}天)")
    print(f"  验证: {val_dates[0].date()}~{val_dates[-1].date()} ({len(val_dates)}天)")

    from torch.utils.data import DataLoader
    from data_layer import SequenceDataset
    from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer

    tp = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
    vp = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
    tds = SequenceDataset(tp, 30, TECHNICAL_FEATURES_DL[:12])
    vds = SequenceDataset(vp, 30, TECHNICAL_FEATURES_DL[:12])
    tr, vl = DataLoader(tds, 128, shuffle=True), DataLoader(vds, 128, shuffle=False)

    dl_models = {}
    for nm, cls in [('LSTM',LSTMStockPredictor),('Transformer',TransformerStockPredictor),('CNN',CNNChartPatternRecognizer)]:
        print(f"  训练 {nm}...")
        m = cls(12, 64, 1, 0).to(DEVICE) if nm=='LSTM' else (cls(12, 64, 2, 2).to(DEVICE) if nm=='Transformer' else cls(12).to(DEVICE))
        m, _, _ = train_model(m, tr, vl, 5, model_name=f'{nm}_NEV'); dl_models[nm.lower()] = m

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
    ps = panel.loc[v0:v1]; ms = market_df.loc[v0:v1]
    print(f"  区间: {v0.date()}~{v1.date()}, {len(ms)}天")

    bt = DeepQuantBacktester(ps, ms, all_models, scaler, xgb_models=xgb_models, xgb_scalers=xgb_scalers)
    results = {}
    for mode_name, uf, ux, ud in [
        ('Multimodal Fusion (DL+XGB+NLP+LLM)', True, True, True),
        ('Deep Learning Only (LSTM+Transformer+CNN)', False, False, True),
        ('XGBoost Only', False, True, False),
    ]:
        print(f"\n  {mode_name}")
        results[mode_name] = bt.run(use_fusion=uf, use_xgb=ux, use_dl=ud)

    # 可视化+保存
    chart_paths = plot_all(results, OUTPUT_DIR)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, daily in results.items():
        fname = name.replace(' ','_').replace('(','').replace(')','').replace('+','n')
        daily.to_csv(os.path.join(OUTPUT_DIR, f'nev_{fname}_{ts}.csv'), encoding='utf-8-sig')

    bench = next(iter(results.values()))['Cum_Benchmark']
    print(f"\n{'='*60}\n  [新能源电动车] 回测结果\n{'='*60}")
    print(f"  基准: {bench.iloc[-1]*100:.2f}%")
    for name, daily in results.items():
        print(f"  {name}: {daily['Cum_Strategy'].iloc[-1]*100:.2f}%")

    # 飞书
    print(f"\n{'='*60}\n  [6/6] 推送飞书...\n{'='*60}")
    pusher = FeishuPusher(
        webhook_url=os.environ.get('FEISHU_WEBHOOK_URL', ''),
        secret=os.environ.get('FEISHU_WEBHOOK_SECRET', ''))
    pusher.send_backtest_report(
        results=results,
        pool_name=f'新能源电动车{len(NEV_POOL)}股',
        benchmark_name='新能源电动车等权指数',
        period=f"{v0.date()}~{v1.date()} ({len(ms)}天)",
        generated=datetime.now().strftime('%Y-%m-%d %H:%M'))

    print(f"\n{'='*70}\n  新能源电动车测试完成!\n{'='*70}")


if __name__ == '__main__':
    run_nev_pipeline()