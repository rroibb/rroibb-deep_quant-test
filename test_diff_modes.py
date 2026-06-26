"""
验证三种模式是否真正产生差异
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
BASE = 'C:\\Users\\24259\\PycharmProjects\\金融数据分析\\deep_quant'
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import STOCK_POOL, OUTPUT_DIR, DEVICE
from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
from torch.utils.data import DataLoader
from data_layer import SequenceDataset
from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
from trainer import train_model, prepare_dl_panel, train_xgboost_models
from backtest import DeepQuantBacktester
from models.fusion import MultiModalFusionModel
from models.nlp_sentiment import NLPSentimentAnalyzer
from models.llm_analyzer import LLMAnalyzer
from scipy import stats
import torch

# 1. Load data
CACHE_DIR = 'C:/Users/24259/PycharmProjects/金融数据分析/a_stock_cache'
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
    except:
        pass

print(f"Loaded {len(loaded)} stocks")

df_list = []
for tick, df in loaded.items():
    temp = df.copy().reset_index()
    temp['code'] = tick
    df_list.append(temp)
panel = pd.concat(df_list, ignore_index=True)
panel['date'] = pd.to_datetime(panel['trade_date'])
panel = panel.set_index(['date', 'code']).sort_index()

all_returns = []
for ticker, df in loaded.items():
    all_returns.append(df['close'].pct_change().rename(ticker))
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
print(f"Panel: {panel.shape}, Market: {len(market_df)} days")

panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(
    lambda x: x.shift(-20) / x - 1
)
panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)

all_dates = sorted(panel_dl.index.get_level_values(0).unique())
recent_dates = all_dates[-500:]
split_idx = int(len(recent_dates) * 0.8)
train_dates = recent_dates[:split_idx]
val_dates = recent_dates[split_idx:]

train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]

seq_len = 30
feat_cols = TECHNICAL_FEATURES_DL[:12]
train_ds = SequenceDataset(train_p, seq_len, feat_cols)
val_ds = SequenceDataset(val_p, seq_len, feat_cols)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

dl_models = {}
lstm = LSTMStockPredictor(input_size=12, hidden_size=64, num_layers=1, bidirectional=False).to(DEVICE)
lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=3, model_name='LSTM')
dl_models['lstm'] = lstm

trans = TransformerStockPredictor(input_size=12, d_model=64, nhead=2, num_encoder_layers=2).to(DEVICE)
trans, _, _ = train_model(trans, train_loader, val_loader, epochs=3, model_name='Transformer')
dl_models['transformer'] = trans

cnn = CNNChartPatternRecognizer(in_channels=12).to(DEVICE)
cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=3, model_name='CNN')
dl_models['cnn'] = cnn

xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

# 2. SINGLE DAY TEST - check predictions and rankings
print("\n" + "=" * 60)
print("=== Single Day Prediction Test ===")
print("=" * 60)

test_date = val_dates[0]
print(f"Test date: {test_date.date()}")

# --- DL predictions per stock (build fresh from val_p for test_date) ---
test_codes_in_panel = val_p.xs(test_date, level=0).index.unique() if test_date in val_p.index.get_level_values(0) else []
dl_preds = {}
if len(test_codes_in_panel) > 0:
    for code in test_codes_in_panel:
        hist = val_p.xs(code, level=1).loc[:test_date].tail(seq_len)
        if len(hist) < 5:
            continue
        seq_vals = hist[feat_cols].fillna(0).values.astype(np.float32)
        if len(seq_vals) < seq_len:
            pad = np.zeros((seq_len - len(seq_vals), seq_vals.shape[1]), dtype=np.float32)
            seq_vals = np.concatenate([pad, seq_vals], axis=0)
        x = torch.from_numpy(seq_vals[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            l = float(lstm(x).cpu().numpy().mean())
            t = float(trans(x).cpu().numpy().mean())
            c = float(cnn(x).cpu().numpy().mean())
        dl_preds[code] = {'lstm': l, 'transformer': t, 'cnn': c, 'avg': np.mean([l, t, c])}

if len(dl_preds) == 0:
    print("No DL stocks matched (val_p may have different dates)")
else:
    print(f"Stocks with DL predictions: {len(dl_preds)}")
    vals = {k: [v[k] for v in dl_preds.values()] for k in ['lstm', 'transformer', 'cnn', 'avg']}
    for name in ['lstm', 'transformer', 'cnn', 'avg']:
        v = vals[name]
        print(f"  {name:12s}: mean={np.mean(v):.4f}, std={np.std(v):.4f}")

    # Rank correlations
    print(f"\nRank correlations:")
    model_names = ['lstm', 'transformer', 'cnn', 'avg']
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            r, _ = stats.spearmanr(vals[n1], vals[n2])
            print(f"  {n1} vs {n2}: rho={r:.4f}")

    # Top 5 by each mode
    print(f"\nTop 5 stocks by each mode:")
    for mode in ['avg', 'lstm', 'transformer', 'cnn']:
        top5 = sorted(dl_preds.items(), key=lambda x: x[1][mode], reverse=True)[:5]
        codes_str = [c for c, v in top5]
        avg_top5 = set(c for c, v in sorted(dl_preds.items(), key=lambda x: x[1]['avg'], reverse=True)[:5])
        overlap = len(avg_top5 & set(codes_str))
        print(f"  {mode:12s}: {codes_str}")

    # Overlap between avg_dl and individual models
    avg_top5 = set(c for c, v in sorted(dl_preds.items(), key=lambda x: x[1]['avg'], reverse=True)[:5])
    for mode in ['lstm', 'transformer', 'cnn']:
        mode_top5 = set(c for c, v in sorted(dl_preds.items(), key=lambda x: x[1][mode], reverse=True)[:5])
        print(f"  Overlap avg_top5 vs {mode}: {len(avg_top5 & mode_top5)}/5")

# --- XGB predictions per stock ---
print(f"\nXGB predictions for same day:")
test_panel_day = panel.loc[test_date].copy()
test_codes = test_panel_day.index.tolist()
for regime_name, model in xgb_models.items():
    scaler = xgb_scalers[regime_name]
    regime_feats = {
        'trend': ['Ret_20', 'Ret_60', 'Price_MA_60_Ratio', 'MA_20_60_Cross', 'Volatility_20', 'Amount_Ratio'],
        'mean_reversion': ['Ret_5', 'Ret_10', 'RSI', 'BB_Position', 'VWAP_Dist', 'Price_Position', 'High_Low_Ratio'],
        'neutral': TECHNICAL_FEATURES[:18],
    }.get(regime_name, [])
    feats = [c for c in regime_feats if c in test_panel_day.columns]
    if len(feats) < 2:
        continue
    X = test_panel_day[feats].fillna(0)
    preds = model.predict(scaler.transform(X))
    top5 = [test_codes[i] for i in np.argsort(preds)[-5:][::-1]]
    bot5 = [test_codes[i] for i in np.argsort(preds)[:5]]
    print(f"  {regime_name:15s}: top5={top5[:3]}... bot5={bot5[:3]}...")

# 3. SIXTY-DAY BACKTEST (need >= 30 days for seq_len DL precompute)
print(f"\n" + "=" * 60)
print("=== 60-Day Backtest ===")
print("=" * 60)

win_dates = val_dates[:60]
ten_panel = panel.loc[win_dates[0]:win_dates[-1]]
ten_market = market_df.loc[win_dates[0]:win_dates[-1]]

fusion_model = MultiModalFusionModel()
nlp_model = NLPSentimentAnalyzer()
llm_model = LLMAnalyzer(use_mock=True)
all_models = {**dl_models, 'fusion': fusion_model, 'nlp': nlp_model, 'llm': llm_model}

bt = DeepQuantBacktester(ten_panel, ten_market, all_models, scaler,
                         xgb_models=xgb_models, xgb_scalers=xgb_scalers)

# Reset cached DL preds between modes, but share them for fairness
bt._cached_dl_map = None

for mode_name, use_f, use_x, use_d in [
    ('Multimodal Fusion', True, True, True),
    ('Deep Learning Only', False, False, True),
    ('XGBoost Only', False, True, False),
]:
    print(f"\n{mode_name}")
    bt._cached_dl_map = None  # force recompute with same data each time
    daily = bt.run(use_fusion=use_f, use_xgb=use_x, use_dl=use_d)
    strat = daily['Cum_Strategy'].iloc[-1]
    bench = daily['Cum_Benchmark'].iloc[-1]
    strat_ret_mean = daily['Strategy_Ret'].iloc[30:].mean() if len(daily) > 30 else daily['Strategy_Ret'].mean()
    print(f"  Strategy cumulative: {strat*100:.2f}%   Benchmark: {bench*100:.2f}%   Excess: {(strat-bench)*100:.2f}%")
    print(f"  Mean daily return (day 31+): {strat_ret_mean*100:.4f}%")
    bench_ret_col = 'Simple_Return'
    print(f"  First 3 strategy returns: {[f'{x:.4f}' for x in daily['Strategy_Ret'].iloc[:3].tolist()]}")

print("\nDone!")
