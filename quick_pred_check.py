"""
快速诊断: 3种模式的预测值对比（某一天）
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
BASE = 'C:\\Users\\24259\\PycharmProjects\\金融数据分析\\deep_quant'
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import STOCK_POOL, DEVICE
from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
from torch.utils.data import DataLoader
from data_layer import SequenceDataset
from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
from trainer import train_model, prepare_dl_panel, train_xgboost_models
from models.fusion import MultiModalFusionModel
import torch

# Load
CACHE_DIR = 'C:/Users/24259/PycharmProjects/金融数据分析/a_stock_cache'
loaded = {}
for fname in os.listdir(CACHE_DIR):
    if not fname.endswith('.parquet'): continue
    ticker = fname.split('_')[0]
    if ticker not in STOCK_POOL: continue
    try:
        df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
        if len(df) > 60:
            df = engineer_features(df); df['code'] = ticker; loaded[ticker] = df
    except: pass

df_list = []
for tick, df in loaded.items():
    temp = df.copy().reset_index(); temp['code'] = tick; df_list.append(temp)
panel = pd.concat(df_list, ignore_index=True)
panel['date'] = pd.to_datetime(panel['trade_date'])
panel = panel.set_index(['date', 'code']).sort_index()

all_returns = []
for ticker, df in loaded.items():
    all_returns.append(df['close'].pct_change().rename(ticker))
ret_panel = pd.concat(all_returns, axis=1)
ew_ret = ret_panel.mean(axis=1)
panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20) / x - 1)
panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)

all_dates = sorted(panel_dl.index.get_level_values(0).unique())
recent = all_dates[-500:]
sp = int(len(recent)*0.8)
train_dates = recent[:sp]; val_dates = recent[sp:]
train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
feat_cols = TECHNICAL_FEATURES_DL[:12]

train_ds = SequenceDataset(train_p, 30, feat_cols)
val_ds = SequenceDataset(val_p, 30, feat_cols)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

dl_models = {}
lstm = LSTMStockPredictor(12, 64, 1, False).to(DEVICE)
lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=2, model_name='LSTM'); dl_models['lstm'] = lstm
trans = TransformerStockPredictor(12, 64, 2, 2).to(DEVICE)
trans, _, _ = train_model(trans, train_loader, val_loader, epochs=2, model_name='Transformer'); dl_models['transformer'] = trans
cnn = CNNChartPatternRecognizer(12).to(DEVICE)
cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=2, model_name='CNN'); dl_models['cnn'] = cnn

market_df = pd.DataFrame(index=ew_ret.index)
market_df['close'] = (1 + ew_ret).cumprod()
market_df['open'] = market_df['close']; market_df['high'] = market_df['close']
market_df['low'] = market_df['close']; market_df['pre_close'] = market_df['close'].shift(1).fillna(1)
market_df['volume'] = 0; market_df['amount'] = 0
xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

# Pick first 5 val days for verification
fusion_model = MultiModalFusionModel()

test_dates = val_dates[:30]  # need 30 for seq_len
test_panel = panel.loc[test_dates[0]:test_dates[-1]]
test_panel_dl = panel_dl.loc[test_dates[0]:test_dates[-1]]

# Precompute DL predictions
seq_ds = SequenceDataset(test_panel_dl, 30, feat_cols)
dl_map = {}
with torch.no_grad():
    for i, (dt, code) in enumerate(seq_ds.idx_to_dt_code):
        x, _ = seq_ds[i]
        x_t = x.unsqueeze(0).to(DEVICE)
        l = float(lstm(x_t).cpu().numpy().mean())
        t = float(trans(x_t).cpu().numpy().mean())
        c = float(cnn(x_t).cpu().numpy().mean())
        for name, val in [('lstm', l), ('transformer', t), ('cnn', c)]:
            dl_map[(dt, code, name)] = val

print(f"\nFirst date: {test_dates[0].date()}")
print(f"DL map entries: {len(dl_map)}")

# Compare predictions for one day
dt = test_dates[0]
day_data = panel.loc[dt].copy()
codes = day_data.index.tolist()
regime = 'neutral'
reg_idx = {'trend': 0, 'neutral': 1, 'mean_reversion': 2}.get(regime, 1)

results = {}
for code in codes:
    sc = {}
    for name in ['lstm', 'transformer', 'cnn']:
        val = dl_map.get((dt, code, name), np.nan)
        if not np.isnan(val):
            sc[name] = val
    
    # Add XGB
    from trainer import TECHNICAL_FEATURES as TF
    for regime_name, model in xgb_models.items():
        if regime_name not in xgb_scalers: continue
        feats = {'trend': ['Ret_20', 'Ret_60', 'Price_MA_60_Ratio', 'MA_20_60_Cross', 'Volatility_20', 'Amount_Ratio'],
                'mean_reversion': ['Ret_5', 'Ret_10', 'RSI', 'BB_Position', 'VWAP_Dist', 'Price_Position', 'High_Low_Ratio'],
                'neutral': TECHNICAL_FEATURES[:18],}.get(regime_name, TECHNICAL_FEATURES[:18])
        feats = [c for c in feats if c in day_data.columns]
        if feats:
            X = day_data.loc[[code]][feats].fillna(0).values
            if len(X) > 0:
                pred = model.predict(xgb_scalers[regime_name].transform(X))
                sc[f'xgb_{regime_name}'] = float(pred[0])
    
    # Method 1: DL mean
    dl_vals = [v for k, v in sc.items() if k in ['lstm', 'transformer', 'cnn'] and not np.isnan(v)]
    avg_dl = np.mean(dl_vals) if dl_vals else 0
    
    # Method 2: Fusion
    fusion_val = fusion_model.fuse_predictions(sc, regime_idx=np.array([reg_idx]))
    
    # Method 3: XGB only (use regime-specific)
    xgb_key = f'xgb_{regime}'
    xgb_val = sc.get(xgb_key, 0)
    
    # Method 4: XGB all mean
    xgb_vals = [v for k, v in sc.items() if k.startswith('xgb_') and not np.isnan(v)]
    avg_xgb = np.mean(xgb_vals) if xgb_vals else 0
    
    results[code] = {'avg_dl': avg_dl, 'fusion': fusion_val, 'xgb': xgb_val, 'avg_xgb': avg_xgb}

# Show top 10 stocks by each method
for method in ['avg_dl', 'fusion', 'xgb', 'avg_xgb']:
    sorted_codes = sorted(results.items(), key=lambda x: x[1][method], reverse=True)
    top5 = [(c, f'{v[method]:.4f}') for c, v in sorted_codes[:5]]
    bottom5 = [(c, f'{v[method]:.4f}') for c, v in sorted_codes[-5:]]
    print(f'\n{method:10s}: top5={top5[:3]}...')
    print(f'{"":10s}  bottom5={bottom5[:3]}...')
    # Check overlap with avg_dl
    avg_dl_top5 = set(c for c, v in sorted(results.items(), key=lambda x: x[1]['avg_dl'], reverse=True)[:5])
    my_top5 = set(c for c, v in sorted_codes[:5])
    overlap = len(avg_dl_top5 & my_top5)
    print(f'{"":10s}  overlap with avg_dl top5: {overlap}/5')

# Check specific prediction values for a few stocks
print(f'\nDetailed comparison for a few stocks:')
sample_codes = list(results.keys())[:5]
for code in sample_codes:
    r = results[code]
    print(f'  {code}: avg_dl={r["avg_dl"]:.6f}  fusion={r["fusion"]:.6f}  xgb={r["xgb"]:.6f}  avg_xgb={r["avg_xgb"]:.6f}')

# Rank Spearman
import scipy.stats as st
all_avg = [r['avg_dl'] for r in results.values()]
all_fus = [r['fusion'] for r in results.values()]
all_xgb = [r['xgb'] for r in results.values()]
all_axb = [r['avg_xgb'] for r in results.values()]
print(f'\nRank correlations:')
print(f'  avg_dl vs fusion: {st.spearmanr(all_avg, all_fus)[0]:.4f}')
print(f'  avg_dl vs xgb:    {st.spearmanr(all_avg, all_xgb)[0]:.4f}')
print(f'  avg_dl vs avg_xgb:{st.spearmanr(all_avg, all_axb)[0]:.4f}')
print(f'  fusion vs xgb:    {st.spearmanr(all_fus, all_xgb)[0]:.4f}')
print(f'  xgb vs avg_xgb:   {st.spearmanr(all_xgb, all_axb)[0]:.4f}')

print(f'\nValue ranges:')
print(f'  avg_dl: [{min(all_avg):.6f}, {max(all_avg):.6f}]')
print(f'  fusion: [{min(all_fus):.6f}, {max(all_fus):.6f}]')
print(f'  xgb:    [{min(all_xgb):.6f}, {max(all_xgb):.6f}]')
print(f'  avg_xgb:[{min(all_axb):.6f}, {max(all_axb):.6f}]')

print('\nDone!')
