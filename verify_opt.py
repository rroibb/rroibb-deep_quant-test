"""电子科技板块 - 优化版验证"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=UserWarning)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import engineer_features, TECHNICAL_FEATURES_DL
from trainer import train_xgboost_models, prepare_dl_panel, train_model
from backtest import DeepQuantBacktester
from config import STOCK_POOL, SECTOR_MAP

# 用C盘缓存 (电子科技数据)
CACHE = r'C:\Users\24259\PycharmProjects\金融数据分析\a_stock_cache'
OUTPUT = os.path.join(BASE, 'output')
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 60)
print("  电子科技50股 - 优化版验证")
print("=" * 60)

# 加载
loaded = {}
for fname in os.listdir(CACHE):
    if not fname.endswith('.parquet'): continue
    ticker = fname.split('_')[0]
    if ticker not in STOCK_POOL: continue
    try:
        df = pd.read_parquet(os.path.join(CACHE, fname))
        if len(df) > 60:
            df = engineer_features(df)
            df['code'] = ticker
            loaded[ticker] = df
    except: pass
print(f"加载: {len(loaded)} stocks")

df_list = []
for tick, df in loaded.items():
    temp = df.copy().reset_index()
    temp['code'] = tick
    df_list.append(temp)
panel = pd.concat(df_list, ignore_index=True)
panel['date'] = pd.to_datetime(panel['trade_date'])
panel = panel.set_index(['date', 'code']).sort_index()

# 基准
all_returns = []
for ticker, df in loaded.items():
    all_returns.append(df['close'].pct_change().rename(ticker))
ret_panel = pd.concat(all_returns, axis=1)
eq_ret = ret_panel.mean(axis=1)
market_df = pd.DataFrame(index=eq_ret.index)
market_df['close'] = (1 + eq_ret).cumprod()
market_df['Simple_Return'] = eq_ret

# 训练
panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)
panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)
all_dates = sorted(panel_dl.index.get_level_values(0).unique())
recent_dates = all_dates[-500:]
sp = int(len(recent_dates)*0.8)
train_dates, val_dates = recent_dates[:sp], recent_dates[sp:]

from torch.utils.data import DataLoader
from data_layer import SequenceDataset
from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
import torch
from config import DEVICE

train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
train_ds = SequenceDataset(train_p, 30, TECHNICAL_FEATURES_DL[:12])
val_ds = SequenceDataset(val_p, 30, TECHNICAL_FEATURES_DL[:12])
train_loader = DataLoader(train_ds, 128, shuffle=True)
val_loader = DataLoader(val_ds, 128, shuffle=False)

dl_models = {}
print("\nTraining DL models...")
lstm = LSTMStockPredictor(12, 64, 1, 0).to(DEVICE)
lstm, _, _ = train_model(lstm, train_loader, val_loader, 5, model_name='LSTM')
dl_models['lstm'] = lstm
trans = TransformerStockPredictor(12, 64, 2, 2).to(DEVICE)
trans, _, _ = train_model(trans, train_loader, val_loader, 5, model_name='Transformer')
dl_models['transformer'] = trans
cnn = CNNChartPatternRecognizer(12).to(DEVICE)
cnn, _, _ = train_model(cnn, train_loader, val_loader, 5, model_name='CNN')
dl_models['cnn'] = cnn

xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

from models.fusion import MultiModalFusionModel
from models.nlp_sentiment import NLPSentimentAnalyzer
from models.llm_analyzer import LLMAnalyzer
all_models = {**dl_models, 'fusion': MultiModalFusionModel(),
              'nlp': NLPSentimentAnalyzer(), 'llm': LLMAnalyzer(use_mock=True)}

# 回测
v0, v1 = val_dates[0], val_dates[-1]
bt = DeepQuantBacktester(panel.loc[v0:v1], market_df.loc[v0:v1],
                         all_models, scaler, xgb_models=xgb_models, xgb_scalers=xgb_scalers)

results_tech = {}
for name, uf, ux, ud in [
    ('Multimodal Fusion (DL+XGB+NLP+LLM)', True, True, True),
    ('Deep Learning Only (LSTM+Transformer+CNN)', False, False, True),
    ('XGBoost Only', False, True, False),
]:
    daily = bt.run(use_fusion=uf, use_xgb=ux, use_dl=ud)
    results_tech[name] = daily

bench = results_tech[list(results_tech.keys())[0]]['Cum_Benchmark'].iloc[-1]*100
print(f"\n电子科技50 - 优化版结果:")
print(f"  基准: {bench:.2f}%")
for name, d in results_tech.items():
    print(f"  {name}: {d['Cum_Strategy'].iloc[-1]*100:.2f}%")