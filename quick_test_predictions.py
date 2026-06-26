"""
快速验证: 三种模式下同一日股票的预测排序差异
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, TOP_N
from features import TECHNICAL_FEATURES_DL
from data_layer import SequenceDataset
from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
from trainer import train_model
from torch.utils.data import DataLoader
import torch

panel = pd.read_pickle('G:/QuantData/day_stocks.pkl')
panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20) / x - 1)
panel = panel.dropna(subset=TECHNICAL_FEATURES_DL[:12] + ['Future_20d_Ret'])

all_dates = sorted(panel.index.get_level_values(0).unique())
recent = all_dates[-500:]
split = int(len(recent) * 0.8)
train_p = panel[panel.index.get_level_values(0).isin(recent[:split])]
val_p = panel[panel.index.get_level_values(0).isin(recent[split:])]

feat_cols = TECHNICAL_FEATURES_DL[:12]
train_ds = SequenceDataset(train_p, 30, feat_cols)
val_ds = SequenceDataset(val_p, 30, feat_cols)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

lstm = LSTMStockPredictor(12, 64, 1, False).to(DEVICE)
lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=3, model_name='LSTM')
trans = TransformerStockPredictor(12, 64, 2, 2).to(DEVICE)
trans, _, _ = train_model(trans, train_loader, val_loader, epochs=3, model_name='Transformer')
cnn = CNNChartPatternRecognizer(in_channels=12).to(DEVICE)
cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=3, model_name='CNN')

test_date = recent[split]
test_data = panel[panel.index.get_level_values(0) == test_date]
codes = sorted(test_data.index.get_level_values(1).unique())
print(f'Test date: {test_date.date()}, {len(codes)} stocks')

predictions = {code: {} for code in codes}
with torch.no_grad():
    for code in codes:
        hist = panel.xs(code, level=1).loc[:test_date].tail(30)
        if len(hist) < 5:
            continue
        seq = hist[feat_cols].fillna(0).values.astype(np.float32)
        if len(seq) < 30:
            pad = np.zeros((30 - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        x = torch.from_numpy(seq[-30:]).unsqueeze(0).to(DEVICE)
        predictions[code]['lstm'] = float(lstm(x).cpu().numpy().mean())
        predictions[code]['transformer'] = float(trans(x).cpu().numpy().mean())
        predictions[code]['cnn'] = float(cnn(x).cpu().numpy().mean())
        predictions[code]['avg_dl'] = (predictions[code]['lstm'] + predictions[code]['transformer'] + predictions[code]['cnn']) / 3

for mode_name, key in [('DL Avg', 'avg_dl'), ('LSTM', 'lstm'), ('Transformer', 'transformer'), ('CNN', 'cnn')]:
    sorted_items = sorted(predictions.items(), key=lambda x: x[1].get(key, 0), reverse=True)
    top5 = sorted_items[:5]
    bottom5 = sorted_items[-5:]
    print(f'-- {mode_name} --')
    print(f'  Top5:    {", ".join([f"{c}: {s[key]:.4f}" for c, s in top5])}')
    print(f'  Bottom5: {", ".join([f"{c}: {s[key]:.4f}" for c, s in bottom5])}')
    # Check if ranking matches DL Avg
    if key != 'avg_dl':
        avg_top5 = set(c for c, s in top5)
        dl_avg_top5 = set(c for c, s in sorted(predictions.items(), key=lambda x: x[1].get('avg_dl', 0), reverse=True)[:5])
        overlap = avg_top5 & dl_avg_top5
        print(f'  Overlap with DL Avg top5: {len(overlap)}/5')

print()
print('=== Top 10 by DL Avg with all 3 model preds ===')
dl_top = sorted(predictions.items(), key=lambda x: x[1].get('avg_dl', 0), reverse=True)[:10]
for code, preds in dl_top:
    print(f'  {code}: avg={preds["avg_dl"]:.4f}  lstm={preds["lstm"]:.4f}  trans={preds["transformer"]:.4f}  cnn={preds["cnn"]:.4f}')

print('=== Bottom 10 by DL Avg ===')
dl_bot = sorted(predictions.items(), key=lambda x: x[1].get('avg_dl', 0), reverse=True)[-10:]
for code, preds in dl_bot:
    print(f'  {code}: avg={preds["avg_dl"]:.4f}  lstm={preds["lstm"]:.4f}  trans={preds["transformer"]:.4f}  cnn={preds["cnn"]:.4f}')

# Rank correlation between models
print()
print('=== Spearman Rank Correlation between models ===')
import scipy.stats as stats
model_names = ['lstm', 'transformer', 'cnn', 'avg_dl']
valid = {c: p for c, p in predictions.items() if len(p) == 4}
for i, n1 in enumerate(model_names):
    for n2 in model_names[i+1:]:
        v1 = [p[n1] for _, p in valid.items()]
        v2 = [p[n2] for _, p in valid.items()]
        rho, _ = stats.spearmanr(v1, v2)
        print(f'  {n1} vs {n2}: rho={rho:.4f}')
