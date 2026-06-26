print("Step 1: imports")
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
print("Step 2: engine features")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
print("Step 3: trainer")
from trainer import prepare_dl_panel
print("Step 4: config")
from config import STOCK_POOL
print("Step 5: parquet test")
df = pd.read_parquet(os.path.join(r'C:\Users\24259\PycharmProjects\金融数据分析\a_stock_cache', '000063.SZ_2021-06-22_2026-06-22.parquet'))
print(f"Step 6: got parquet {len(df)} rows, columns={list(df.columns[:5])}")
df = engineer_features(df)
print(f"Step 7: engineer_features OK, cols={len(df.columns)}")
print("ALL OK - running tests passing")