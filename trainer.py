import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm

from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOP_PATIENCE, OUTPUT_DIR
from features import TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES


def prepare_dl_panel(df_panel, feature_cols=None, target_col='Future_20d_Ret', seq_len=60):
    if feature_cols is None:
        feature_cols = TECHNICAL_FEATURES_DL
    all_feat = feature_cols + [target_col]
    df = df_panel.dropna(subset=all_feat).copy()

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(df[feature_cols])
    for i, col in enumerate(feature_cols):
        df[col] = feat_scaled[:, i]
    return df, scaler


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, 
                early_stop_patience=EARLY_STOP_PATIENCE, model_name='model'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    model = model.to(DEVICE)
    print(f"\n训练 {model_name} on {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_batches += 1
                val_preds.extend(y_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)
        history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        val_ic, _ = stats.spearmanr(val_preds, val_targets) if len(val_preds) > 1 else (0, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | Val IC: {val_ic:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  早停于 Epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_pred = model(x_batch)
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    final_ic, _ = stats.spearmanr(all_preds, all_targets) if len(all_preds) > 1 else (0, 1)
    print(f"  {model_name} 最终验证IC: {final_ic:.4f}")

    return model, history, final_ic


def train_all_deep_models(df_panel, market_df, seq_len=60):
    from data_layer import SequenceDataset
    from features import SENTIMENT_FEATURES

    print(f"\n{'='*60}")
    print("训练深度学习模型")
    print(f"{'='*60}")

    df_panel = df_panel.copy()
    df_panel['Future_20d_Ret'] = df_panel.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1
    )

    # 动态检测情感特征列
    sent_avail = [c for c in SENTIMENT_FEATURES if c in df_panel.columns]
    feat_cols = TECHNICAL_FEATURES_DL + sent_avail

    df_panel, scaler = prepare_dl_panel(df_panel, feat_cols)
    input_size = len(feat_cols)
    print(f"输入特征维度: {input_size} (技术 {len(TECHNICAL_FEATURES_DL)} + 情感 {len(sent_avail)})")

    all_dates = sorted(df_panel.index.get_level_values(0).unique())
    split_idx = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]

    train_panel = df_panel[df_panel.index.get_level_values(0).isin(train_dates)]
    val_panel = df_panel[df_panel.index.get_level_values(0).isin(val_dates)]

    train_dataset = SequenceDataset(train_panel, seq_len, feat_cols)
    val_dataset = SequenceDataset(val_panel, seq_len, feat_cols)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("数据集为空，跳过深度学习训练")
        return {}, {}, None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    models = {}

    from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer

    lstm_model = LSTMStockPredictor(input_size=input_size).to(DEVICE)
    lstm_model, lstm_hist, lstm_ic = train_model(
        lstm_model, train_loader, val_loader, model_name='LSTM'
    )
    models['lstm'] = lstm_model
    lstm_model.save(os.path.join(OUTPUT_DIR, 'lstm_model.pt'))

    transformer_model = TransformerStockPredictor(input_size=input_size).to(DEVICE)
    transformer_model, trans_hist, trans_ic = train_model(
        transformer_model, train_loader, val_loader, model_name='Transformer'
    )
    models['transformer'] = transformer_model
    transformer_model.save(os.path.join(OUTPUT_DIR, 'transformer_model.pt'))

    cnn_model = CNNChartPatternRecognizer(in_channels=input_size).to(DEVICE)
    cnn_model, cnn_hist, cnn_ic = train_model(
        cnn_model, train_loader, val_loader, model_name='CNN_Chart'
    )
    models['cnn'] = cnn_model
    cnn_model.save(os.path.join(OUTPUT_DIR, 'cnn_model.pt'))

    print(f"\n深度学习模型训练完成!")
    print(f"  LSTM IC: {lstm_ic:.4f}")
    print(f"  Transformer IC: {trans_ic:.4f}")
    print(f"  CNN IC: {cnn_ic:.4f}")

    return models, scaler, input_size


def train_xgboost_models(df_panel, market_df):
    from xgboost import XGBRegressor
    from config import MODEL_CONFIGS as XGB_CONFIGS
    from regime import detect_market_regime

    print(f"\n{'='*60}")
    print("训练XGBoost基准模型")
    print(f"{'='*60}")

    df = df_panel.copy()
    df['Future_20d_Ret'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1
    )

    unique_dates = df.index.get_level_values(0).unique()
    regime_map = {}
    for date in unique_dates:
        market_slice = market_df.loc[:date] if date in market_df.index else market_df
        if isinstance(market_slice, pd.DataFrame) and not market_slice.empty:
            regime_map[date] = detect_market_regime(market_slice)
        else:
            regime_map[date] = 'neutral'
    df['regime'] = df.index.get_level_values(0).map(regime_map)

    xgb_models = {}
    xgb_scalers = {}
    feat_importances = []

    xgb_config = {
        'trend': {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8},
        'mean_reversion': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.6},
        'neutral': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7},
    }
    from features import SENTIMENT_FEATURES
    sent_avail = [c for c in SENTIMENT_FEATURES if c in df_panel.columns]
    xgb_feats = {
        'trend': ['Ret_20', 'Ret_60', 'Price_MA_60_Ratio', 'MA_20_60_Cross', 'Volatility_20', 'Amount_Ratio'] + sent_avail,
        'mean_reversion': ['Ret_5', 'Ret_10', 'RSI', 'BB_Position', 'VWAP_Dist', 'Price_Position', 'High_Low_Ratio'] + sent_avail,
        'neutral': TECHNICAL_FEATURES[:18] + sent_avail,
    }

    for regime in xgb_config:
        mask = df['regime'] == regime
        reg_data = df[mask].dropna(subset=xgb_feats[regime] + ['Future_20d_Ret'])
        if len(reg_data) < 50:
            print(f"  {regime}: 样本不足({len(reg_data)}), 跳过")
            continue

        X = reg_data[xgb_feats[regime]].replace([np.inf, -np.inf], np.nan)
        y = reg_data['Future_20d_Ret'].replace([np.inf, -np.inf], np.nan)
        not_null = y.notna() & X.notna().all(axis=1)
        X = X[not_null]
        y = y[not_null]
        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        s = StandardScaler()
        X_tr_sc = s.fit_transform(X_tr)
        X_val_sc = s.transform(X_val)

        model = XGBRegressor(**xgb_config[regime], random_state=42, verbosity=0)
        model.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)
        pred = model.predict(X_val_sc)
        ic, _ = stats.spearmanr(y_val, pred) if len(pred) > 1 else (0, 1)
        print(f"  {regime}: {len(reg_data)}样本, 验证IC={ic:.4f}")

        xgb_models[regime] = model
        xgb_scalers[regime] = s
        imp = pd.DataFrame({'feature': xgb_feats[regime], 'importance': model.feature_importances_, 'regime': regime})
        feat_importances.append(imp)

    return xgb_models, xgb_scalers, pd.concat(feat_importances) if feat_importances else pd.DataFrame()
