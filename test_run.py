"""
deep_quant 测试脚本 - 使用缓存数据快速验证
基准: 股票池等权构建电子科技指数
"""
import sys, os, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # 抑制sklearn特征名警告
BASE = 'C:\\Users\\24259\\PycharmProjects\\金融数据分析\\deep_quant'
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import STOCK_POOL, SECTOR_MAP, OUTPUT_DIR, DEVICE
from features import engineer_features, TECHNICAL_FEATURES_DL
from trainer import train_all_deep_models, train_xgboost_models, prepare_dl_panel
from backtest import DeepQuantBacktester
from regime import detect_market_regime, RegimeController
from datetime import datetime

CACHE_DIR = 'C:/Users/24259/PycharmProjects/金融数据分析/a_stock_cache'

def load_cached_data():
    """从缓存加载50只股票数据并构建面板"""
    print(f"缓存目录: {CACHE_DIR}")
    cache_files = os.listdir(CACHE_DIR)
    loaded = {}

    for fname in cache_files:
        if not fname.endswith('.parquet'):
            continue
        ticker = fname.split('_')[0]  # e.g. "000063.SZ"
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

    print(f"加载: {len(loaded)}/{len(STOCK_POOL)} 只股票")

    df_list = []
    for tick, df in loaded.items():
        temp = df.copy().reset_index()
        temp['code'] = tick
        df_list.append(temp)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"面板: {panel.shape}")
    return panel, loaded


def build_electronic_tech_index(loaded_data):
    """等权构建电子科技50指数作为基准"""
    print("\n构建电子科技50等权指数作为基准...")
    all_returns = []

    for ticker, df in loaded_data.items():
        ret = df['close'].pct_change().rename(ticker)
        all_returns.append(ret)

    if not all_returns:
        return pd.DataFrame()

    # 合并所有收益率，按日取等权平均
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

    print(f"指数构建完成: {len(market_df)} 个交易日")
    print(f"指数范围: {market_df.index[0].date()} ~ {market_df.index[-1].date()}")
    print(f"指数累计收益: {(market_df['close'].iloc[-1]/market_df['close'].iloc[0] - 1)*100:.2f}%")
    return market_df


def fast_test():
    print("=" * 70)
    print("  deep_quant 快速测试 - 电子科技50股")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载缓存数据
    panel, loaded = load_cached_data()
    if len(loaded) < 30:
        print(f"标的不足: {len(loaded)}")
        return

    # 2. 构建电子科技指数基准
    market_df = build_electronic_tech_index(loaded)

    # 3. 准备预测目标
    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(
        lambda x: x.shift(-20) / x - 1
    )

    # 4. 只训练 DL 模型 (快速: 少量epoch)
    print(f"\n{'='*60}")
    print("训练深度学习模型 (快速模式: epochs=10)")
    print(f"{'='*60}")

    panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)
    input_size = len(TECHNICAL_FEATURES_DL)

    from torch.utils.data import DataLoader
    from data_layer import SequenceDataset
    from models import LSTMStockPredictor, TransformerStockPredictor, CNNChartPatternRecognizer
    from trainer import train_model
    import torch
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    # 用最近2年数据加速测试
    all_dates = sorted(panel_dl.index.get_level_values(0).unique())
    recent_dates = all_dates[-500:]
    split_idx = int(len(recent_dates) * 0.8)
    train_dates = recent_dates[:split_idx]
    val_dates = recent_dates[split_idx:]
    print(f"训练区间: {train_dates[0].date()} ~ {train_dates[-1].date()}")
    print(f"验证区间: {val_dates[0].date()} ~ {val_dates[-1].date()}")

    train_p = panel_dl[panel_dl.index.get_level_values(0).isin(train_dates)]
    val_p = panel_dl[panel_dl.index.get_level_values(0).isin(val_dates)]
    train_ds = SequenceDataset(train_p, 30, TECHNICAL_FEATURES_DL[:12])  # 减少特征+序列长
    val_ds = SequenceDataset(val_p, 30, TECHNICAL_FEATURES_DL[:12])

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("数据集为空")
        return

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    dl_models = {}

    # LSTM (缩小模型)
    lstm = LSTMStockPredictor(input_size=12, hidden_size=64, num_layers=1, bidirectional=False).to(DEVICE)
    lstm, _, _ = train_model(lstm, train_loader, val_loader, epochs=5, model_name='LSTM')
    dl_models['lstm'] = lstm

    # Transformer (缩小)
    trans = TransformerStockPredictor(input_size=12, d_model=64, nhead=2, num_encoder_layers=2).to(DEVICE)
    trans, _, _ = train_model(trans, train_loader, val_loader, epochs=5, model_name='Transformer')
    dl_models['transformer'] = trans

    # CNN (缩小)
    cnn = CNNChartPatternRecognizer(in_channels=12).to(DEVICE)
    cnn, _, _ = train_model(cnn, train_loader, val_loader, epochs=5, model_name='CNN')
    dl_models['cnn'] = cnn

    # 5. 训练 XGBoost 基线
    xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df)

    # 6. 初始化融合模型
    from models.fusion import MultiModalFusionModel

    fusion_model = MultiModalFusionModel()

    all_models = {**dl_models, 'fusion': fusion_model}

    # 7. 回测对比 (使用最后6个月验证期，加速)
    print("\n限制回测区间到最后6个月...")
    val_start = val_dates[0]
    val_end = val_dates[-1]
    panel_subset = panel.loc[val_start:val_end]
    market_subset = market_df.loc[val_start:val_end]
    print(f"回测区间: {val_start.date()} ~ {val_end.date()}, 共 {len(market_subset)} 天")

    bt = DeepQuantBacktester(panel_subset, market_subset, all_models, scaler,
                             xgb_models=xgb_models, xgb_scalers=xgb_scalers)

    results = {}
    for mode_name, use_f, use_x, use_d in [
        ('全模态融合 (DL+XGB)', True, True, True),
        ('纯深度学习 (LSTM+Transformer+CNN)', False, False, True),
        ('纯 XGBoost', False, True, False),
    ]:
        print(f"\n{'='*60}")
        print(f"{mode_name}")
        print(f"{'='*60}")
        daily = bt.run(use_fusion=use_f, use_xgb=use_x, use_dl=use_d)
        results[mode_name] = daily

    # 8. 生成对比报告
    print(f"\n{'='*60}")
    print("策略对比汇总 (基准: 电子科技50等权指数)")
    print(f"{'='*60}")

    bench_total = results[list(results.keys())[0]]['Cum_Benchmark'].iloc[-1]

    print(f"\n{'策略':<30} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'超额':>10}")
    print('-' * 80)

    for name, daily in results.items():
        total = daily['Cum_Strategy'].iloc[-1]
        ann = daily['Strategy_Ret'].mean() * 240
        vol = daily['Strategy_Ret'].std() * np.sqrt(240)
        sharpe = (ann - 0.03) / vol if vol > 1e-8 else 0
        cum_max = daily['Cum_Strategy'].cummax()
        mdd = (cum_max - daily['Cum_Strategy']).max()
        excess = total - bench_total
        print(f"{name:<30} {total*100:>8.2f}% {ann*100:>8.2f}% {sharpe:>7.2f} {mdd*100:>7.2f}% {excess*100:>8.2f}%")

    print(f"\n{'电子科技50等权指数':<30} {bench_total*100:>8.2f}%")
    print(f"\n基准指数累计收益: {bench_total*100:.2f}%")

    # 9. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, daily in results.items():
        fname = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'n')
        daily.to_csv(os.path.join(OUTPUT_DIR, f'{fname}_{timestamp}.csv'), encoding='utf-8-sig')

    # 汇总
    summary = pd.DataFrame([{
        '策略': name,
        '总收益': f"{daily['Cum_Strategy'].iloc[-1]*100:.2f}%",
        '年化收益': f"{daily['Strategy_Ret'].mean()*240*100:.2f}%",
        '年化波动': f"{daily['Strategy_Ret'].std()*np.sqrt(240)*100:.2f}%",
        '夏普比率': f"{(daily['Strategy_Ret'].mean()*240 - 0.03)/(daily['Strategy_Ret'].std()*np.sqrt(240)):.2f}",
        '最大回撤': f"{(daily['Cum_Strategy'].cummax() - daily['Cum_Strategy']).max()*100:.2f}%",
        '超额收益': f"{(daily['Cum_Strategy'].iloc[-1] - bench_total)*100:.2f}%",
    } for name, daily in results.items()])
    summary.to_csv(os.path.join(OUTPUT_DIR, f'策略对比汇总_{timestamp}.csv'), index=False, encoding='utf-8-sig')

    print(f"\n结果已保存至: {OUTPUT_DIR}")
    print(f"\n{'='*70}")
    print("测试完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    fast_test()
