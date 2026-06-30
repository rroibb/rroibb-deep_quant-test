"""
================================================================================
  深度学习多模态量化系统 - 主入口
  ==============================================================================
  领域覆盖:
    - CV (CNN):     K线图表模式识别
    - ML (XGBoost): 传统机器学习基线
    - Robotics:     时序预测 (LSTM/Transformer)
  
  模型架构:
    LSTM → 时序预测
    Transformer → 注意力价格预测
    CNN → 图表模式
    Fusion → 多模态融合
  
  执行流程:
    1. 获取数据 (Tushare)
    2. 特征工程
    3. 训练DL模型 (PyTorch)
    4. 训练XGBoost基线
    5. 多模态融合回测
    6. 报告导出
================================================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCK_POOL, SECTOR_MAP, OUTPUT_DIR, DEVICE
from data_layer import fetch_all_stocks, fetch_market_index
from features import engineer_features, TECHNICAL_FEATURES_DL, merge_sentiment_features, SENTIMENT_FEATURES
from trainer import train_all_deep_models, train_xgboost_models
from backtest import DeepQuantBacktester


def setup_chinese_font():
    font_path = './SimHei.ttf'
    if not os.path.exists(font_path):
        try:
            import urllib.request
            url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
            urllib.request.urlretrieve(url, font_path)
        except:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return prop
    return None


def plot_results(daily_ret, title="深度学习量化策略"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(daily_ret.index, daily_ret['Cum_Benchmark'] * 100, label='等权基准', alpha=0.8)
    axes[0, 0].plot(daily_ret.index, daily_ret['Cum_Strategy'] * 100, label='DL多模态策略', alpha=0.8)
    axes[0, 0].set_title(f'{title} 累积净值 (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    monthly = daily_ret.resample('ME').agg({'Simple_Return': 'sum', 'Strategy_Ret': 'sum'}) * 100
    x = np.arange(len(monthly))
    w = 0.35
    axes[0, 1].bar(x - w / 2, monthly['Simple_Return'], w, label='基准', alpha=0.7)
    axes[0, 1].bar(x + w / 2, monthly['Strategy_Ret'], w, label='策略', alpha=0.7)
    axes[0, 1].set_title('月度收益对比 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    if 'Regime' in daily_ret.columns:
        colors = {'trend': '#d62728', 'mean_reversion': '#2ca02c', 'neutral': '#1f77b4'}
        for r in daily_ret['Regime'].unique():
            m = daily_ret['Regime'] == r
            axes[1, 0].scatter(daily_ret.index[m], daily_ret['Cum_Strategy'][m] * 100,
                               label=r, alpha=0.6, s=12, color=colors.get(r, 'gray'))
        axes[1, 0].set_title('策略收益按市场环境着色')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    roll_vol = daily_ret['Strategy_Ret'].rolling(20).std() * np.sqrt(240) * 100
    axes[1, 1].plot(roll_vol.index, roll_vol, alpha=0.7)
    axes[1, 1].set_title('滚动20日年化波动率 (%)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'dl_quant_chart_{datetime.now():%Y%m%d_%H%M%S}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {path}")
    plt.show()


def main():
    print("=" * 70)
    print("  深度学习多模态量化系统 v2.0 — 严格OOS回测")
    print("  " + "=" * 50)
    print("  时间切分: 训练 60% / 验证 20% / 测试 20%")
    print("  领域覆盖:")
    print("    ├── CV  (CNN):      K线图表模式识别")
    print("    ├── ML  (XGBoost):  传统机器学习基线")
    print("    └── Robotics (LSTM/Transformer): 时序预测")
    print("  " + "=" * 50)
    print(f"  计算设备: {DEVICE}")
    print("=" * 70)

    setup_chinese_font()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    end_dt = datetime.now().strftime('%Y-%m-%d')
    start_dt = (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    print(f"\n数据区间: {start_dt} ~ {end_dt}")

    # === 1. 数据获取 ===
    stock_data = fetch_all_stocks(start_dt, end_dt)
    if len(stock_data) < 30:
        print(f"有效标的仅{len(stock_data)}只, 退出")
        return

    # === 2. 特征工程 ===
    print("\n特征工程...")
    for tick in stock_data:
        stock_data[tick] = engineer_features(stock_data[tick])
        stock_data[tick]['code'] = tick

    df_list = []
    for tick, df in stock_data.items():
        temp = df.copy().reset_index()
        temp['code'] = tick
        df_list.append(temp)

    df_panel = pd.concat(df_list, ignore_index=True)
    df_panel['date'] = pd.to_datetime(df_panel['trade_date'])
    df_panel = df_panel.set_index(['date', 'code']).sort_index()
    print(f"面板数据: {df_panel.shape}")

    # === 2.5 融合情感特征 ===
    print("\n加载投资者情绪数据（东方财富/雪球）...")
    df_panel = merge_sentiment_features(df_panel, STOCK_POOL)

    # === 3. 市场指数 ===
    market_df = fetch_market_index(start_dt, end_dt)
    if market_df.empty:
        print("合成市场指数...")
        daily_ret = df_panel.groupby(level=0)['Simple_Return'].mean()
        market_df = pd.DataFrame(index=daily_ret.index)
        market_df['close'] = (1 + daily_ret).cumprod()
    print(f"指数数据: {len(market_df)}天")

    # ──────────────────────────────────────────────────
    # 时间切分：训练 60% / 验证 20% / 测试 20%
    # 严格保证：回测只跑测试集（模型从未见过的未来数据）
    # ──────────────────────────────────────────────────
    all_dates = sorted(df_panel.index.get_level_values(0).unique())
    n = len(all_dates)
    split_train = int(n * 0.6)
    split_test  = int(n * 0.8)

    train_dates = set(all_dates[:split_train])
    val_dates   = set(all_dates[split_train:split_test])
    test_dates  = set(all_dates[split_test:])

    print(f"\n{'='*60}")
    print(f"时间切分: 训练 {all_dates[0].date()} ~ {all_dates[split_train-1].date()}")
    print(f"          验证 {all_dates[split_train].date()} ~ {all_dates[split_test-1].date()}")
    print(f"          测试 {all_dates[split_test].date()} ~ {all_dates[-1].date()}")
    print(f"{'='*60}")

    idx_level = df_panel.index.get_level_values(0)

    train_panel = df_panel[idx_level.isin(train_dates)].copy()
    val_panel   = df_panel[idx_level.isin(val_dates)].copy()
    test_panel  = df_panel[idx_level.isin(test_dates)].copy()

    # 回测面板：测试期 + 120日回看缓冲（确保特征/序列可计算）
    buffer_dates = set(all_dates[max(0, split_test - 120):split_test])
    backtest_panel = df_panel[idx_level.isin(buffer_dates | test_dates)].copy()
    backtest_start_date = all_dates[split_test]

    # === 4. 训练深度学习模型 (PyTorch) ===
    # DL训练内部会做缩放，返回的 scaler 拟合于 train+val
    train_val_panel = pd.concat([train_panel, val_panel])
    dl_models, scaler_dl, input_size = train_all_deep_models(train_val_panel, market_df)

    # 用训练集的 scaler 变换回测面板特征（OOS变换）
    feat_cols = TECHNICAL_FEATURES_DL + [c for c in SENTIMENT_FEATURES if c in df_panel.columns]
    avail_cols = [c for c in feat_cols if c in backtest_panel.columns]
    if scaler_dl is not None and avail_cols:
        bt_feat_scaled = scaler_dl.transform(backtest_panel[avail_cols].fillna(0))
        for i, col in enumerate(avail_cols):
            backtest_panel[col] = bt_feat_scaled[:, i]

    # === 5. 训练XGBoost基线 ===
    xgb_models, xgb_scalers, feat_imp = train_xgboost_models(train_val_panel, market_df)
    if not feat_imp.empty:
        feat_imp.to_csv(os.path.join(OUTPUT_DIR, f'feature_importance_{datetime.now():%Y%m%d}.csv'),
                        index=False, encoding='utf-8-sig')
        print(f"特征重要性已保存")

    # === 6. 初始化多模态融合模型 ===
    from models.fusion import MultiModalFusionModel

    fusion_model = MultiModalFusionModel()

    all_models = {**dl_models, 'fusion': fusion_model}

    # === 7. 回测（仅测试集） ===
    backtester = DeepQuantBacktester(
        backtest_panel, market_df, all_models, scaler_dl,
        xgb_models=xgb_models, xgb_scalers=xgb_scalers,
        backtest_start=backtest_start_date
    )

    print(f"\n{'='*60}")
    print("启动全模态融合回测 (OOS测试集)")
    print(f"{'='*60}")
    daily_fusion = backtester.run(use_fusion=True, use_xgb=True, use_dl=True)

    print(f"\n{'='*60}")
    print("启动纯DL模型回测对比 (OOS测试集)")
    print(f"{'='*60}")
    daily_dl_only = backtester.run(use_fusion=False, use_xgb=False, use_dl=True)

    print(f"\n{'='*60}")
    print("启动纯XGBoost模型回测对比 (OOS测试集)")
    print(f"{'='*60}")
    daily_xgb_only = backtester.run(
        use_fusion=False, use_xgb=True, use_dl=False
    )

    # === 8. 保存结果 ===
    backtester.save_results(daily_fusion, 'fusion_oos')
    backtester.save_results(daily_dl_only, 'dl_only_oos')
    backtester.save_results(daily_xgb_only, 'xgb_only_oos')

    # === 9. 绘图 ===
    if not daily_fusion.empty:
        plot_results(daily_fusion, "DL多模态融合策略 (OOS)")
    if not daily_dl_only.empty:
        plot_results(daily_dl_only, "纯DL策略 (OOS)")
    if not daily_xgb_only.empty:
        plot_results(daily_xgb_only, "纯XGB策略 (OOS)")

    print(f"\n{'='*70}")
    print("✅ 深度学习多模态量化系统运行完成!")
    print(f"   训练期: {all_dates[0].date()} ~ {all_dates[split_train-1].date()}")
    print(f"   验证期: {all_dates[split_train].date()} ~ {all_dates[split_test-1].date()}")
    print(f"   测试期: {all_dates[split_test].date()} ~ {all_dates[-1].date()}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
