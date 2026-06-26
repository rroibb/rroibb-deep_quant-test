"""
================================================================================
  深度学习多模态量化系统 - 主入口
  ==============================================================================
  领域覆盖:
    - CV (CNN):     K线图表模式识别
    - NLP:          新闻情感分析 (BERT)
    - ML (XGBoost): 传统机器学习基线
    - LLM:          大模型综合分析
    - Robotics:     时序预测 (LSTM/Transformer)
  
  模型架构:
    LSTM → 时序预测
    Transformer → 注意力价格预测
    CNN → 图表模式
    BERT → 文本情感
    LLM → 宏观分析
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
from features import engineer_features, TECHNICAL_FEATURES_DL
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
    print("  深度学习多模态量化系统 v1.0")
    print("  " + "=" * 50)
    print("  领域覆盖:")
    print("    ├── CV  (CNN):      K线图表模式识别")
    print("    ├── NLP (BERT):     新闻情感分析")
    print("    ├── ML  (XGBoost):  传统机器学习基线")
    print("    ├── LLM (LLaMA):    大模型综合分析")
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

    # === 3. 市场指数 ===
    market_df = fetch_market_index(start_dt, end_dt)
    if market_df.empty:
        print("合成市场指数...")
        daily_ret = df_panel.groupby(level=0)['Simple_Return'].mean()
        market_df = pd.DataFrame(index=daily_ret.index)
        market_df['close'] = (1 + daily_ret).cumprod()
    print(f"指数数据: {len(market_df)}天")

    # === 4. 训练深度学习模型 (PyTorch) ===
    dl_models, scaler_dl, input_size = train_all_deep_models(df_panel, market_df)

    # === 5. 训练XGBoost基线 ===
    xgb_models, xgb_scalers, feat_imp = train_xgboost_models(df_panel, market_df)
    if not feat_imp.empty:
        feat_imp.to_csv(os.path.join(OUTPUT_DIR, f'feature_importance_{datetime.now():%Y%m%d}.csv'),
                        index=False, encoding='utf-8-sig')
        print(f"特征重要性已保存")

    # === 6. 初始化多模态融合模型 (mock模式) ===
    from models.fusion import MultiModalFusionModel
    from models.nlp_sentiment import NLPSentimentAnalyzer
    from models.llm_analyzer import LLMAnalyzer

    fusion_model = MultiModalFusionModel()
    nlp_model = NLPSentimentAnalyzer()
    llm_model = LLMAnalyzer(use_mock=True)

    all_models = {**dl_models, 'fusion': fusion_model, 'nlp': nlp_model, 'llm': llm_model}

    # === 7. 回测 ===
    backtester = DeepQuantBacktester(
        df_panel, market_df, all_models, scaler_dl,
        xgb_models=xgb_models, xgb_scalers=xgb_scalers
    )

    # 全模态回测
    print(f"\n{'='*60}")
    print("启动全模态融合回测 (DL + XGBoost + NLP + LLM)")
    print(f"{'='*60}")
    daily_fusion = backtester.run(use_fusion=True, use_xgb=True, use_dl=True)

    # 纯DL回测对比
    print(f"\n{'='*60}")
    print("启动纯DL模型回测对比")
    print(f"{'='*60}")
    daily_dl_only = backtester.run(use_fusion=False, use_xgb=False, use_dl=True)

    # 纯XGBoost回测对比
    print(f"\n{'='*60}")
    print("启动纯XGBoost模型回测对比")
    print(f"{'='*60}")
    daily_xgb_only = backtester.run(
        use_fusion=False, use_xgb=True, use_dl=False
    )

    # === 8. 保存结果 ===
    backtester.save_results(daily_fusion, 'fusion')
    backtester.save_results(daily_dl_only, 'dl_only')
    backtester.save_results(daily_xgb_only, 'xgb_only')

    # === 9. 绘图 ===
    plot_results(daily_fusion, "DL多模态融合策略")
    plot_results(daily_dl_only, "纯DL策略")
    plot_results(daily_xgb_only, "纯XGB策略")

    print(f"\n{'='*70}")
    print("✅ 深度学习多模态量化系统运行完成!")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
