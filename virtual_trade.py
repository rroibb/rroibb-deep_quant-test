"""
虚拟盘交易 — Top5股票权重分配
初始资金200,000 · 同花顺交易成本 · 整手(100股)约束
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=UserWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from features import engineer_features, TECHNICAL_FEATURES_DL, TECHNICAL_FEATURES
from trainer import prepare_dl_panel, train_xgboost_models, train_model
from config import STOCK_POOL, SECTOR_MAP, DEVICE, CYCLE_CONFIG
from backtest import DeepQuantBacktester
from regime import detect_market_cycle, CycleController
from feishu_pusher import FeishuPusher
from datetime import datetime
import torch

CACHE = r'C:\Users\24259\PycharmProjects\金融数据分析\a_stock_cache'
OUTPUT = os.path.join(BASE, 'output')
os.makedirs(OUTPUT, exist_ok=True)

# ============ 虚拟盘参数 ============
INITIAL_CAPITAL = 200000
TOP_K = 5
COMMISSION = 0.00025       # 佣金万2.5
STAMP_DUTY = 0.001         # 印花税千1 (卖出)
TRANSFER_FEE = 0.00001     # 过户费万0.1
PRICE_TOLERANCE = 0.02     # 挂单缓冲 (高于现价2%确保成交)
LOT_SIZE = 100             # A股1手=100股
MIN_LOTS = 1               # 每只股票最少买入手数 (1手=100股)
CANDIDATE_POOL = 15        # 候选池大小: 从Top15中挑选可购买股票

def _get_stock_price(code, loaded):
    """从缓存中获取股票最新价格"""
    df = loaded.get(code)
    if df is None:
        return None
    if 'trade_date' in df.columns:
        df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    return float(df['close'].iloc[-1])


def _calculate_portfolio(selected_df, loaded, position_pct=1.0):
    """计算可购买持仓: 权重分配 → 整手约束 → 输出明细"""
    effective_capital = INITIAL_CAPITAL * position_pct
    # 归一化权重 (按预测值比例)
    pred_vals = selected_df['Prediction'].values
    if pred_vals.sum() > 0:
        weights = pred_vals / pred_vals.sum()
    else:
        weights = np.ones(len(selected_df)) / len(selected_df)

    rows = []
    total_cost = 0
    for i, (_, row) in enumerate(selected_df.iterrows()):
        code = row['code']
        name = row['name']
        weight = weights[i]
        pred = row['Prediction']

        price = _get_stock_price(code, loaded)
        if price is None:
            price = float(row.get('close', 100))
        pre_close = float(row.get('pre_close', price))
        daily_chg = (price / pre_close - 1) * 100 if pre_close > 0 else 0

        order_price = price * (1 + PRICE_TOLERANCE)
        budget = effective_capital * weight
        buy_cost_rate = COMMISSION + TRANSFER_FEE

        max_shares = int(budget / (order_price * LOT_SIZE)) * LOT_SIZE
        actual_cost = max_shares * order_price * (1 + buy_cost_rate)
        while actual_cost > budget and max_shares >= LOT_SIZE:
            max_shares -= LOT_SIZE
            actual_cost = max_shares * order_price * (1 + buy_cost_rate)

        actual_cost_capital = max_shares * order_price
        commission_fee = actual_cost_capital * COMMISSION
        transfer_fee = actual_cost_capital * TRANSFER_FEE

        rows.append({
            '股票代码': code,
            '股票名称': name,
            '预测得分': f"{pred:.4f}",
            '权重': f"{weight*100:.1f}%",
            '现价': f"{price:.2f}",
            '涨跌幅': f"{daily_chg:+.2f}%",
            '挂单价': f"{order_price:.2f}",
            '买入(手)': max_shares // LOT_SIZE,
            '买入(股)': max_shares,
            '占用资金': f"{actual_cost_capital:,.2f}",
            '佣金': f"{commission_fee:.2f}",
            '过户费': f"{transfer_fee:.2f}",
            '买入总成本': f"{actual_cost:,.2f}",
        })
        total_cost += actual_cost

    portfolio = pd.DataFrame(rows)
    cash_left = effective_capital - total_cost
    return portfolio, cash_left


def _select_affordable_topk(last_day, loaded, top_k=5, candidate_pool=15):
    """
    可购买性感知的Top K选股:
    1. 从候选池中按预测得分排序
    2. 检查每只股票是否能至少买 MIN_LOTS 手
    3. 若不能, 跳过并尝试下一候选
    4. 直到凑满 top_k 只可购买股票
    """
    candidates = last_day.nlargest(candidate_pool, 'Prediction')
    selected = []
    skipped = []

    # 逐轮筛选: 每选入一只, 剩余候选的权重会增加, 提高后续购买力
    remaining = candidates.copy()
    while len(selected) < top_k and len(remaining) > 0:
        best = remaining.iloc[0]
        remaining = remaining.iloc[1:]

        code = best['code']
        price = _get_stock_price(code, loaded)
        if price is None:
            skipped.append(f"{code}(无价格)")
            continue

        order_price = price * (1 + PRICE_TOLERANCE)
        min_cost = order_price * LOT_SIZE * MIN_LOTS * (1 + COMMISSION + TRANSFER_FEE)

        # 估算可用资金: 当前选中数+1 只股票近似均分
        estimated_slots = max(len(selected) + 1, top_k)
        approx_budget = INITIAL_CAPITAL / estimated_slots * 1.2  # 留20%余量

        can_afford = approx_budget >= min_cost
        if can_afford:
            selected.append(best)
            print(f"  [选中] {best['name']}({code}) 价格={price:.2f} [OK]")
        else:
            skipped.append(f"{best['name']}({code}) 价格={price:.2f}过高")
            print(f"  [替换] {best['name']}({code}) 价格={price:.2f}, 无法买{MIN_LOTS}手, 跳过")

    # 若仍不足 top_k, 放宽约束从剩余中补充
    if len(selected) < top_k:
        print(f"  [提示] 仅找到 {len(selected)}/{top_k} 只可购买股票, 从剩余中补充...")
        for _, row in remaining.iterrows():
            if len(selected) >= top_k:
                break
            if row['code'] not in [s['code'] for s in selected]:
                selected.append(row)
                print(f"  [补充] {row['name']}({row['code']})")

    if skipped:
        print(f"  [跳过] {'; '.join(skipped)}")

    return pd.DataFrame(selected)


def load_and_train():
    """加载数据 + 训练全部模型"""
    print("=" * 60)
    print("  虚拟盘交易信号生成")
    print(f"  资金: {INITIAL_CAPITAL:,.0f} | 选股: Top {TOP_K} | 池: 电子科技50股")
    print("=" * 60)

    # 加载
    loaded = {}
    for fname in os.listdir(CACHE):
        if not fname.endswith('.parquet'): continue
        ticker = fname.split('_')[0]
        if ticker not in STOCK_POOL: continue
        try:
            try:
                df = pd.read_parquet(os.path.join(CACHE, fname))
            except:
                df = pd.read_parquet(os.path.join(CACHE, fname), engine='fastparquet')
            if len(df) > 60:
                df = engineer_features(df)
                df['code'] = ticker
                loaded[ticker] = df
        except Exception as e:
            print(f"  [跳过] {ticker}: {e}")
    print(f"  加载: {len(loaded)}/{len(STOCK_POOL)} 只股票")

    df_list = []
    for tick, df in loaded.items():
        df_t = df.copy()
        if 'trade_date' in df_t.columns:
            df_t = df_t.set_index('trade_date')
        df_t.index = pd.to_datetime(df_t.index)
        df_t = df_t.sort_index().reset_index()
        df_t['code'] = tick
        df_list.append(df_t)

    panel = pd.concat(df_list, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['trade_date'])
    panel = panel.set_index(['date', 'code']).sort_index()

    # 训练 DL
    print("\n  训练深度学习模型...")
    panel['Future_20d_Ret'] = panel.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)
    panel_dl, scaler = prepare_dl_panel(panel, TECHNICAL_FEATURES_DL)
    all_dates = sorted(panel_dl.index.get_level_values(0).unique())
    recent = all_dates[-500:]
    sp = int(len(recent)*0.8)
    train_dates, val_dates = recent[:sp], recent[sp:]

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
        print(f"    训练 {nm}...")
        m = cls(12, 64, 1, 0).to(DEVICE) if nm=='LSTM' else (cls(12, 64, 2, 2).to(DEVICE) if nm=='Transformer' else cls(12).to(DEVICE))
        m, _, ic = train_model(m, tr, vl, 5, model_name=f'{nm}_Virtual')
        dl_models[nm.lower()] = m

    # 构建 market_df
    all_ret2 = []
    for ticker, df in loaded.items():
        df_t2 = df.copy()
        if 'trade_date' in df_t2.columns:
            df_t2 = df_t2.set_index('trade_date')
        df_t2.index = pd.to_datetime(df_t2.index)
        all_ret2.append(df_t2['close'].pct_change().rename(ticker))
    eq_ret2 = pd.concat(all_ret2, axis=1).mean(axis=1)
    market_df_train = pd.DataFrame(index=eq_ret2.index)
    market_df_train['close'] = (1 + eq_ret2).cumprod()
    market_df_train['Simple_Return'] = eq_ret2
    market_df_train['open'] = market_df_train['high'] = market_df_train['low'] = market_df_train['close']
    market_df_train['pre_close'] = market_df_train['close'].shift(1).fillna(market_df_train['close'].iloc[0])
    market_df_train['volume'] = market_df_train['amount'] = 0
    xgb_models, xgb_scalers, _ = train_xgboost_models(panel, market_df_train)

    from models.fusion import MultiModalFusionModel
    all_models = {**dl_models, 'fusion': MultiModalFusionModel()}

    return panel, loaded, all_models, scaler, xgb_models, xgb_scalers


def generate_top_picks(panel, loaded, all_models, scaler, xgb_models, xgb_scalers):
    """生成当日前Top5股票的预测和权重"""
    # 构建market_df
    all_ret = []
    for ticker, df in loaded.items():
        df_t = df.copy()
        if 'trade_date' in df_t.columns:
            df_t = df_t.set_index('trade_date')
        df_t.index = pd.to_datetime(df_t.index)
        all_ret.append(df_t['close'].pct_change().rename(ticker))
    eq_ret = pd.concat(all_ret, axis=1).mean(axis=1)
    market_df = pd.DataFrame(index=eq_ret.index)
    market_df['close'] = (1 + eq_ret).cumprod()
    market_df['Simple_Return'] = eq_ret
    market_df['open'] = market_df['high'] = market_df['low'] = market_df['close']
    market_df['pre_close'] = market_df['close'].shift(1).fillna(market_df['close'].iloc[0])
    market_df['volume'] = market_df['amount'] = 0

    # 用最后100天回测，取最新预测
    all_dates = sorted(panel.index.get_level_values(0).unique())
    val_dates = all_dates[-100:]
    v0, v1 = val_dates[0], val_dates[-1]
    panel_subset = panel.loc[v0:v1]
    market_subset = market_df.loc[v0:v1]

    bt = DeepQuantBacktester(panel_subset, market_subset, all_models, scaler,
                             xgb_models=xgb_models, xgb_scalers=xgb_scalers)
    daily = bt.run(use_fusion=True, use_xgb=True, use_dl=True)

    # 获取最后一天的预测值
    last_dt = sorted(daily.index)[-1]
    df_pred = panel_subset.copy()
    df_pred['Future_20d_Ret'] = df_pred.groupby('code')['close'].transform(lambda x: x.shift(-20)/x-1)

    # 重新跑一次只取prediction的最终日
    # 用backtest的预计算
    from trainer import TECHNICAL_FEATURES_DL as DL_FEATS
    from data_layer import SequenceDataset as SeqDS
    dl_feat = [c for c in DL_FEATS if c in panel_subset.columns]
    bt._cached_dl_map = None
    dl_map = bt._precompute_dl(dl_feat[:12])

    # 最后一天的预测
    last_day = panel_subset.xs(last_dt, level=0).copy().reset_index()
    if last_day.empty:
        last_day = panel_subset.loc[pd.IndexSlice[panel_subset.index.get_level_values(0).unique()[-1], :], :].reset_index()

    codes = last_day['code'].tolist()
    n = len(codes)

    # DL predictions
    for name in bt.dl_models:
        vals = np.array([dl_map.get((last_dt, c, name), np.nan) for c in codes], dtype=np.float32)
        last_day[f'dl_{name}'] = vals

    # XGBoost predictions
    from regime import detect_market_regime, RegimeController
    rc = RegimeController(min_hold_days=10)
    reg_raw = detect_market_regime(market_df.loc[:last_dt])
    reg = rc.update(last_dt, reg_raw)
    reg_idx = {'trend': 0, 'neutral': 1, 'mean_reversion': 2}.get(reg, 1)

    xgb_feat_cfg = {
        'trend': ['Ret_20','Ret_60','Price_MA_60_Ratio','MA_20_60_Cross','Volatility_20','Amount_Ratio'],
        'mean_reversion': ['Ret_5','Ret_10','RSI','BB_Position','VWAP_Dist','Price_Position','High_Low_Ratio'],
        'neutral': [c for c in DL_FEATS if c in TECHNICAL_FEATURES][:18],
    }

    for rn, model in bt.xgb_models.items():
        if rn not in bt.xgb_scalers: continue
        cfg = xgb_feat_cfg.get(rn, xgb_feat_cfg['neutral'])
        feats = [c for c in cfg if c in last_day.columns]
        if not feats: continue
        X_r = bt.xgb_scalers[rn].transform(last_day[feats].fillna(0).values)
        last_day[f'xgb_{rn}'] = model.predict(X_r).astype(np.float32)

    # Fusion predictions
    dl_names = list(bt.dl_models.keys())
    xgb_names = [f'xgb_{rn}' for rn in bt.xgb_models]
    has_dl = [f'dl_{n}' for n in dl_names if f'dl_{n}' in last_day.columns]
    has_xgb = [c for c in xgb_names if c in last_day.columns]

    if bt.fusion_model and (has_dl or has_xgb):
        vals = np.zeros(n, dtype=np.float32)
        for k in range(n):
            sc = {}
            for name in dl_names:
                col = f'dl_{name}'
                if col in last_day.columns:
                    sc[name] = float(last_day[col].iloc[k])
            for rn in xgb_names:
                if rn in last_day.columns:
                    sc[rn] = float(last_day[rn].iloc[k])
            if sc:
                vals[k] = bt.fusion_model.fuse_predictions(sc, regime_idx=np.array([reg_idx]))
        last_day['Prediction'] = vals
    else:
        pred_cols = has_dl + has_xgb
        if pred_cols:
            last_day['Prediction'] = last_day[pred_cols].mean(axis=1).fillna(0).values
        else:
            last_day['Prediction'] = 0.0

    # === 市场周期感知的仓位管理 ===
    last_day['name'] = last_day['code'].map(SECTOR_MAP)

    # 检测当前市场周期
    cycle_raw = detect_market_cycle(market_df.loc[:last_dt])
    cycle_cc = CycleController()
    current_cycle = cycle_cc.update(last_dt, cycle_raw)
    cycle_cfg = CYCLE_CONFIG.get(current_cycle, CYCLE_CONFIG['range'])

    dynamic_top_k = cycle_cfg['top_n']
    dynamic_pool = max(dynamic_top_k * 3, TOP_K * 2)  # 候选池至少为top_k的3倍
    position_pct = cycle_cfg['position_pct']

    print(f"\n  当前市场周期: [{current_cycle.upper()}] 仓位={position_pct*100:.0f}% TOP_K={dynamic_top_k}")

    selected_df = _select_affordable_topk(
        last_day, loaded, top_k=dynamic_top_k, candidate_pool=dynamic_pool
    )

    # 计算实际持仓 (周期感知仓位)
    portfolio, cash_left = _calculate_portfolio(selected_df, loaded, position_pct=position_pct)

    return portfolio, selected_df, daily, last_dt, cash_left, current_cycle


def push_to_feishu(portfolio, daily, last_dt, cash_left, current_cycle='range'):
    """推送虚拟盘交易指令到飞书"""
    import numpy as np

    cycle_cfg = CYCLE_CONFIG.get(current_cycle, CYCLE_CONFIG['range'])

    # 统计
    total_shares = portfolio['买入(股)'].astype(int).sum()
    total_commission = portfolio['佣金'].astype(float).sum()
    total_transfer = portfolio['过户费'].astype(float).sum()
    total_invested = portfolio['占用资金'].str.replace(',', '', regex=False).astype(float).sum()

    # 历史绩效
    bench_total = daily['Cum_Benchmark'].iloc[-1] * 100
    strat_total = daily['Cum_Strategy'].iloc[-1] * 100

    # 构建消息
    cycle_emoji = {'bull': '🐂', 'bear': '🐻', 'range': '➡️'}
    lines = []
    lines.append(f"**虚拟盘交易指令**")
    lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**基准日:** {last_dt.date()}")
    lines.append(f"**市场周期:** {cycle_emoji.get(current_cycle, '')} {current_cycle.upper()} | 仓位 {cycle_cfg['position_pct']*100:.0f}% | Top{cycle_cfg['top_n']}")
    lines.append(f"**初始资金:** CNY {INITIAL_CAPITAL:,}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**持仓明细:**")
    lines.append("")
    for _, row in portfolio.iterrows():
        lines.append(f"  **{row['股票代码']}** {row['股票名称']}")
        lines.append(f"    预测得分: {row['预测得分']} | 权重: {row['权重']} | 挂单价: {row['挂单价']}")
        lines.append(f"    买入: {row['买入(手)']}手 ({row['买入(股)']}股) | 资金: CNY {row['占用资金']}")
        lines.append(f"    费用: 佣金CNY {row['佣金']} + 过户费CNY {row['过户费']} = CNY {float(row['佣金'])+float(row['过户费']):.2f}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**成本汇总:**")
    lines.append(f"  总投入: CNY {total_invested:,.0f}")
    lines.append(f"  总佣金: CNY {total_commission:.2f}")
    lines.append(f"  总过户费: CNY {total_transfer:.2f}")
    lines.append(f"  总成本: CNY {total_invested + total_commission + total_transfer:,.0f}")
    lines.append(f"  剩余现金: CNY {cash_left:,.0f}")
    lines.append("")
    lines.append("**卖出预估成本 (仅供参考):**")
    sell_cost_rate = COMMISSION + STAMP_DUTY + TRANSFER_FEE
    sell_est = total_invested * sell_cost_rate
    lines.append(f"  佣金+印花税+过户费 ≈ CNY {sell_est:.2f} ({sell_cost_rate*100:.3f}%)")
    lines.append(f"  预期净收益: 需涨幅 > {(sell_est+total_commission+total_transfer)/total_invested*100:.2f}% 方可盈利")
    lines.append("")
    lines.append(f"**历史回测绩效 (100天):** 策略 {strat_total:+.2f}% vs 基准 {bench_total:+.2f}%")

    text = "\n".join(lines)

    pusher = FeishuPusher(
        webhook_url=os.environ.get('FEISHU_WEBHOOK_URL', ''),
        secret=os.environ.get('FEISHU_WEBHOOK_SECRET', '')
    )
    pusher.send_text(text)

    # CSV保存
    csv_path = os.path.join(OUTPUT, f'virtual_trade_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    portfolio.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  保存至: {csv_path}")


def generate_thss_report(portfolio, selected_df, last_dt, cash_left, current_cycle='range'):
    """
    生成同花顺模拟盘格式的交易指令报告
    格式: 股票代码 | 股票名称 | 操作 | 委托价格 | 委托数量 | 金额(元)
    """
    cycle_cfg = CYCLE_CONFIG.get(current_cycle, CYCLE_CONFIG['range'])
    week_num = last_dt.isocalendar()[1]
    cycle_emoji = {'bull': '🐂', 'bear': '🐻', 'range': '➡️'}
    lines = []
    lines.append("=" * 70)
    lines.append("  同花顺模拟盘 · 每周股票投资权重建议")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  基准日: {last_dt.strftime('%Y-%m-%d')} (第{week_num}周)")
    lines.append(f"  市场周期: {cycle_emoji.get(current_cycle, '')} {current_cycle.upper()} | 仓位 {cycle_cfg['position_pct']*100:.0f}% | Top{cycle_cfg['top_n']}")
    lines.append(f"  初始资金: {INITIAL_CAPITAL:,} 元 | 股票池: 电子科技50股")
    lines.append(f"  可购买性检查: 每只至少买入 {MIN_LOTS} 手 ({MIN_LOTS*LOT_SIZE}股)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("【买入指令】")
    lines.append(f"  {'股票代码':<12} {'股票名称':<10} {'操作':<8} {'委托价':<10} {'数量(股)':<10} {'金额(元)':<12} {'权重':<8}")
    lines.append(f"  {'-'*70}")

    total_amount = 0
    for _, row in portfolio.iterrows():
        amount = float(str(row['占用资金']).replace(',', ''))
        price = float(row['挂单价'])
        shares = int(row['买入(股)'])
        weight = row['权重']
        if shares > 0:
            lines.append(f"  {row['股票代码']:<12} {row['股票名称']:<10} {'买入':<8} {price:<10.2f} {shares:<10} {amount:<12,.2f} {weight:<8}")
            total_amount += amount

    lines.append(f"  {'-'*70}")
    lines.append(f"  {'合计':<51} {total_amount:<12,.2f} {'100%':<8}")
    lines.append(f"  剩余可用资金: {cash_left:,.2f} 元")
    lines.append("")
    lines.append("【权重分配明细】")
    for _, row in portfolio.iterrows():
        shares = int(row['买入(股)'])
        if shares > 0:
            lines.append(f"  {row['股票代码']} {row['股票名称']}: 权重 {row['权重']}, 买入 {row['买入(手)']}手, 现价 {row['现价']}, 预测得分 {row['预测得分']}")

    lines.append("")
    lines.append("【操作提示】")
    lines.append("  1. 打开同花顺模拟盘 -> 买入")
    lines.append("  2. 按以上列表逐只买入 (价格可参考现价)")
    lines.append("  3. 挂单价已含2%缓冲, 确保优先成交")
    lines.append("  4. 若某只股票无法买入, 则跳过, 资金留待下周")
    lines.append("  5. 每周一开盘前执行调仓")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  deep_quant 量化系统 · 多模态融合预测")
    lines.append("  Top5 权重建议 · 可购买性优化")
    lines.append("=" * 70)

    text = "\n".join(lines)

    # 保存报告
    report_path = os.path.join(OUTPUT, f'thss_weekly_picks_{datetime.now().strftime("%Y%m%d")}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n  同花顺报告: {report_path}")

    return text


def main():
    panel, loaded, all_models, scaler, xgb_models, xgb_scalers = load_and_train()
    portfolio, selected_df, daily, last_dt, cash_left, current_cycle = generate_top_picks(
        panel, loaded, all_models, scaler, xgb_models, xgb_scalers)

    cycle_cfg = CYCLE_CONFIG.get(current_cycle, CYCLE_CONFIG['range'])

    print(f"\n{'='*60}")
    print(f"  [{current_cycle.upper()}周期] Top{cycle_cfg['top_n']} 可购买投资组合")
    print(f"{'='*60}")
    cols_to_show = ['股票代码','股票名称','权重','现价','买入(手)','买入(股)','占用资金']
    print(portfolio[cols_to_show].to_string(index=False))
    print(f"\n  剩余现金: {cash_left:,.0f} CNY")
    print(f"  选股过程: 从Top{CANDIDATE_POOL}候选池中筛选可购买标的")

    # 推送飞书
    push_to_feishu(portfolio, daily, last_dt, cash_left, current_cycle)

    # 同花顺格式报告
    thss_report = generate_thss_report(portfolio, selected_df, last_dt, cash_left, current_cycle)
    print(f"\n{'='*70}")
    print("  同花顺模拟盘每周Top5推荐已生成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()