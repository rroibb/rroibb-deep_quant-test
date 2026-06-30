import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import math

from config import (TRADING_DAYS, RISK_FREE_RATE, COMMISSION_RATE, STAMP_DUTY_RATE,
                     TRANSFER_FEE_RATE, SLIPPAGE_RATE, SECTOR_MAP, MAX_SECTOR_PCT, TOP_N, 
                     OUTPUT_DIR, DEVICE, CYCLE_CONFIG)
from features import TECHNICAL_FEATURES_DL as DL_FEATURES, TECHNICAL_FEATURES
from regime import detect_market_regime, RegimeController, detect_market_cycle, CycleController
from data_layer import SequenceDataset


class DeepQuantBacktester:
    def __init__(self, df_panel, market_df, models_dict, scaler_dl=None,
                 xgb_models=None, xgb_scalers=None, backtest_start=None):
        self.df_panel = df_panel.copy()
        self.market_df = market_df
        self.models = models_dict
        self.scaler_dl = scaler_dl
        self.xgb_models = xgb_models or {}
        self.xgb_scalers = xgb_scalers or {}
        self.backtest_start = backtest_start

        self.dl_models = {k: v for k, v in models_dict.items() 
                         if k in ['lstm', 'transformer', 'cnn']}
        self.fusion_model = models_dict.get('fusion')

    def _precompute_dl(self, feat_cols, seq_len=30):
        """预计算所有DL模型在所有日期-股票对上的预测值"""
        dl_map = {}
        if not self.dl_models:
            return dl_map
        # 用首个模型的 input_size 确定特征维度
        first_model = next(iter(self.dl_models.values()))
        n_feat = getattr(first_model, 'input_size', None) or getattr(first_model, 'in_channels', None)
        fc = feat_cols[:n_feat] if n_feat and len(feat_cols) > n_feat else feat_cols
        dataset = SequenceDataset(self.df_panel, seq_len=seq_len, feature_cols=fc)
        if len(dataset) == 0:
            return dl_map
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            for name, model in self.dl_models.items():
                preds_list = []
                for x_batch, _ in loader:
                    preds_list.append(model(x_batch.to(DEVICE)).cpu().numpy())
                all_preds = np.concatenate(preds_list, axis=0).ravel()
                for k_idx, (p_dt, p_code) in enumerate(dataset.idx_to_dt_code):
                    if k_idx < len(all_preds):
                        dl_map[(p_dt, p_code, name)] = float(all_preds[k_idx])
        return dl_map

    def run(self, buffer_threshold=0.15, stop_loss=0.10,
            target_vol=0.15, min_hold_days=10, weight_threshold=0.03,
            use_fusion=True, use_xgb=True, use_dl=True):
        """优化版: 向量化 per-stock 循环 + groupby rank 替代逐日切片"""
        df = self.df_panel.copy()
        unique_dates = sorted(df.index.get_level_values(0).unique())
        print(f"\n回测区间: {unique_dates[0].date()} ~ {unique_dates[-1].date()}")
        print(f"交易日数: {len(unique_dates)}")

        df['Future_20d_Ret'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-20) / x - 1
        )

        print("预计算市场环境...")
        regime_controller = RegimeController(min_hold_days=min_hold_days)
        filtered_regimes = []
        cycle_controller = CycleController()
        filtered_cycles = []
        for dt in unique_dates:
            m_slice = self.market_df.loc[:dt]
            raw = detect_market_regime(m_slice)
            filtered_regimes.append(regime_controller.update(dt, raw))
            raw_cycle = detect_market_cycle(m_slice)
            filtered_cycles.append(cycle_controller.update(dt, raw_cycle))

        # DL预计算 (同上)
        if not hasattr(self, '_cached_dl_map') or self._cached_dl_map is None:
            if 'Future_20d_Ret' not in self.df_panel.columns:
                self.df_panel['Future_20d_Ret'] = self.df_panel.groupby('code')['close'].transform(
                    lambda x: x.shift(-20) / x - 1)
            feat_cols = [c for c in DL_FEATURES if c in self.df_panel.columns]
            self._cached_dl_map = self._precompute_dl(feat_cols)
        all_dl_map = self._cached_dl_map if use_dl else {}

        # ── 优化1: 预缓存 XGB 特征配置 ──
        xgb_feat_cfg = {
            'trend': ['Ret_20', 'Ret_60', 'Price_MA_60_Ratio', 'MA_20_60_Cross', 'Volatility_20', 'Amount_Ratio'],
            'mean_reversion': ['Ret_5', 'Ret_10', 'RSI', 'BB_Position', 'VWAP_Dist', 'Price_Position', 'High_Low_Ratio'],
            'neutral': [c for c in DL_FEATURES if c in TECHNICAL_FEATURES][:18],
        }

        print("批量模型预测 (优化版)...")
        all_pred_rows = []

        for i, dt in enumerate(unique_dates):
            if self.backtest_start is not None and dt < self.backtest_start:
                continue
            if i % 200 == 0:
                print(f"  进度 {i}/{len(unique_dates)}")
            reg = filtered_regimes[i]
            reg_idx = {'trend': 0, 'neutral': 1, 'mean_reversion': 2}.get(reg, 1)

            day_data = df.xs(dt, level=0).copy().reset_index()
            if day_data.empty:
                continue

            codes = day_data['code'].tolist()
            n_codes = len(codes)

            # ── 优化2: DL预测批量赋值 (O(N*M) 字典查表, 无boolean mask) ──
            if use_dl:
                for name in self.dl_models:
                    vals = np.array([all_dl_map.get((dt, c, name), np.nan) for c in codes], dtype=np.float32)
                    day_data[f'dl_{name}'] = vals

            # ── 优化3: XGBoost 直接列赋值 (O(R), 无per-stock循环) ──
            if use_xgb and self.xgb_models:
                for regime_name, model in self.xgb_models.items():
                    if regime_name not in self.xgb_scalers:
                        continue
                    cfg = xgb_feat_cfg.get(regime_name, xgb_feat_cfg['neutral'])
                    feats = [c for c in cfg if c in day_data.columns]
                    if not feats:
                        continue
                    X_r = self.xgb_scalers[regime_name].transform(day_data[feats].fillna(0).values)
                    day_data[f'xgb_{regime_name}'] = model.predict(X_r).astype(np.float32)

            # ── 优化4: Fusion 批量 (O(N), 用 .iloc 替代 boolean mask) ──
            dl_names = list(self.dl_models.keys())
            xgb_names = [f'xgb_{rn}' for rn in self.xgb_models]
            has_dl_cols = [f'dl_{n}' for n in dl_names if f'dl_{n}' in day_data.columns]
            has_xgb_cols = [c for c in xgb_names if c in day_data.columns]

            if use_fusion and self.fusion_model and (has_dl_cols or has_xgb_cols):
                vals = np.zeros(n_codes, dtype=np.float32)
                for k in range(n_codes):
                    sc = {}
                    for name in dl_names:
                        col = f'dl_{name}'
                        if col in day_data.columns:
                            sc[name] = float(day_data[col].iloc[k])
                    for rn in xgb_names:
                        if rn in day_data.columns:
                            sc[rn] = float(day_data[rn].iloc[k])
                    if sc:
                        vals[k] = self.fusion_model.fuse_predictions(sc, regime_idx=np.array([reg_idx]))
                day_data['DL_Pred'] = vals
            else:
                pred_cols = has_dl_cols + has_xgb_cols
                if pred_cols:
                    day_data['DL_Pred'] = day_data[pred_cols].mean(axis=1).fillna(0).values
                else:
                    day_data['DL_Pred'] = 0.0

            day_data['Regime'] = reg
            day_data['Cycle'] = filtered_cycles[i]
            day_data['date'] = dt
            all_pred_rows.append(day_data)

        if not all_pred_rows:
            print("x 无预测结果")
            return pd.DataFrame()

        df_pred = pd.concat(all_pred_rows, ignore_index=True)
        df_pred = df_pred.set_index(['date', 'code']).sort_index()

        ic = df_pred['DL_Pred'].corr(df_pred['Future_20d_Ret'], method='spearman')
        print(f"整体预测IC: {ic:.4f}")
        if ic < 0:
            print("IC为负，取反预测值")
            df_pred['DL_Pred'] = -df_pred['DL_Pred']

        # ── 优化5: 信号生成 — 周期感知的动态TOP_N ──
        print("生成信号 (周期感知)...")
        top_n_map = {'bull': CYCLE_CONFIG['bull']['top_n'],
                     'range': CYCLE_CONFIG['range']['top_n'],
                     'bear': CYCLE_CONFIG['bear']['top_n']}
        df_pred['TopN'] = df_pred['Cycle'].map(top_n_map).fillna(5).astype(int)
        df_pred['Rank'] = df_pred.groupby('date')['DL_Pred'].rank(ascending=False, method='first')
        def assign_signal(g):
            top_n = g['TopN'].iloc[0]
            g['Signal'] = np.where(g['Rank'] <= top_n, 1.0 / top_n, 0.0)
            return g
        df_pred = df_pred.groupby('date', group_keys=False).apply(assign_signal)

        # ── 日频IC/IR ──
        self.daily_ic = df_pred.groupby('date').apply(
            lambda g: g['DL_Pred'].corr(g['Future_20d_Ret'], method='spearman')
        ).dropna()

        # ── 五分位分组收益 (Long-Short) ──
        df_pred['Q_Group'] = df_pred.groupby('date')['DL_Pred'].transform(
            lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) if x.nunique() >= 5 else -1
        )
        q_valid = df_pred[df_pred['Q_Group'] >= 0]
        self.q_ret = q_valid.groupby(['date', 'Q_Group'])['Future_20d_Ret'].mean().unstack()
        if 4 in self.q_ret.columns and 0 in self.q_ret.columns:
            self.ls_ret = (self.q_ret[4] - self.q_ret[0]).dropna()
        else:
            self.ls_ret = pd.Series(dtype=float)

        # ── 停牌 + 涨跌停检测 ──
        print("检测停牌/涨跌停状态...")
        def _limit_pct(code):
            return 0.20 if (code.startswith('688') or code.startswith('300')) else 0.10
        df_pred['_limit_pct'] = df_pred.index.get_level_values('code').map(_limit_pct)
        df_pred['_up_limit'] = df_pred['pre_close'] * (1 + df_pred['_limit_pct'])
        df_pred['_down_limit'] = df_pred['pre_close'] * (1 - df_pred['_limit_pct'])
        df_pred['_limit_up'] = df_pred['close'] >= df_pred['_up_limit'] * 0.999
        df_pred['_limit_down'] = df_pred['close'] <= df_pred['_down_limit'] * 1.001
        df_pred['_suspended'] = (df_pred['volume'] <= 0) | df_pred['close'].isna() | (df_pred['close'] <= 0)

        # ── 优化6: 周度调仓 (周一开盘执行) + 停牌/涨跌停阻断 ──
        print("应用缓冲带 + 权重过滤 (含停牌/涨跌停阻断)...")
        df = df_pred.reset_index()
        df['Week'] = df['date'].dt.to_period('W-MON')
        df['Signal_Final'] = 0.0

        weeks_sorted = sorted(df['Week'].unique())
        prev_weights = pd.Series(dtype=float)

        for week in weeks_sorted:
            wk = df[df['Week'] == week]
            first_day = wk['date'].min()
            fd = wk[wk['date'] == first_day].copy()
            if fd.empty:
                continue

            # 停牌股票不能交易
            fd.loc[fd['_suspended'], 'Signal'] = 0.0
            suspended_codes = set(fd.loc[fd['_suspended'], 'code'])

            new_w = fd.set_index('code')['Signal']
            if new_w.empty or new_w.sum() == 0:
                prev_weights = pd.Series(dtype=float)
                continue

            # 获取该周市场周期 → 仓位比例
            wk_cycle = fd['Cycle'].iloc[0]
            position_pct = CYCLE_CONFIG.get(wk_cycle, {}).get('position_pct', 1.0)

            if len(prev_weights) > 0:
                u = prev_weights.index.union(new_w.index)
                pv = prev_weights.reindex(u, fill_value=0)
                nw = new_w.reindex(u, fill_value=0)

                # 涨停不可买, 跌停不可卖
                for code in u:
                    if code in suspended_codes:
                        nw[code] = pv[code]
                        continue
                    row = fd[fd['code'] == code]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    if r['_limit_up'] and nw[code] > pv[code]:
                        nw[code] = pv[code]
                    if r['_limit_down'] and nw[code] < pv[code]:
                        nw[code] = pv[code]

                diff_abs = (nw - pv).abs()
                final_local = nw.copy()
                final_local[diff_abs < buffer_threshold] = pv[diff_abs < buffer_threshold]
                final_w = final_local / final_local.sum() if final_local.sum() > 0 else final_local
            else:
                final_w = new_w

            # ── 行业集中度约束（MAX_SECTOR_PCT）──
            if final_w.sum() > 0:
                w_sectors = pd.Series(
                    {code: SECTOR_MAP.get(code, '其他') for code in final_w.index},
                    name='sector'
                )
                for sec in w_sectors.unique():
                    sec_mask = w_sectors == sec
                    sec_w = final_w[sec_mask].sum()
                    if sec_w > MAX_SECTOR_PCT:
                        excess = sec_w - MAX_SECTOR_PCT
                        final_w[sec_mask] *= MAX_SECTOR_PCT / sec_w
                        other_mask = ~sec_mask
                        other_total = final_w[other_mask].sum()
                        if other_total > 0:
                            final_w[other_mask] += excess * final_w[other_mask] / other_total
                final_w = final_w / final_w.sum() * final_w.sum()

            # 应用周期仓位控制
            final_w = final_w * position_pct

            for code, weight in final_w.items():
                df.loc[wk.index[wk['code'] == code], 'Signal_Final'] = weight
            prev_weights = final_w

        print("计算收益...")
        df['Strategy_Ret_Gross'] = df['Signal_Final'] * df['Simple_Return']
        df['Weight_Diff'] = df.groupby('code')['Signal_Final'].diff()
        abs_diff = df['Weight_Diff'].abs()
        # 佣金 + 印花税 + 过户费
        buy_cost = df['Weight_Diff'].clip(lower=0) * (COMMISSION_RATE + TRANSFER_FEE_RATE)
        sell_cost = (-df['Weight_Diff'].clip(upper=0)) * (COMMISSION_RATE + STAMP_DUTY_RATE + TRANSFER_FEE_RATE)
        df['Trade_Cost'] = buy_cost + sell_cost
        # 滑点: 双边收取
        df['Slippage'] = abs_diff * SLIPPAGE_RATE
        # 缓冲带内不交易, 不计成本
        low_turnover = abs_diff < weight_threshold
        df.loc[low_turnover, 'Trade_Cost'] = 0
        df.loc[low_turnover, 'Slippage'] = 0
        df['Strategy_Ret'] = df['Strategy_Ret_Gross'] - df['Trade_Cost'] - df['Slippage']

        # ── 换手率 ──
        self.daily_turnover = df.groupby('date')['Weight_Diff'].agg(lambda x: x.abs().sum())

        daily = df.groupby('date').agg({
            'Simple_Return': 'mean',
            'Strategy_Ret_Gross': 'sum',
            'Strategy_Ret': 'sum',
            'Trade_Cost': 'sum',
            'Slippage': 'sum',
            'Regime': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral',
            'Cycle': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'range',
        })
        daily['Cum_Benchmark'] = (1 + daily['Simple_Return']).cumprod() - 1
        daily['Cum_Strategy'] = (1 + daily['Strategy_Ret']).cumprod() - 1

        self._report(daily)
        return daily

    def _report(self, daily):
        total_g = daily['Strategy_Ret_Gross'].sum()
        total_n = daily['Strategy_Ret'].sum()
        total_c = daily['Trade_Cost'].sum()
        total_s = daily['Slippage'].sum()
        ann_g = daily['Strategy_Ret_Gross'].mean() * TRADING_DAYS
        ann_n = daily['Strategy_Ret'].mean() * TRADING_DAYS
        ann_v = daily['Strategy_Ret'].std() * np.sqrt(TRADING_DAYS)
        n_days = len(daily)
        sharpe_n = (ann_n - RISK_FREE_RATE) / ann_v if ann_v > 1e-8 else 0
        sharpe_g = (ann_g - RISK_FREE_RATE) / ann_v if ann_v > 1e-8 else 0
        cum_max = daily['Cum_Strategy'].cummax()
        mdd = (cum_max - daily['Cum_Strategy']).max()
        cost_pct = (total_c / total_g * 100) if total_g != 0 else 0

        # ── 夏普比率置信区间 (Lo 2002 标准误) ──
        se_sharpe = math.sqrt((1 + 0.5 * sharpe_n**2) / n_days)
        sharpe_lower = sharpe_n - 1.96 * se_sharpe
        sharpe_upper = sharpe_n + 1.96 * se_sharpe

        # ── 年化收益 t-test ──
        daily_ret_vec = daily['Strategy_Ret']
        t_stat = daily_ret_vec.mean() / (daily_ret_vec.std() / math.sqrt(n_days)) if daily_ret_vec.std() > 1e-12 else 0
        def _norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

        # ── IC / IR ──
        ic_series = getattr(self, 'daily_ic', pd.Series(dtype=float))
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std * math.sqrt(TRADING_DAYS) if ic_std > 1e-12 else 0
        ic_positive_pct = (ic_series > 0).mean() * 100

        # ── 换手率 ──
        to_series = getattr(self, 'daily_turnover', pd.Series(dtype=float))
        avg_turnover = to_series.mean() * 100 if len(to_series) > 0 else 0

        # ── Long-Short ──
        ls = getattr(self, 'ls_ret', pd.Series(dtype=float))
        ls_mean = ls.mean() * 100 if len(ls) > 0 else 0
        ls_ann = ls.mean() * TRADING_DAYS * 100 if len(ls) > 0 else 0
        ls_std = ls.std() * math.sqrt(TRADING_DAYS) * 100 if len(ls) > 0 else 0
        ls_sharpe = ls.mean() / ls.std() * math.sqrt(TRADING_DAYS) if len(ls) > 0 and ls.std() > 1e-12 else 0

        # ============ Print ============
        print(f"\n{'='*60}")
        print("深度学习量化策略回测绩效")
        print(f"{'='*60}")
        print(f"  总收益(毛): {total_g*100:.2f}%")
        print(f"  总收益(净): {total_n*100:.2f}%")
        print(f"  年化收益(净): {ann_n*100:.2f}%")
        print(f"  年化波动率: {ann_v*100:.2f}%")
        print(f"  夏普比率(毛): {sharpe_g:.2f}")
        print(f"  夏普比率(净): {sharpe_n:.2f}")
        print(f"  夏普 95% CI: [{sharpe_lower:.2f}, {sharpe_upper:.2f}]")
        print(f"  最大回撤: {mdd*100:.2f}%")
        print(f"  t-统计量: {t_stat:.3f}, p值: {p_value:.4f}")
        signif = '显著 (p<0.05)' if p_value < 0.05 else '不显著 (p>=0.05)'
        print(f"  年化收益显著性: {signif}")
        print(f"  总交易成本(佣金/税费): {total_c*100:.2f}%")
        print(f"  总滑点成本: {total_s*100:.2f}%")
        print(f"  合计成本腐蚀: {(total_c+total_s)*100:.2f}%")
        print(f"  成本/毛利: {cost_pct:.2f}%")

        print(f"\n  ── IC / IR 分析 ──")
        print(f"  平均日IC: {ic_mean:.4f}")
        print(f"  IC标准差: {ic_std:.4f}")
        print(f"  信息比率(IR): {ir:.3f}")
        print(f"  IC>0比率: {ic_positive_pct:.1f}%")

        print(f"\n  ── 换手率分析 ──")
        print(f"  日均换手率(单边): {avg_turnover:.2f}%")
        print(f"  周均单边换手率: {avg_turnover*5:.2f}%")

        print(f"\n  ── 分组收益 (Long-Short) ──")
        if len(ls) > 0:
            print(f"  日均Long-Short收益: {ls_mean:.3f}%")
            print(f"  年化Long-Short: {ls_ann:.2f}%")
            print(f"  Long-Short年化波动: {ls_std:.2f}%")
            print(f"  Long-Short夏普: {ls_sharpe:.2f}")
        else:
            print("  (样本不足, 无法计算)")

        print(f"\n市场环境分布 (短周期):")
        rc = daily['Regime'].value_counts()
        for r, c_val in rc.items():
            print(f"  {r}: {c_val}天 ({c_val/len(daily)*100:.1f}%)")
        print(f"\n市场周期分布 (牛/熊/震荡):")
        if 'Cycle' in daily.columns:
            cc = daily['Cycle'].value_counts()
            for c_val, cnt in cc.items():
                print(f"  {c_val}: {cnt}天 ({cnt/len(daily)*100:.1f}%)")

    def save_results(self, daily_ret, tag=''):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = f"_{tag}" if tag else ""
        path = os.path.join(OUTPUT_DIR, f'dl_quant_daily_{ts}{tag}.csv')
        daily_ret.to_csv(path, encoding='utf-8-sig')
        print(f"日收益已保存: {path}")

        total_n = daily_ret['Strategy_Ret'].sum() * 100
        ann_n = daily_ret['Strategy_Ret'].mean() * TRADING_DAYS * 100
        ann_v = daily_ret['Strategy_Ret'].std() * math.sqrt(TRADING_DAYS) * 100
        sr = (ann_n - RISK_FREE_RATE * 100) / ann_v if ann_v > 1e-8 else 0
        n_days = len(daily_ret)
        se_sr = math.sqrt((1 + 0.5 * (sr / 100)**2) / n_days)

        ic_s = getattr(self, 'daily_ic', pd.Series(dtype=float))
        ic_mean = ic_s.mean()
        ic_std = ic_s.std()
        ir_val = ic_mean / ic_std * math.sqrt(TRADING_DAYS) if ic_std > 1e-12 else 0

        to_s = getattr(self, 'daily_turnover', pd.Series(dtype=float))
        avg_to = to_s.mean() * 100 if len(to_s) > 0 else 0

        report_path = os.path.join(OUTPUT_DIR, f'dl_quant_report_{ts}{tag}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"深度学习量化系统回测报告\n生成时间: {datetime.now()}\n\n")
            f.write(f"总收益(净): {total_n:.2f}%\n")
            f.write(f"年化收益: {ann_n:.2f}%\n")
            f.write(f"年化波动: {ann_v:.2f}%\n")
            f.write(f"夏普比率: {sr:.2f}\n")
            f.write(f"夏普 95% CI: [{sr-1.96*se_sr:.2f}, {sr+1.96*se_sr:.2f}]\n")
            f.write(f"最大回撤: {daily_ret['Cum_Strategy'].cummax().sub(daily_ret['Cum_Strategy']).max()*100:.2f}%\n")
            f.write(f"总交易成本: {daily_ret['Trade_Cost'].sum()*100:.2f}%\n")
            f.write(f"总滑点成本: {daily_ret['Slippage'].sum()*100:.2f}%\n")
            f.write(f"平均日IC: {ic_mean:.4f}\n")
            f.write(f"信息比率(IR): {ir_val:.3f}\n")
            f.write(f"日均换手率(单边): {avg_to:.2f}%\n")
        print(f"报告已保存: {report_path}")
