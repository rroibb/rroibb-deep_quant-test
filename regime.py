import numpy as np
import pandas as pd


def _ols_trend(close_values):
    """对 log(close) 做 OLS 回归, 返回 slope(年化%), R², t-stat, ann_vol"""
    n = len(close_values)
    if n < 5:
        return 0.0, 0.0, 0.0, 0.0

    log_p = np.log(np.asarray(close_values, dtype=float))
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    log_p_mean = log_p.mean()

    dx = x - x_mean
    dy = log_p - log_p_mean
    ssxx = np.sum(dx * dx)
    ssxy = np.sum(dx * dy)

    slope = ssxy / ssxx if ssxx > 1e-12 else 0.0
    intercept = log_p_mean - slope * x_mean

    predicted = intercept + slope * x
    residuals = log_p - predicted
    rss = np.sum(residuals ** 2)
    tss = np.sum(dy * dy)
    r2 = 1.0 - rss / tss if tss > 1e-12 else 0.0

    mse = rss / max(n - 2, 1)
    se_slope = np.sqrt(mse / ssxx) if ssxx > 1e-12 else 0.0
    t_stat = slope / se_slope if se_slope > 1e-12 else 0.0

    ann_slope = slope * 252 * 100

    returns = np.diff(log_p)
    ann_vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0.0

    return ann_slope, r2, t_stat, ann_vol


def _extract_close(market_df):
    if isinstance(market_df, pd.Series):
        return market_df.values
    if isinstance(market_df, pd.DataFrame):
        if 'close' in market_df.columns:
            c = market_df['close']
        else:
            c = market_df.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c.values
    return np.asarray(market_df, dtype=float)


def detect_market_regime(market_df, lookback=60):
    """
    改进版市场状态检测:
      - OLS 对数价格趋势 (年化斜率, R², t-stat)
      - 波动率水平
      - 收益自相关
    返回: 'trend', 'mean_reversion', 'neutral'

    注意: market_df 必须是截至当前时刻的切片, 函数内部取尾端 lookback 条.
    """
    if not isinstance(market_df, (pd.Series, pd.DataFrame)) or len(market_df) < 20:
        return 'neutral'

    close_values = _extract_close(market_df)
    if len(close_values) < lookback:
        return 'neutral'
    close_values = close_values[-lookback:]

    ann_slope, r2, t_stat, ann_vol = _ols_trend(close_values)
    n = len(close_values)
    returns = np.diff(np.log(close_values))
    autocorr = pd.Series(returns).autocorr(lag=1) if len(returns) > 1 else 0.0

    rv_20 = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else ann_vol

    has_trend = r2 > 0.30 and abs(t_stat) > 1.8
    strong_trend = r2 > 0.50 and abs(t_stat) > 2.5

    if strong_trend and abs(ann_slope) > 5:
        return 'trend'
    if has_trend and abs(ann_slope) > 3:
        return 'trend'

    is_mr = r2 < 0.15 and autocorr < -0.08 and rv_20 < 35
    if is_mr:
        return 'mean_reversion'

    if rv_20 > 50 and abs(ann_slope) > 8:
        return 'trend'

    return 'neutral'


def detect_regime_ex(market_df, lookback=60):
    """
    扩展版市场状态检测, 返回结构化信息.
    返回 dict:
      - regime: 'trend' / 'mean_reversion' / 'neutral'
      - trend_pct: 年化趋势 (%)
      - r2: 趋势拟合优度
      - t_stat: 趋势 t 统计量
      - vol_pct: 年化波动率 (%)
      - autocorr: 1 阶自相关
    """
    result = {
        'regime': 'neutral', 'trend_pct': 0.0, 'r2': 0.0,
        't_stat': 0.0, 'vol_pct': 0.0, 'autocorr': 0.0,
    }
    if not isinstance(market_df, (pd.Series, pd.DataFrame)) or len(market_df) < 20:
        return result

    close_values = _extract_close(market_df)
    if len(close_values) < lookback:
        return result
    close_values = close_values[-lookback:]

    ann_slope, r2, t_stat, ann_vol = _ols_trend(close_values)
    returns = np.diff(np.log(close_values))
    autocorr = pd.Series(returns).autocorr(lag=1) if len(returns) > 1 else 0.0
    rv_20 = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else ann_vol

    result['trend_pct'] = round(ann_slope, 2)
    result['r2'] = round(r2, 4)
    result['t_stat'] = round(t_stat, 3)
    result['vol_pct'] = round(ann_vol, 2)
    result['autocorr'] = round(autocorr, 4)

    has_trend = r2 > 0.30 and abs(t_stat) > 1.8
    strong_trend = r2 > 0.50 and abs(t_stat) > 2.5
    is_mr = r2 < 0.15 and autocorr < -0.08 and rv_20 < 35

    if strong_trend and abs(ann_slope) > 5:
        result['regime'] = 'trend'
    elif has_trend and abs(ann_slope) > 3:
        result['regime'] = 'trend'
    elif is_mr:
        result['regime'] = 'mean_reversion'
    elif rv_20 > 50 and abs(ann_slope) > 8:
        result['regime'] = 'trend'

    return result


class RegimeController:
    def __init__(self, min_hold_days=10, max_hold_days=45):
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.current_regime = None
        self.regime_start_date = None
        self.days_in_regime = 0
        self.switch_log = []

    def update(self, date, raw_regime):
        if self.current_regime is None:
            self.current_regime = raw_regime
            self.regime_start_date = date
            self.days_in_regime = 0
            return self.current_regime

        self.days_in_regime += 1

        if raw_regime == self.current_regime:
            return self.current_regime

        min_hold_passed = self.days_in_regime >= self.min_hold_days
        max_hold_reached = self.days_in_regime >= self.max_hold_days

        if min_hold_passed or max_hold_reached:
            self.switch_log.append({
                'date': str(date), 'from': self.current_regime,
                'to': raw_regime, 'held_days': self.days_in_regime,
                'reason': 'max_hold' if max_hold_reached else 'min_hold'
            })
            self.current_regime = raw_regime
            self.regime_start_date = date
            self.days_in_regime = 0
            return self.current_regime
        else:
            return self.current_regime


def detect_market_cycle(market_df, lookback=200):
    """
    改进版长周期市场状态检测 (bull/bear/range).
    基于 OLS 趋势 + MA200 偏离 + 回撤深度.
    """
    if not isinstance(market_df, (pd.Series, pd.DataFrame)) or len(market_df) < 120:
        return 'range'

    close_values = _extract_close(market_df)
    if len(close_values) < lookback:
        return 'range'
    close_values = close_values[-lookback:]

    ann_slope, r2, t_stat, ann_vol = _ols_trend(close_values)
    current_price = close_values[-1]

    ma200 = np.mean(close_values)
    price_vs_ma200 = (current_price - ma200) / ma200 if ma200 > 0 else 0.0

    high_252 = np.max(close_values)
    drawdown = (high_252 - current_price) / high_252 if high_252 > 0 else 0.0

    returns = np.diff(np.log(close_values))
    vol_20 = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else 0.0
    vol_60 = ann_vol
    vol_ratio = vol_20 / vol_60 if vol_60 > 5 else 1.0

    is_bull = (t_stat > 1.5 and price_vs_ma200 > -0.03 and drawdown < 0.12 and vol_ratio < 1.4)
    is_bear = (t_stat < -1.5) or (price_vs_ma200 < -0.08) or (drawdown > 0.18) or (vol_ratio > 1.6)

    if is_bull and not is_bear:
        return 'bull'
    if is_bear:
        return 'bear'
    return 'range'


class CycleController:
    def __init__(self, min_hold_days=20, max_hold_days=120):
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.current_cycle = None
        self.cycle_start_date = None
        self.days_in_cycle = 0
        self.switch_log = []

    def update(self, date, raw_cycle):
        if self.current_cycle is None:
            self.current_cycle = raw_cycle
            self.cycle_start_date = date
            self.days_in_cycle = 0
            return self.current_cycle

        self.days_in_cycle += 1

        if raw_cycle == self.current_cycle:
            return self.current_cycle

        min_hold_passed = self.days_in_cycle >= self.min_hold_days
        max_hold_reached = self.days_in_cycle >= self.max_hold_days

        if min_hold_passed or max_hold_reached:
            self.switch_log.append({
                'date': str(date), 'from': self.current_cycle,
                'to': raw_cycle, 'held_days': self.days_in_cycle,
                'reason': 'max_hold' if max_hold_reached else 'min_hold'
            })
            self.current_cycle = raw_cycle
            self.cycle_start_date = date
            self.days_in_cycle = 0
            return self.current_cycle
        else:
            return self.current_cycle
