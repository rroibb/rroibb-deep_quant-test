import numpy as np
import pandas as pd


def detect_market_regime(market_df, lookback=60):
    if len(market_df) < lookback:
        return 'neutral'

    if isinstance(market_df['close'], pd.DataFrame):
        close_series = market_df['close'].iloc[:, 0]
    else:
        close_series = market_df['close']

    df_slice = pd.DataFrame({'close': close_series.iloc[-lookback:]})
    returns = df_slice['close'].pct_change().dropna()

    if len(returns) < 20:
        return 'neutral'

    ma5 = df_slice['close'].rolling(5).mean()
    ma20 = df_slice['close'].rolling(20).mean()
    ma60 = df_slice['close'].rolling(60).mean()
    current_price = df_slice['close']
    price_vs_ma20 = (current_price - ma20) / ma20

    bull_aligned = (ma5 > ma20) & (ma20 > ma60) & (current_price > ma5)
    bear_aligned = (ma5 < ma20) & (ma20 < ma60) & (current_price < ma5)
    strong_trend = bull_aligned | bear_aligned

    bull_moderate = (
        (current_price > ma20) & (ma5 > ma20) &
        (price_vs_ma20 > 0.01) &
        (price_vs_ma20 > price_vs_ma20.shift(2).fillna(0))
    )
    bear_moderate = (
        (current_price < ma20) & (ma5 < ma20) &
        (price_vs_ma20 < -0.01) &
        (price_vs_ma20 < price_vs_ma20.shift(2).fillna(0))
    )
    moderate_trend = bull_moderate | bear_moderate

    daily_trend_signal = strong_trend | moderate_trend

    trend_window = 5
    recent_signals = daily_trend_signal.iloc[-trend_window:]
    signal_count = recent_signals.sum()
    signal_ratio = signal_count / trend_window

    is_trend = (signal_ratio >= 0.6) and bool(daily_trend_signal.iloc[-1]) if len(daily_trend_signal) > 0 else False
    strong_recent = bool(strong_trend.iloc[-2:].all()) if len(strong_trend) >= 2 else False

    trend_strength = float(abs(price_vs_ma20).mean())
    autocorr = float(returns.autocorr(lag=1)) if len(returns) > 1 else 0.0

    if is_trend or strong_recent:
        return 'trend'
    if trend_strength > 0.03 and autocorr > 0.03:
        return 'trend'

    near_ma20 = abs(price_vs_ma20) < 0.015
    price_near_ma20 = bool(near_ma20.iloc[-trend_window:].sum() >= (trend_window * 0.6)) if len(near_ma20) >= trend_window else False
    if price_near_ma20 and autocorr < -0.05 and trend_strength < 0.02:
        return 'mean_reversion'

    return 'neutral'


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
