import numpy as np
import pandas as pd


def engineer_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df['Return'] = np.log(df['close'] / df['pre_close'])
    df['Simple_Return'] = df['close'].pct_change()
    df['High_Low_Ratio'] = df['high'] / df['low'] - 1
    df['Close_Open_Ratio'] = df['close'] / df['open'] - 1

    for period in [5, 10, 20, 60]:
        df[f'Ret_{period}'] = df['close'].pct_change(period).shift(1)

    df['Volume_MA_5'] = df['volume'].rolling(5).mean()
    df['Volume_MA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio_5'] = (df['volume'] / df['Volume_MA_5']).shift(1)
    df['Volume_Ratio_20'] = (df['volume'] / df['Volume_MA_20']).shift(1)

    df['Amount_MA_20'] = df['amount'].rolling(20).mean()
    df['Amount_Ratio'] = (df['amount'] / df['Amount_MA_20']).shift(1)

    for window in [5, 10, 20]:
        df[f'Volatility_{window}'] = df['Simple_Return'].rolling(window).std().shift(1)
    df['Volatility_Ratio'] = (df['Volatility_5'] / df['Volatility_20']).shift(1)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)

    for ma_period in [5, 10, 20, 60]:
        df[f'MA_{ma_period}'] = df['close'].rolling(ma_period).mean()
    df['Price_MA_5_Ratio'] = (df['close'] / df['MA_5'] - 1).shift(1)
    df['Price_MA_20_Ratio'] = (df['close'] / df['MA_20'] - 1).shift(1)
    df['Price_MA_60_Ratio'] = (df['close'] / df['MA_60'] - 1).shift(1)
    df['MA_5_20_Cross'] = ((df['MA_5'] > df['MA_20']).astype(int)).shift(1)
    df['MA_20_60_Cross'] = ((df['MA_20'] > df['MA_60']).astype(int)).shift(1)

    df['BB_Middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Position'] = ((df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])).shift(1)

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (df['volume'] * typical_price).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['VWAP_Dist'] = (df['close'] / df['VWAP'] - 1).shift(1)

    df['High_20'] = df['high'].rolling(20).max()
    df['Low_20'] = df['low'].rolling(20).min()
    df['Price_Position'] = ((df['close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])).shift(1)

    df['Daily_Return'] = df['Simple_Return']
    df['Is_Up_Limit'] = (df['Daily_Return'] >= 0.098).astype(int)
    df['Is_Down_Limit'] = (df['Daily_Return'] <= -0.098).astype(int)

    log_return = np.log(df['close'] / df['close'].shift(1)).shift(1)
    df['Momentum_5'] = log_return.rolling(5).sum()
    df['Momentum_10'] = log_return.rolling(10).sum()
    df['Momentum_20'] = log_return.rolling(20).sum()

    df['MA_Cross_Strength'] = (df['MA_5'] / df['MA_20'] - 1).shift(1)
    df['Volume_Price_Trend'] = (df['volume'].rolling(20).mean() * df['close'].rolling(20).mean()).shift(1)
    df['Log_Volume'] = np.log1p(df['volume']).shift(1)
    df['Turnover'] = (df['volume'] * df['close'] / 1e8).shift(1)

    df['Chaikin_MF'] = (
        (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    ).rolling(20).sum().shift(1)

    return df


TECHNICAL_FEATURES = [
    'Ret_5', 'Ret_10', 'Ret_20', 'Ret_60',
    'Volume_Ratio_5', 'Volume_Ratio_20', 'Amount_Ratio',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'RSI', 'Price_MA_5_Ratio', 'Price_MA_20_Ratio', 'Price_MA_60_Ratio',
    'MA_5_20_Cross', 'MA_20_60_Cross',
    'BB_Position', 'VWAP_Dist', 'Price_Position',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'MA_Cross_Strength', 'Log_Volume', 'Turnover',
    'Chaikin_MF', 'High_Low_Ratio', 'Close_Open_Ratio',
]

TECHNICAL_FEATURES_DL = [
    'Ret_5', 'Ret_10', 'Ret_20', 'Ret_60',
    'Volume_Ratio_5', 'Volume_Ratio_20', 'Amount_Ratio',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'RSI', 'Price_MA_5_Ratio', 'Price_MA_20_Ratio', 'Price_MA_60_Ratio',
    'MA_5_20_Cross', 'MA_20_60_Cross',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Log_Volume', 'Turnover', 'Price_Position', 'BB_Position', 'VWAP_Dist',
    'High_Low_Ratio', 'Close_Open_Ratio',
]
