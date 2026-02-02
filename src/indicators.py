import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def get_rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_macd(series, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def get_atr(df, period=14):
        """Average True Range for Dynamic Stop Loss"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def add_time_features(df):
        """
        Cyclical encoding for time. 
        Machines don't understand that Hour 23 and Hour 0 are close.
        Sin/Cos transforms fix this.
        """
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_of_week'] = df.index.dayofweek
        return df