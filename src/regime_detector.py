import pandas as pd
import numpy as np

class RegimeDetector:
    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculates ADX (Trend Strength).
        ADX > 25: Strong Trend
        ADX < 20: Weak/Choppy Market
        """
        df = df.copy()
        
        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smooth
        df['atr_smooth'] = df['tr'].rolling(window=period).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr_smooth'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr_smooth'])
        
        # DX & ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df['adx']

    @staticmethod
    def get_regime(row):
        """
        Returns the Regime Classification.
        """
        adx = row['adx']
        volatility = row['atr'] / row['close'] # Percentage volatility
        
        if adx > 25:
            return "TRENDING" # Safe to use lower threshold
        elif adx < 20:
            return "NOISE"    # DANGEROUS! High failure rate.
        else:
            return "STABLE"   # Normal trading