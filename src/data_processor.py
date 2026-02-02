import pandas as pd
import numpy as np
from src.config import Config
from src.indicators import TechnicalIndicators

class DataProcessor:
    def __init__(self):
        self.raw_path = Config.DATA_RAW
        self.save_path = Config.DATA_PROCESSED
        self.timeframe = Config.TIMEFRAME

    def load_and_resample(self):
        print(f"Loading raw data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        
        # 1. Handle Timestamp safely
        if 'Timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            # Fallback: assume first column is time
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            
        df = df.set_index('datetime')
        df = df.sort_index()

        print(f"Resampling to {self.timeframe}...")
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Standardize column names before resampling
        df.columns = [c.capitalize() for c in df.columns] 
        
        df_resampled = df.resample(self.timeframe).agg(ohlc_dict)
        
        # 2. DROP invalid rows immediately (Zero Volume or NaNs from empty bins)
        df_resampled = df_resampled[df_resampled['Volume'] > 0]
        df_resampled.dropna(inplace=True)
        
        return df_resampled

    def engineer_features(self, df):
        print("Engineering features...")
        
        df.columns = [c.lower() for c in df.columns]
        
        # Technical Indicators
        ti = TechnicalIndicators()
        df['rsi'] = ti.get_rsi(df['close'])
        df['macd'], df['macd_signal'] = ti.get_macd(df['close'])
        df['atr'] = ti.get_atr(df)
        df = ti.add_time_features(df)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Safe Log Returns (Handle division by zero / negative prices)
        # We use a small epsilon or fillna to prevent -inf
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Target: Shift -1 (Predict next candle)
        df['target_return'] = df['log_returns'].shift(-1)
        
        # --- CRITICAL FIX: CLEAN INF/NAN ---
        print("Cleaning Infinite and NaN values...")
        
        # 1. Replace inf/-inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Drop NaNs
        before_len = len(df)
        df.dropna(inplace=True)
        after_len = len(df)
        
        print(f"Dropped {before_len - after_len} rows containing NaN/Inf values.")
        
        return df

    def run_pipeline(self):
        df = self.load_and_resample()
        df = self.engineer_features(df)
        
        save_file = f"{self.save_path}/btc_{self.timeframe}_processed.parquet"
        print(f"Saving processed data to {save_file}...")
        df.to_parquet(save_file)
        print("Data processing complete.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run_pipeline()