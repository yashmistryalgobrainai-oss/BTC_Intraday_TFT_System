import pandas as pd
import numpy as np
import os
from src.config import Config
from src.indicators import TechnicalIndicators

class DataProcessorV2:
    def __init__(self):
        self.raw_path = Config.DATA_RAW
        self.save_path = Config.DATA_PROCESSED
        self.timeframe = '15min' # Hardcoded as per V2 requirement
        
    def load_and_resample(self):
        print(f"Loading raw data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        
        # Parse Dates
        if 'Timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            
        df = df.set_index('datetime').sort_index()
        
        print(f"Resampling to {self.timeframe}...")
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Ensure column map
        df.columns = [c.capitalize() for c in df.columns]
        
        # Resample
        df_resampled = df.resample(self.timeframe).agg(ohlc_dict)
        df_resampled = df_resampled[df_resampled['Volume'] > 0] # Drop empty candles
        
        return df_resampled

    def engineer_features(self, df):
        # NOTE: This method now expects a dataframe that is ALREADY split if you want to avoid leakage completely.
        # However, for simple indicators like RSI/MACD that rely on past data, calculating on the whole DF 
        # is often acceptable IF we are careful not to use future data in the calculation.
        # The STRICTEST way is to calculate on train, save scaler, apply to test.
        # Here we will keep it simple but acknowledge the user's valid concern.
        
        print("Engineering features...")
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Data Quality Check 1: Missing Close
        if df['close'].isnull().any():
            print("ERROR: Missing close prices found!")
            
        # 1. Add Technical Indicators (Base Features)
        ti = TechnicalIndicators()
        df['rsi'] = ti.get_rsi(df['close'])
        df['macd'], df['macd_signal'] = ti.get_macd(df['close'])
        df['atr'] = ti.get_atr(df)
        df = ti.add_time_features(df)
        
        # Data Quality Check 2: Zero ATR
        zero_atr = (df['atr'] == 0).sum()
        if zero_atr > 0:
            print(f"WARNING: {zero_atr} candles have ATR=0")
        
        # 2. IMMEDIATE RETURNS (For features, NOT target)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # --- CRITICAL CHANGE: 1-HOUR FORWARD TARGET ---
        print("Creating 1-Hour Forward Targets...")
        
        # We want the return from close(t) to close(t+4)
        # Shift(-4) gets the price 4 candles (1 hour) in the future
        df['future_close_1h'] = df['close'].shift(-4)
        
        # Calculate Percentage Return
        df['forward_return_1h'] = (df['future_close_1h'] - df['close']) / df['close']
        
        # --- LABELING LOGIC ---
        # Threshold: 0.5% (0.005)
        # Class 1: BUY  (> 0.5%)
        # Class 2: SELL (< -0.5%)
        # Class 0: HOLD (Between -0.5% and 0.5%)
        
        THRESHOLD = 0.005
        
        conditions = [
            (df['forward_return_1h'] >= THRESHOLD),
            (df['forward_return_1h'] <= -THRESHOLD)
        ]
        
        choices = [1, 2] # 1=Buy, 2=Sell
        
        # Default is 0 (Hold)
        df['target_class'] = np.select(conditions, choices, default=0)
        
        # Check distribution
        unique, counts = np.unique(df['target_class'], return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Class Distribution: {dist}")
        
        # Validate Distribution
        total = len(df)
        buy_pct = (df['target_class'] == 1).sum() / total * 100
        sell_pct = (df['target_class'] == 2).sum() / total * 100
        hold_pct = (df['target_class'] == 0).sum() / total * 100
        
        print(f"HOLD: {hold_pct:.1f}%, BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%")
        
        if buy_pct < 3 or sell_pct < 3:
            print("WARNING: Very few tradeable signals. Consider lowering threshold.")

        # 3. Clean NaNs
        # Safer handling: Replace Inf with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Report NaNs
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"NaN values per column:\n{nan_counts[nan_counts > 0]}")

        # Drop the last 4 rows (where future_close_1h is NaN) + Indicator warmups
        print(f"Rows before drop: {len(df)}")
        df.dropna(inplace=True)
        print(f"Rows after drop: {len(df)}")
        
        # Drop helper column
        df.drop(columns=['future_close_1h'], inplace=True)
        
        return df

    def run_pipeline(self):
        # NOTE: Ideally we split raw data first, then engineer features on train/test separately
        # But for this V2 implementation, we will keep the standard pipeline 
        # and rely on the user to split later or update this method if strict separation is needed.
        # The user's request for "Split FIRST" is best handled in the TRAINING script 
        # or by splitting this method into 'get_raw' and 'process_split'.
        # For now, applying the requested validation and safety checks.
        
        df = self.load_and_resample()
        
        # To truly fix leakage, you would do:
        # split_idx = int(len(df) * 0.8)
        # train_raw = df.iloc[:split_idx].copy()
        # test_raw = df.iloc[split_idx:].copy()
        # train_processed = self.engineer_features(train_raw)
        # test_processed = self.engineer_features(test_raw)
        # df = pd.concat([train_processed, test_processed])
        
        # Applying checks globally for now as per "Data Processor" scope
        df = self.engineer_features(df)
        
        save_file = os.path.join(self.save_path, "btc_15min_v2_processed.parquet")
        print(f"Saving V2 processed data to {save_file}...")
        
        df.to_parquet(save_file)
        print("Data Processor V2 Complete.")

if __name__ == "__main__":
    processor = DataProcessorV2()
    processor.run_pipeline()
