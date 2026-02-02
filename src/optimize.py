import pandas as pd
import numpy as np
import xgboost as xgb
import os
from src.config import Config

# Re-define logic to avoid import errors
def add_lag_features(df):
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['rolling_volatility'] = df['returns'].rolling(window=12).std()
    return df

class Optimizer:
    def __init__(self, model_path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.capital_base = 10000
        self.fee = 0.001

    def load_data(self):
        data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
        df = pd.read_parquet(data_path)
        df = add_lag_features(df)
        df.dropna(inplace=True)
        
        # Select Features
        model_features = self.model.get_booster().feature_names
        self.features = model_features
        
        # Test on last 20%
        split_idx = int(len(df) * 0.8)
        self.sim_df = df.iloc[split_idx:].copy()
        self.X_test = self.sim_df[self.features]
        
        # Pre-calculate probabilities ONCE for speed
        print("Pre-calculating probabilities...")
        probs = self.model.predict_proba(self.X_test)
        self.sim_df['prob_buy'] = probs[:, 1]
        self.sim_df['prob_sell'] = probs[:, 2]

    def test_threshold(self, threshold):
        """Runs a fast vectorized backtest for a specific threshold"""
        capital = self.capital_base
        trades = 0
        wins = 0
        
        # Vectorized Logic (Faster than looping)
        # 1. Identify Entry Points
        # Buy Signal: Prob > Threshold AND ATR > 0
        buy_signals = (self.sim_df['prob_buy'] > threshold) & (self.sim_df['atr'] > 0)
        sell_signals = (self.sim_df['prob_sell'] > threshold) & (self.sim_df['atr'] > 0)
        
        # We need to loop only through signals to check TP/SL logic
        # To save time, we filter the dataframe to only rows with signals
        signals = self.sim_df[buy_signals | sell_signals].copy()
        
        if len(signals) == 0:
            return 0, 0, 0 # No trades
            
        # Realistic Loop for PnL
        position = 0 # 0, 1, -1
        entry_price = 0
        sl = 0
        tp = 0
        
        # Note: This is a simplified sequential simulation
        # It assumes we take the first signal and hold until exit
        # It ignores overlapping signals (which is correct for a single-position system)
        
        df_iter = self.sim_df.iterrows()
        
        for i, row in df_iter:
            price = row['close']
            atr = row['atr']
            
            # Check Exit
            if position != 0:
                hit_sl = (position == 1 and price <= sl) or (position == -1 and price >= sl)
                hit_tp = (position == 1 and price >= tp) or (position == -1 and price <= tp)
                
                if hit_sl or hit_tp:
                    # Calculate PnL
                    if position == 1: pnl = (price - entry_price) / entry_price
                    else: pnl = (entry_price - price) / entry_price
                    
                    capital = capital * (1 + pnl) # Profit/Loss
                    capital = capital * (1 - self.fee) # Exit Fee
                    position = 0
                    if pnl > 0: wins += 1
                continue
            
            # Check Entry (Only if no position)
            if row['prob_buy'] > threshold:
                position = 1
                entry_price = price
                sl = price - (1.5 * atr)
                tp = price + (2.5 * atr)
                capital = capital * (1 - self.fee) # Entry Fee
                trades += 1
                
            elif row['prob_sell'] > threshold:
                position = -1
                entry_price = price
                sl = price + (1.5 * atr)
                tp = price - (2.5 * atr)
                capital = capital * (1 - self.fee) # Entry Fee
                trades += 1
                
        roi = ((capital - self.capital_base) / self.capital_base) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return trades, win_rate, roi

    def run_optimization(self):
        print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'ROI %':<10}")
        print("-" * 45)
        
        # Test thresholds from 0.50 to 0.98
        best_roi = -999
        best_thresh = 0
        
        for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            trades, wr, roi = self.test_threshold(t)
            print(f"{t:<10} | {trades:<8} | {wr:<10.2f} | {roi:<10.2f}")
            
            if roi > best_roi:
                best_roi = roi
                best_thresh = t
                
        print("-" * 45)
        print(f"BEST RESULT: Threshold {best_thresh} -> {best_roi:.2f}% ROI")

if __name__ == "__main__":
    # Point to your ENHANCED model
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_enhanced.json')
    opt = Optimizer(model_path)
    opt.load_data()
    opt.run_optimization()