import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from src.config import Config
from src.regime_detector import RegimeDetector

def add_lag_features(df):
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['rolling_volatility'] = df['returns'].rolling(window=12).std()
    return df

class AdaptiveBacktester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.capital = 10000
        self.equity = [self.capital]
        self.trades = []
        self.position = 0
        self.entry_price = 0
        self.sl = 0
        self.tp = 0
        self.fee = 0.001 

        print(f"Loading model from {model_path}...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def load_data(self):
        data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
        df = pd.read_parquet(data_path)
        
        # 1. Add Features
        df = add_lag_features(df)
        
        # 2. Add Regime Indicators (ADX)
        print("Calculating Market Regimes...")
        df['adx'] = RegimeDetector.calculate_adx(df)
        df.dropna(inplace=True)
        
        # Select Features (Automated)
        model_features = self.model.get_booster().feature_names
        self.features = model_features
        
        # Test on last 20%
        split_idx = int(len(df) * 0.8)
        self.sim_df = df.iloc[split_idx:].copy()
        
        self.X_test = self.sim_df[self.features]
        print(f"Backtesting on {len(self.sim_df)} candles...")

    def run_simulation(self):
        print("Generating Probabilities...")
        probs = self.model.predict_proba(self.X_test)
        self.sim_df['prob_buy'] = probs[:, 1]
        self.sim_df['prob_sell'] = probs[:, 2]
        
        print("Running ADAPTIVE Trading Loop...")
        
        regime_stats = {"TRENDING": 0, "STABLE": 0, "NOISE": 0}
        
        for i, row in self.sim_df.iterrows():
            current_price = row['close']
            atr = row['atr']
            
            # --- 1. DETECT REGIME ---
            regime = RegimeDetector.get_regime(row)
            
            # --- 2. DYNAMIC THRESHOLD ---
            if regime == "TRENDING":
                CONFIDENCE = 0.60 # Aggressive
            elif regime == "STABLE":
                CONFIDENCE = 0.75 # Conservative
            else: # NOISE
                CONFIDENCE = 0.95 # Basically impossible -> Don't trade
                
            # Manage Position
            if self.position != 0:
                if (self.position == 1 and current_price <= self.sl) or \
                   (self.position == -1 and current_price >= self.sl):
                    self.close_position(current_price, "SL")
                elif (self.position == 1 and current_price >= self.tp) or \
                     (self.position == -1 and current_price <= self.tp):
                    self.close_position(current_price, "TP")
                continue

            # Entry Logic
            if atr == 0: continue
            
            if row['prob_buy'] > CONFIDENCE:
                self.position = 1
                self.entry_price = current_price
                self.sl = current_price - (1.5 * atr)
                self.tp = current_price + (2.5 * atr)
                self.capital -= (self.capital * self.fee)
                regime_stats[regime] += 1
            
            elif row['prob_sell'] > CONFIDENCE:
                self.position = -1
                self.entry_price = current_price
                self.sl = current_price + (1.5 * atr)
                self.tp = current_price - (2.5 * atr)
                self.capital -= (self.capital * self.fee)
                regime_stats[regime] += 1

            self.equity.append(self.capital)
            
        print("\nTrades per Regime:", regime_stats)

    def close_position(self, price, reason):
        if self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - price) / self.entry_price
            
        profit = self.capital * pnl
        self.capital += profit
        self.capital -= (self.capital * self.fee)
        
        self.trades.append({'pnl_pct': pnl * 100, 'reason': reason})
        self.position = 0

    def analyze(self):
        if not self.trades:
            print("No trades made.")
            return

        res = pd.DataFrame(self.trades)
        wins = res[res['pnl_pct'] > 0]
        
        print("\n" + "="*30)
        print("   ADAPTIVE STRATEGY RESULTS   ")
        print("="*30)
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return:  {((self.capital - 10000)/10000)*100:.2f}%")
        print(f"Total Trades:  {len(res)}")
        print(f"Win Rate:      {(len(wins)/len(res))*100:.2f}%")
        
        plt.figure(figsize=(12,6))
        plt.plot(self.equity)
        plt.title("Adaptive Strategy Equity Curve")
        plt.grid(True)
        plt.savefig('backtest_adaptive.png')
        print("Chart saved to backtest_adaptive.png")

if __name__ == "__main__":
    model_file = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_enhanced.json')
    bt = AdaptiveBacktester(model_file)
    bt.load_data()
    bt.run_simulation()
    bt.analyze()