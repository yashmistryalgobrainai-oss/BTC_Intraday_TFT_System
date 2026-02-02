import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.utils.class_weight import compute_sample_weight
from src.config import Config

def add_lag_features(df):
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['rolling_volatility'] = df['returns'].rolling(window=12).std()
    return df

class WalkForwardValidator:
    def __init__(self):
        self.capital = 10000
        self.equity = [self.capital]
        self.trades = []
        
        # Hyperparams
        self.THRESHOLD = 0.60
        self.TRAIN_WINDOW = 5000 
        self.TEST_WINDOW = 1000   

    def load_data(self):
        print("Loading Data...")
        data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
        df = pd.read_parquet(data_path)
        df = add_lag_features(df)
        df.dropna(inplace=True)
        
        # Prepare Target
        THRESHOLD = 0.0015
        df['target_class'] = 0
        df.loc[df['target_return'] > THRESHOLD, 'target_class'] = 1
        df.loc[df['target_return'] < -THRESHOLD, 'target_class'] = 2
        
        self.df = df.reset_index(drop=True)
        print(f"Total Candles: {len(self.df)}")

    def run_validation(self):
        print(f"--- Starting Walk-Forward (Train {self.TRAIN_WINDOW} -> Trade {self.TEST_WINDOW}) ---")
        
        current_idx = self.TRAIN_WINDOW
        
        while current_idx < len(self.df):
            # 1. Define Segments
            train_start = current_idx - self.TRAIN_WINDOW
            train_end = current_idx
            test_end = min(current_idx + self.TEST_WINDOW, len(self.df))
            
            train_data = self.df.iloc[train_start:train_end]
            test_data = self.df.iloc[train_end:test_end].copy()
            
            if len(test_data) == 0: break
            
            # --- FIX: SMART DROP ---
            # We explicitly list potential non-feature columns
            # and filter for only the ones that actually exist in the dataframe.
            potential_drops = ['target_class', 'target_return', 'datetime', 'group_id', 'time_idx']
            drop_cols = [c for c in potential_drops if c in train_data.columns]

            X_train = train_data.drop(columns=drop_cols)
            y_train = train_data['target_class']
            
            # Check if we have valid classes
            if len(y_train.unique()) <= 1:
                # Skip windows where price never moves (rare but possible)
                current_idx += self.TEST_WINDOW
                continue

            weights = compute_sample_weight(class_weight='balanced', y=y_train)
            
            # 2. Train Model
            model = xgb.XGBClassifier(
                n_estimators=200,    
                max_depth=6,         
                learning_rate=0.05,
                objective='multi:softprob',
                num_class=3,
                tree_method='hist',
                device='cuda',       
                verbosity=0
            )
            model.fit(X_train, y_train, sample_weight=weights)
            
            # 3. Predict
            X_test = test_data.drop(columns=drop_cols)
            probs = model.predict_proba(X_test)
            test_data['prob_buy'] = probs[:, 1]
            test_data['prob_sell'] = probs[:, 2]
            
            # 4. Simulate
            self.simulate_window(test_data)
            
            # Progress Print
            if current_idx % (self.TEST_WINDOW * 10) == 0:
                print(f"Processed up to candle {test_end}. Current Capital: ${self.capital:.0f}")
            
            current_idx += self.TEST_WINDOW

    def simulate_window(self, df):
        position = 0
        entry_price = 0
        sl = 0
        tp = 0
        fee = 0.001
        
        for i, row in df.iterrows():
            price = row['close']
            atr = row['atr']
            
            # Exit Logic
            if position != 0:
                hit_sl = (position == 1 and price <= sl) or (position == -1 and price >= sl)
                hit_tp = (position == 1 and price >= tp) or (position == -1 and price <= tp)
                
                if hit_sl or hit_tp:
                    pnl = (price - entry_price)/entry_price if position == 1 else (entry_price - price)/entry_price
                    self.capital *= (1 + pnl - fee)
                    self.trades.append(pnl)
                    position = 0
                continue
                
            # Entry Logic
            if atr == 0: continue
            
            if row['prob_buy'] > self.THRESHOLD:
                position = 1
                entry_price = price
                sl = price - (1.5 * atr)
                tp = price + (2.5 * atr)
                self.capital *= (1 - fee)
                
            elif row['prob_sell'] > self.THRESHOLD:
                position = -1
                entry_price = price
                sl = price + (1.5 * atr)
                tp = price - (2.5 * atr)
                self.capital *= (1 - fee)

if __name__ == "__main__":
    wf = WalkForwardValidator()
    wf.load_data()
    wf.run_validation()
    
    print("\n--- WALK-FORWARD RESULTS ---")
    print(f"Final Capital: ${wf.capital:.2f}")
    print(f"Total Return:  {((wf.capital-10000)/10000)*100:.2f}%")
    print(f"Total Trades:  {len(wf.trades)}")