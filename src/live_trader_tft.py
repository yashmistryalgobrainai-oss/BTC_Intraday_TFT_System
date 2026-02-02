import time
import requests
import pandas as pd
import numpy as np
import torch
import os
import glob
from pytorch_forecasting import TemporalFusionTransformer
from src.config import Config
from src.model_tft import TFTBuilder

class LiveTFTTrader:
    def __init__(self):
        self.device = Config.DEVICE
        
        # 1. Find and Load the Latest Model
        checkpoints = glob.glob(os.path.join(Config.MODEL_SAVE_PATH, "*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No models found in {Config.MODEL_SAVE_PATH}")
        
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Loading Live Model: {latest_checkpoint}...")
        
        self.model = TemporalFusionTransformer.load_from_checkpoint(latest_checkpoint)
        self.model.eval()
        self.model.to(self.device)
        print("Model Loaded Successfully.")

    def fetch_live_data(self):
        """
        Fetches the last 100 candles (15m) from Binance.
        We need enough history for the 'INPUT_WINDOW' (60) + Indicator/Lag calculation buffers.
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "limit": 150 # Fetch extra to ensure valid calculation
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            # Columns: Open Time, Open, High, Low, Close, Volume, ...
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'q_vol', 'num_trades', 'taker_base', 'taker_quote', 'ignore']
            
            df = pd.DataFrame(data, columns=cols)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df[numeric_cols] # Return only OHLCV
            
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return None

    def engineer_features(self, df):
        """
        Must identically match 'src/indicators.py' or 'data_processor.py' logic.
        """
        df = df.copy()
        
        # --- 1. Technical Indicators ---
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # --- 2. Time Features ---
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_of_week'] = df.index.dayofweek
        
        # --- 3. TFT Specific Requirements ---
        df = df.dropna() # Drop any NaN from indicators
        df['group_id'] = 'BTCUSD'
        df['time_idx'] = np.arange(len(df)) # Continuous index for this window
        
        # Dummy target (not needed for prediction, but required by Dataset class)
        df[Config.TARGET] = 0.0 
        
        return df

    def predict(self, df):
        # We need at least INPUT_WINDOW rows
        if len(df) <= Config.INPUT_WINDOW:
            print(f"Not enough data yet. Have {len(df)}, need > {Config.INPUT_WINDOW}")
            return None
            
        # Use simple pytorch-forecasting method to predict directly from dataframe
        # This automatically handles the dataset creation internally if configured right, 
        # BUT explicit dataset creation is safer for consistency.
        
        # Take the last context window
        # TFTBuilder expects the whole dataframe to build the structure
        try:
            dataset = TFTBuilder(df).create_dataset()[0]
            dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            # Predict
            raw_pred = self.model.predict(dataloader, mode="quantiles", return_x=True)
            
            # Prediction shape: [Batch, Prediction_Length, Quantiles]
            # We want the 0.5 quantile (median) usually, or mean
            # predictions output is usually just the mean if mode="prediction" or 0.5 quantile
            
            # Let's extract the forecast for the next 4 candles (Cumulative Return)
            # raw_pred.output has shape [1, 4, 7] (7 quantiles)
            # Center quantiles index is usually 3 (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
            
            forecast_steps = raw_pred.output[0, :, 3].cpu().numpy() # 0.5 quantile (Median)
            total_predicted_return = forecast_steps.sum()
            
            return total_predicted_return, forecast_steps
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, None

    def run(self):
        print(f"\n--- STARTING TFT LIVE MONITOR ({Config.TIMEFRAME}) ---")
        print("Press Ctrl+C to stop.\n")
        
        last_processed_time = None
        
        while True:
            try:
                # 1. Fetch
                df_raw = self.fetch_live_data()
                if df_raw is None:
                    time.sleep(60)
                    continue
                
                # Check if we have a new completed candle
                # Logic: We usually predict on COMPLETED candles. 
                # The last row in Binance is the "open" (forming) candle.
                # So we look at the second to last row.
                current_candle_time = df_raw.index[-1]
                
                # Only log heartbeat every loop
                print(f"\rScanning... Current Candle: {current_candle_time}. Price: {df_raw['close'].iloc[-1]:.2f}", end="")
                
                # 2. Process & Predict (We can do this every loop or only on new candles)
                # Let's do it every loop to see how the forecast changes as the candle forms (optional)
                # OR stick to confirmed candles. Let's stick to Confirmed (index -2) for signals.
                 
                df_processed = self.engineer_features(df_raw)
                
                # We want to predict based on the sequence ending at the LAST COMPLETED candle
                # So we exclude the very last row (forming)
                df_context = df_processed.iloc[:-1].copy()
                
                last_completed_time = df_context.index[-1]
                
                if last_processed_time != last_completed_time:
                    print(f"\n\n[NEW CANDLE CLOSE] Analysis at {last_completed_time}")
                    
                    pred_return, raw_steps = self.predict(df_context)
                    
                    if pred_return is not None:
                        # Logic matched to Backtest
                        threshold = 0.0005 # 0.05%
                        
                        print(f"Prediction (Next 1h Return): {pred_return:.6f} ({pred_return*100:.3f}%)")
                        print(f"Breakdown (Next 4 candles): {[f'{x:.5f}' for x in raw_steps]}")
                        
                        price = df_context['close'].iloc[-1]
                        atr = df_context['atr'].iloc[-1]
                        
                        if pred_return > threshold:
                            tp = price + (4.0 * atr)
                            sl = price - (2.0 * atr)
                            print(f"🚀 SIGNAL: BUY (Long)")
                            print(f"   Entry: {price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
                            
                        elif pred_return < -threshold:
                            tp = price - (4.0 * atr)
                            sl = price + (2.0 * atr)
                            print(f"🔻 SIGNAL: SELL (Short)")
                            print(f"   Entry: {price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
                        else:
                            print("😴 SIGNAL: HOLD (Forecast weak)")
                    
                    last_processed_time = last_completed_time
                    print("-" * 50)
                
                # Sleep
                time.sleep(10) # 10s check interval
                
            except KeyboardInterrupt:
                print("\nStopping Live Trader.")
                break
            except Exception as e:
                print(f"\nUnexpected Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = LiveTFTTrader()
    trader.run()
