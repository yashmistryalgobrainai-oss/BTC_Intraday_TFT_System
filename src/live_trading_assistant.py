import time
import requests
import pandas as pd
import numpy as np
import datetime
import os
import pickle
from src.feature_engineer_v2 import TemporalFeatureEngineer
from src.models.xgb_signal_generator import TradingSignalXGB
from src.config import Config

class LiveTradingAssistant:
    def __init__(self, model_filename='trading_assistant_v2.json'):
        # Load fitted scaler from training
        scaler_path = os.path.join(Config.MODEL_SAVE_PATH, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.feature_engineer = pickle.load(f)
            print("Loaded fitted scaler from training.")
        else:
            print("WARNING: Scaler not found. Creating new one (NOT RECOMMENDED FOR PRODUCTION)")
            self.feature_engineer = TemporalFeatureEngineer()
        
        self.model = TradingSignalXGB()
        
        # Load Model
        try:
            self.model.load_model(model_filename)
        except Exception as e:
            print(f"Warning: Model file not found ({e}). System will run in DEMO mode.")

    def fetch_live_candles(self, limit=100):
        """Fetches recent 15m candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "limit": limit
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            # Format DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'q_vol', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
            ])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)
            
            return df[cols]
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def analyze_market(self, user_capital):
        # 1. Get Data
        df = self.fetch_live_candles(limit=150) # Need enough history for lags
        if df is None: return

        # 2. Process Features
        # Use transform only - scaler was fitted during training
        try:
            df_processed = self.feature_engineer.transform(df)
            
            # Get latest COMPLETED candle (binance sends forming candle as last row)
            # If current time is 10:07, the 10:00 candle is forming. 
            # We signal based on the 09:45 candle (completed at 10:00).
            latest_idx = -2 
            row = df_processed.iloc[latest_idx:latest_idx+1]
            current_price = df['close'].iloc[-1] # Live ticker price
            signal_price = df['close'].iloc[latest_idx] # Price signal is based on
            atr = df_processed.get('atr_raw', df_processed['atr']).iloc[latest_idx]
            
            # 3. Predict
            # Returns: {'signal': 1, 'confidence': 0.75, 'probs': {...}}
            analysis = self.model.predict_signal(row) 
            
            self._display_recommendation(analysis, current_price, atr, user_capital)
            
        except Exception as e:
            print(f"Analysis Error: {e}")

    def _calculate_trade_params(self, direction, price, atr, capital):
        """Calculates precise Position Size based on Risk."""
        RISK_PER_TRADE = 0.02 # 2% Risk
        # 1.5:1 Minimum Reward
        
        risk_amt = capital * RISK_PER_TRADE
        
        if direction == 1: # Buy
            sl = price - (2.0 * atr)
            tp = price + (3.0 * atr)
            dist_per_unit = price - sl
        else: # Sell
            sl = price + (2.0 * atr)
            tp = price - (3.0 * atr)
            dist_per_unit = sl - price
            
        if dist_per_unit <= 0: return None
        
        # Position Size = Risk Amount / Stop Loss Distance per unit
        # Example: Risk $200. SL is $100 away. Position = 2 BTC.
        position_units = risk_amt / dist_per_unit
        position_value = position_units * price
        
        # Max Position Cap (e.g., don't use more than 50% of account even if SL is tight)
        # Here we cap leverage at 1x for safety (max position = capital)
        # Or capping at 10% as per requirements
        MAX_POS_SIZE = capital * 0.10
        if position_value > MAX_POS_SIZE:
             position_value = MAX_POS_SIZE
             position_units = position_value / price
             # Re-calc risk (it will be lower than 2%)
             real_risk = position_units * dist_per_unit
        else:
             real_risk = risk_amt
             
        # Profit Potential
        profit_dist = abs(tp - price)
        potential_profit = position_units * profit_dist
        
        return {
            'sl': sl,
            'tp': tp,
            'units': position_units,
            'value': position_value,
            'risk': real_risk,
            'profit': potential_profit
        }

    def _display_recommendation(self, analysis, price, atr, capital):
        signal = analysis['signal'] # 0=Hold, 1=Buy, 2=Sell
        conf = analysis['confidence']
        
        # Threshold Check
        if conf < 0.40 or signal == 0:
            print(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] HOLD | Conf: {conf:.0%} | Price: ${price:.2f}", end="")
            return

        # Prepare Signal Output
        if signal == 1:
            side = "BUY"
            direction = 1
        else:
            side = "SELL"
            direction = -1
            
        params = self._calculate_trade_params(direction, price, atr, capital)
        
        print("\n" + "="*50)
        icon = "🚀" if direction == 1 else "🔻"
        print(f"{icon} RECOMMENDATION: {side}")
        print("="*50)
        print(f"Confidence:     {conf:.0%}")
        print(f"Current Price:  ${price:,.2f}")
        print("-" * 30)
        print(f"Position Size:  ${params['value']:,.2f} ({params['units']:.5f} BTC)")
        print(f"Stop Loss:      ${params['sl']:,.2f} ({((params['sl']-price)/price)*100:.2f}%)")
        print(f"Take Profit:    ${params['tp']:,.2f} ({((params['tp']-price)/price)*100:.2f}%)")
        print("-" * 30)
        print(f"Risk Amount:    ${params['risk']:.2f} ({(params['risk']/capital)*100:.1f}%)")
        print(f"Pot. Profit:    ${params['profit']:.2f}")
        print(f"Risk:Reward:    1:{params['profit']/params['risk']:.2f}")
        print("="*50 + "\n")

    def run_live_monitor(self, capital, interval=60):
        print(f"--- LIVE TRADING ASSISTANT ONLINE ---")
        print(f"Account Size: ${capital:,.2f}")
        print(f"Monitoring BTC/USDT (15m)... Press Ctrl+C to Stop.")
        
        while True:
            try:
                self.analyze_market(capital)
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nAssistant shutting down.")
                break
            except Exception as e:
                print(f"Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    # Example usage
    bot = LiveTradingAssistant() 
    bot.run_live_monitor(capital=10000, interval=15) # Check every 15s
