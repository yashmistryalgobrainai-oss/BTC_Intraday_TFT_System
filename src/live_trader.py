import time
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime
from src.config import Config
from src.regime_detector import RegimeDetector

# --- 1. UTILITIES (Must match Training Logic EXACTLY) ---
def add_lag_features(df):
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['rolling_volatility'] = df['returns'].rolling(window=12).std()
    return df

def get_indicators(df):
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
    
    # Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    return df

class LiveSentinel:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"LOADING AI MODEL: {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_names = self.model.get_booster().feature_names

    def fetch_live_data(self):
        """Get last 100 candles from Binance (Free API)"""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "limit": 100 
        }
        try:
            r = requests.get(url, params=params)
            data = r.json()
            
            # Parse Binance Response
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'q_vol', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert strings to floats
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def analyze_market(self):
        print(f"\n--- SCANNING MARKET: {datetime.now().strftime('%H:%M:%S')} ---")
        
        # 1. Get Data
        df = self.fetch_live_data()
        if df is None: return

        # 2. Engineer Features
        df = get_indicators(df)
        df = add_lag_features(df)
        df['adx'] = RegimeDetector.calculate_adx(df)
        df.dropna(inplace=True)
        
        # --- FIX: SEPARATE LIVE vs. ANALYSIS ---
        
        # A. Live Ticking Data (Latest forming candle)
        # We use this ONLY for the display price so you know it's working
        live_candle = df.iloc[-1]
        live_price = live_candle['close']
        
        # B. Analysis Data (Last COMPLETED candle)
        # We use this for the AI to ensure the signal is confirmed and won't vanish
        analysis_candle = df.iloc[-2].copy()
        analysis_price = analysis_candle['close']
        atr = analysis_candle['atr']
        adx = analysis_candle['adx']
        
        print(f"💰 CURRENT PRICE: ${live_price:,.2f} (Signal Price: ${analysis_price:,.2f})")
        # ---------------------------------------
        
        # 3. Detect Regime (Based on confirmed history)
        regime = RegimeDetector.get_regime(analysis_candle)
        
        # 4. Set Dynamic Threshold
        if regime == "TRENDING":
            THRESHOLD = 0.60
            print(f"✅ REGIME: TRENDING (ADX {adx:.1f}). Aggressive Mode ON.")
        elif regime == "STABLE":
            THRESHOLD = 0.75
            print(f"⚠️ REGIME: STABLE (ADX {adx:.1f}). Defensive Mode ON.")
        else:
            THRESHOLD = 0.99
            print(f"⛔ REGIME: NOISE (ADX {adx:.1f}). TRADING HALTED.")

        # 5. Predict
        input_data = pd.DataFrame([analysis_candle[self.feature_names]])
        
        probs = self.model.predict_proba(input_data)[0]
        prob_buy = probs[1]
        prob_sell = probs[2]
        
        print(f"📊 MODEL OUTPUT -> Buy: {prob_buy:.1%} | Sell: {prob_sell:.1%} | Hold: {probs[0]:.1%}")
        
        # 6. Decision Logic (Execute at Live Price, but decision based on Confirmed Signal)
        if prob_buy > THRESHOLD:
            sl = live_price - (1.5 * atr)
            tp = live_price + (2.5 * atr)
            print("\n" + "!"*40)
            print(f"🚀 BUY SIGNAL DETECTED")
            print(f"Entry: {live_price:,.2f}")
            print(f"Stop Loss: {sl:,.2f}")
            print(f"Take Profit: {tp:,.2f}")
            print(f"Confidence: {prob_buy:.1%}")
            print("!"*40 + "\n")
            
        elif prob_sell > THRESHOLD:
            sl = live_price + (1.5 * atr)
            tp = live_price - (2.5 * atr)
            print("\n" + "!"*40)
            print(f"🔻 SELL SIGNAL DETECTED")
            print(f"Entry: {live_price:,.2f}")
            print(f"Stop Loss: {sl:,.2f}")
            print(f"Take Profit: {tp:,.2f}")
            print(f"Confidence: {prob_sell:.1%}")
            print("!"*40 + "\n")
        else:
            print("💤 No Trade. Waiting for setup...")

if __name__ == "__main__":
    model_file = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_enhanced.json')
    bot = LiveSentinel(model_file)
    
    print("System Online. Scanning every 60 seconds...")
    try:
        while True:
            bot.analyze_market()
            # Wait 60 seconds (In real deployment, you'd sync this to :00, :15, :30, :45 minutes)
            time.sleep(900) 
    except KeyboardInterrupt:
        print("System Shutting Down.")