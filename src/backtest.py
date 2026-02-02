import os
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.config import Config
from src.model_tft import TFTBuilder

class Backtester:
    def __init__(self, model_path, data_path, initial_capital=10000):
        self.model_path = model_path
        self.data_path = data_path
        self.capital = initial_capital
        self.equity = [initial_capital]
        self.trades = []
        self.position = 0  # 0: None, 1: Long, -1: Short
        self.entry_price = 0
        self.sl = 0
        self.tp = 0
        
        # Load Data
        self.df = pd.read_parquet(self.data_path)
        self.df = self.df.iloc[-Config.INPUT_WINDOW*100:] # Test on last portion of data for speed
        self.df = self.df.reset_index(drop=True) # Reset index for cleaner looping
        
        # Load Model
        print(f"Loading model from {model_path}...")
        self.best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.best_tft.eval()
        
    def generate_predictions(self):
        print("Generating predictions (this may take a moment)...")
        
        # Create dataset specifically for prediction
        dataset = TFTBuilder(self.df).create_dataset()[0]
        dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        
        # Get raw predictions
        raw_predictions = self.best_tft.predict(dataloader, mode="quantiles", return_x=True)
        
        # --- THE FIX: Sum the next 4 candles (1 Hour Trend) ---
        # Old: raw_predictions.output[:, 0, 3] (Just next 15m)
        # New: Sum of all 4 prediction steps (Total return for next 1h)
        self.preds = raw_predictions.output[:, :, 3].sum(dim=1).numpy()
        
        # --- DEBUG: Print Prediction Statistics ---
        # --- DEBUG: Print Prediction Statistics ---
        print("\n--- PREDICTION STATS (1-Hour Cumulative) ---")
        print(f"Max Forecast: {np.nanmax(self.preds):.6f}")
        print(f"Min Forecast: {np.nanmin(self.preds):.6f}")
        print(f"Mean Forecast: {np.nanmean(self.preds):.6f}")
        print(f"NaN Count: {np.isnan(self.preds).sum()}")
        print("------------------------\n")

        valid_indices = np.arange(Config.INPUT_WINDOW, len(self.df) - Config.PREDICT_WINDOW)
        self.sim_df = self.df.iloc[valid_indices].copy()
        self.sim_df['model_forecast'] = self.preds[:len(self.sim_df)]
        self.sim_df = self.sim_df.reset_index(drop=True)

    def run_simulation(self):
        print("Running trading simulation...")
        
        fee = 0.001 # 0.1% per trade
        
        # --- REALITY CHECK ---
        # We only trade if the predicted move is > 0.15% (0.0015)
        # This covers the 0.1% entry fee + spread.
        threshold = 0.0005
        
        print(f"Trading Threshold set to: {threshold} (Must beat fees!)")

        for i, row in self.sim_df.iterrows():
            current_price = row['close']
            atr = row['atr']
            forecast = row['model_forecast']
            
            # 1. Manage Open Position
            if self.position != 0:
                if (self.position == 1 and current_price <= self.sl) or \
                   (self.position == -1 and current_price >= self.sl):
                    self.close_position(current_price, fee, reason="SL")
                    
                elif (self.position == 1 and current_price >= self.tp) or \
                     (self.position == -1 and current_price <= self.tp):
                    self.close_position(current_price, fee, reason="TP")
                    
                elif i == len(self.sim_df) - 1:
                    self.close_position(current_price, fee, reason="End")
                continue 

            # 2. Entry Logic
            if atr > 0: 
                if forecast > threshold:
                    # LONG
                    self.position = 1
                    self.entry_price = current_price
                    self.sl = current_price - (2.0 * atr) # Widen SL for 1H trades
                    self.tp = current_price + (4.0 * atr) # Target bigger moves
                    self.capital -= (self.capital * fee) 
                    
                elif forecast < -threshold:
                    # SHORT
                    self.position = -1
                    self.entry_price = current_price
                    self.sl = current_price + (2.0 * atr)
                    self.tp = current_price - (4.0 * atr) 
                    self.capital -= (self.capital * fee)
            
            self.equity.append(self.capital)

    def close_position(self, price, fee, reason):
        # Calculate PnL
        if self.position == 1: # Long
            pnl = (price - self.entry_price) / self.entry_price
        else: # Short
            pnl = (self.entry_price - price) / self.entry_price
            
        # Update Capital
        trade_profit = (self.capital * pnl)
        self.capital += trade_profit
        self.capital -= (self.capital * fee) # Pay exit fee
        
        self.trades.append({
            'type': 'Long' if self.position == 1 else 'Short',
            'entry': self.entry_price,
            'exit': price,
            'pnl_pct': pnl * 100,
            'reason': reason,
            'balance': self.capital
        })
        
        self.position = 0 # Reset
        self.sl = 0
        self.tp = 0

    def analyze(self):
        if not self.trades:
            print("No trades were made.")
            return

        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        
        print("\n" + "="*30)
        print("   BACKTEST RESULTS   ")
        print("="*30)
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return:  {((self.capital - 10000)/10000)*100:.2f}%")
        print(f"Total Trades:  {len(trades_df)}")
        print(f"Win Rate:      {(len(wins)/len(trades_df))*100:.2f}%")
        print(f"Max Win:       {trades_df['pnl_pct'].max():.2f}%")
        print(f"Max Loss:      {trades_df['pnl_pct'].min():.2f}%")
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity, label='Equity Curve')
        plt.title("Account Balance Over Time")
        plt.xlabel("Time (Candles)")
        plt.ylabel("Balance ($)")
        plt.legend()
        plt.grid(True)
        # Save plot to file instead of showing (better for servers/scripts)
        plt.savefig('backtest_result.png')
        print("\nResult chart saved as 'backtest_result.png'")

if __name__ == "__main__":
    # Point to the processed data and the BEST model you just trained
    # UPDATE THIS FILENAME to match exactly what is in your models/ folder
    import glob
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(Config.MODEL_SAVE_PATH, "*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No models found in {Config.MODEL_SAVE_PATH}")
        
    # Get the most recently created checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Using latest checkpoint: {latest_checkpoint}")
    
    MODEL_PATH = latest_checkpoint
    DATA_PATH = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    
    backtester = Backtester(MODEL_PATH, DATA_PATH)
    backtester.generate_predictions()
    backtester.run_simulation()
    backtester.analyze()