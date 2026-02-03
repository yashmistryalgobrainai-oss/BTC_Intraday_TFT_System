import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class RealisticBacktester:
    """
    Simulates real-world trading conditions for BTC/USDT on Binance.
    
    Why "Realistic"?
    Most backtesters assume you enter exactly at the Close price with 0 fees.
    In reality:
    1. Fees: You pay 0.1% to enter (Taker) and 0.1% to exit (Taker).
    2. Spread: The Buy price is slightly higher than the Sell price (~0.01-0.02%).
    3. Slippage: In fast markets, your fill price is worse than the signal price (~0.01%).
    
    Total Cost per Trade ~0.26% (Entry + Exit).
    Any strategy making < 0.26% per trade is actually LOSING money.
    """
    
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.equity = [initial_capital]
        self.trades = []
        
        # Current Position State
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0
        self.entry_time = None
        self.entry_fee = 0
        self.sl = 0
        self.tp = 0
        
        # --- COST MODEL (Binance Tier 1) ---
        self.FEE_RATE = 0.001       # 0.1% Exchange Fee
        self.SPREAD = 0.0002        # 0.02% Bid-Ask Spread assumption
        self.SLIPPAGE = 0.0001      # 0.01% Execution delay impact

    def run(self, df_test, probs, threshold=0.60):
        """
        Main Event Loop.
        
        Args:
            df_test (pd.DataFrame): Test data with OHLCV and ATR.
            probs (np.array): Model probabilities [Prob_Hold, Prob_Buy, Prob_Sell].
            threshold (float): Confidence required to enter.
        """
        print(f"Starting Realistic Backtest on {len(df_test)} candles...")
        
        # Pre-calculate signals to speed up loop
        # We only care if Prob > Threshold
        buy_signals = (probs[:, 1] > threshold)
        sell_signals = (probs[:, 2] > threshold)
        
        # Iterate through time
        for i in range(len(df_test)):
            # "Current" candle data (completed)
            # In live trading, we'd act on the OPEN of the NEXT candle based on this close.
            # But standard backtesting often assumes entry at Close.
            # To be safe/realistic, let's assume we enter at Close (plus slippage).
            
            timestamp = df_test.index[i]
            row = df_test.iloc[i]
            price = row['close']
            atr = row.get('atr_raw', row.get('atr', 0))  # Handle both naming conventions
            if atr == 0:
                continue  # Skip candles with zero ATR
            
            # 1. Manage Existing Position
            if self.position != 0:
                self._check_exit(price, timestamp, row)
            
            # 2. Check for New Entry (Only if no position)
            if self.position == 0:
                if buy_signals[i]:
                    self._open_position(1, price, atr, timestamp)
                elif sell_signals[i]:
                    self._open_position(-1, price, atr, timestamp)
            
            # Track Equity (Mark-to-Market is optional, here keeping it simple: Cash + Unrealized)
            self.equity.append(self.capital)

        # Force close at end
        if self.position != 0:
             self._close_position(df_test['close'].iloc[-1], df_test.index[-1], "End of Period")
             
        return self._calculate_metrics()

    def _open_position(self, direction, price, atr, timestamp):
        """Calculates entry price including Slippage + Spread."""
        if atr == 0: return # Safety
        
        # COST ASSUMPTION:
        # Long Entry: You buy at Ask (Price + Spread/2) + Slippage
        # Short Entry: You sell at Bid (Price - Spread/2) - Slippage
        
        cost_impact = (price * (self.SPREAD / 2)) + (price * self.SLIPPAGE)
        
        if direction == 1:
            self.entry_price = price + cost_impact
            self.sl = self.entry_price - (2.0 * atr)
            self.tp = self.entry_price + (3.0 * atr) # 1.5 Reward Ratio
        else:
            self.entry_price = price - cost_impact
            self.sl = self.entry_price + (2.0 * atr)
            self.tp = self.entry_price - (3.0 * atr)

        self.position = direction
        self.entry_time = timestamp
        
        # Deduct Entry Fee immediately
        # Fee is % of position size. Assuming 100% equity bet (Compounding).
        # In reality, you'd bet fixed risk. Here: Simple Compounding.
        fee_amt = self.capital * self.FEE_RATE
        self.capital -= fee_amt
        self.entry_fee = fee_amt

        # Debug output for first 5 trades
        if len(self.trades) < 5:
            print(f"\n[DEBUG] Opening Trade #{len(self.trades)+1}")
            print(f"  Direction: {'LONG' if direction==1 else 'SHORT'}")
            print(f"  Entry: ${self.entry_price:,.2f}")
            print(f"  SL: ${self.sl:,.2f} (Risk: {abs(self.sl-self.entry_price)/self.entry_price*100:.2f}%)")
            print(f"  TP: ${self.tp:,.2f} (Reward: {abs(self.tp-self.entry_price)/self.entry_price*100:.2f}%)")
            print(f"  ATR: ${atr:,.2f}")
            print(f"  R:R Ratio: 1:{abs(self.tp-self.entry_price)/abs(self.sl-self.entry_price):.2f}")

    def _check_exit(self, current_price, timestamp, candle_row):
        """Checks if Price hit SL or TP during the candle."""
        # Note: Using Close price for checking SL/TP is "Loose" backtesting.
        # Strict backtesting checks Low/High.
        
        if self.position == 1: # Long
            # Did we get stopped out? (Check Low)
            if candle_row['low'] <= self.sl:
                self._close_position(self.sl, timestamp, "Stop Loss")
            # Did we hit TP? (Check High)
            elif candle_row['high'] >= self.tp:
                self._close_position(self.tp, timestamp, "Take Profit")
                
        elif self.position == -1: # Short
            # Did we get stopped out? (Check High)
            if candle_row['high'] >= self.sl:
                self._close_position(self.sl, timestamp, "Stop Loss")
            # Did we hit TP? (Check Low)
            elif candle_row['low'] <= self.tp:
                self._close_position(self.tp, timestamp, "Take Profit")

    def _close_position(self, raw_price, timestamp, reason):
        """Calculates exit price including Slippage + Spread and updates Equity."""
        
        # COST ASSUMPTION:
        # Long Exit (Sell): Bid Price (Price - Spread/2) - Slippage
        # Short Exit (Buy): Ask Price (Price + Spread/2) + Slippage
        
        cost_impact = (raw_price * (self.SPREAD / 2)) + (raw_price * self.SLIPPAGE)
        
        if self.position == 1:
            exit_price = raw_price - cost_impact
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            exit_price = raw_price + cost_impact
            pnl_pct = (self.entry_price - exit_price) / self.entry_price
            
        # Update Capital
        # PnL applied to capital
        profit_amt = self.capital * pnl_pct
        self.capital += profit_amt
        
        # Deduct Exit Fee
        exit_fee = self.capital * self.FEE_RATE
        self.capital -= exit_fee
        
        # Record Trade
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'type': 'Long' if self.position == 1 else 'Short',
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_abs': profit_amt - self.entry_fee - exit_fee, # Net Profit
            'reason': reason,
            'capital': self.capital
        })
        
        # Reset
        self.position = 0

    def _calculate_metrics(self):
        if not self.trades:
            return {"Total Trades": 0, "Return": 0}
            
        df_trades = pd.DataFrame(self.trades)
        
        # 1. Basic Stats
        total_return = ((self.capital - self.equity[0]) / self.equity[0]) * 100
        wins = df_trades[df_trades['pnl_abs'] > 0]
        losses = df_trades[df_trades['pnl_abs'] <= 0]
        win_rate = (len(wins) / len(df_trades)) * 100
        
        # 2. Advanced Stats
        avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
        avg_loss = losses['pnl_pct'].mean() if not losses.empty else 0
        profit_factor = abs(wins['pnl_abs'].sum() / losses['pnl_abs'].sum()) if losses['pnl_abs'].sum() != 0 else float('inf')
        
        # 3. Drawdown
        equity_curve = pd.Series([t['capital'] for t in self.trades])
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        metrics = {
            "Final Capital": self.capital,
            "Total Return %": total_return,
            "Total Trades": len(df_trades),
            "Win Rate %": win_rate,
            "Profit Factor": profit_factor,
            "Avg Win %": avg_win,
            "Avg Loss %": avg_loss,
            "Max Drawdown %": max_drawdown
        }
        
        self.trades_df = df_trades # Store for analysis
        return metrics

    def plot_equity(self):
        if not self.trades: return
        df = pd.DataFrame(self.trades)
        plt.figure(figsize=(12, 6))
        plt.plot(df['exit_time'], df['capital'], label='Equity')
        plt.title('Realistic Equity Curve (Fees + Slippage Included)')
        plt.xlabel('Date')
        plt.ylabel('Capital used ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig('equity_curve_v2.png')
        print("Equity curve saved to equity_curve_v2.png")
