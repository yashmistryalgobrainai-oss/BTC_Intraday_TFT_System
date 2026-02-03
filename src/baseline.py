import pandas as pd
import numpy as np

class BaselineStrategy:
    """Simple MA crossover baseline for comparison."""
    
    def __init__(self, fast=20, slow=50):
        self.fast = fast
        self.slow = slow
    
    def backtest(self, df, initial_capital=10000):
        """Run backtest with 0.2% fees."""
        df = df.copy()
        df['ma_fast'] = df['close'].rolling(self.fast).mean()
        df['ma_slow'] = df['close'].rolling(self.slow).mean()
        df.dropna(inplace=True)
        
        capital = initial_capital
        position = 0
        trades = []
        entry = 0
        
        for i in range(1, len(df)):
            prev_signal = 1 if df['ma_fast'].iloc[i-1] > df['ma_slow'].iloc[i-1] else -1
            curr_signal = 1 if df['ma_fast'].iloc[i] > df['ma_slow'].iloc[i] else -1
            
            if prev_signal != curr_signal:  # Signal flip
                if position != 0:
                    # Close existing
                    price = df['close'].iloc[i]
                    pnl = (price - entry) / entry if position == 1 else (entry - price) / entry
                    capital *= (1 + pnl - 0.002)
                    trades.append(pnl)
                    position = 0
                
                # Open new
                position = curr_signal
                entry = df['close'].iloc[i]
                capital *= 0.998  # Entry fee
        
        win_rate = (np.array(trades) > 0).mean() * 100 if trades else 0
        
        return {
            'final_capital': capital,
            'return': (capital - initial_capital) / initial_capital * 100,
            'trades': len(trades),
            'win_rate': win_rate
        }
