import pandas as pd
import numpy as np
import os
from src.config import Config

def inspect():
    path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    print(f"Inspecting: {path}")
    
    if not os.path.exists(path):
        print("ERROR: File not found!")
        return

    df = pd.read_parquet(path)
    print(f"Total Rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    # Check for NAs
    na_counts = df.isna().sum()
    print("\n--- NaN Counts ---")
    print(na_counts[na_counts > 0])
    
    # Check for Infinite values
    # Select only numeric columns for checking infinite
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    print("\n--- Infinite Counts ---")
    print(inf_counts[inf_counts > 0])
    
    # Check target specifically
    if 'target_return' in df.columns:
        print("\n--- Target Analysis ---")
        tgt = df['target_return']
        print(f"Min: {tgt.min()}, Max: {tgt.max()}")
        print(f"Zeros: {(tgt == 0).sum()}")
        print(f"NaNs: {tgt.isna().sum()}")
        print(f"Infs: {np.isinf(tgt).sum()}")
        
        # Check first and last 5 rows
        print("\nLast 5 rows of Target:")
        print(tgt.tail(5))
    else:
        print("CRITICAL: 'target_return' column missing!")

if __name__ == "__main__":
    inspect()