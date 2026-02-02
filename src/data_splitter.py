import pandas as pd
import numpy as np

class TimeSeriesSplitter:
    """
    Robust Time-Series Splitter for Trading Data.
    Ensures strict chronological order and prevents overlap.
    """
    
    def __init__(self, train_ratio=0.6, val_ratio=0.2):
        """
        Args:
            train_ratio (float): Percentage of data for training (e.g., 0.6).
            val_ratio (float): Percentage of data for validation (e.g., 0.2).
            
        Note: The remaining percentage (1.0 - train - val) is assigned to TEST.
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("Train + Validation ratios must be less than 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, df):
        """
        Splits DataFrame into Train, Validation, and Test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe (Must be sorted by date index).
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # 1. Validation: Sort & Check Index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
            
        if not df.index.is_monotonic_increasing:
            print("Warning: Index not sorted. Sorting now...")
            df = df.sort_index()
            
        total_rows = len(df)
        train_end = int(total_rows * self.train_ratio)
        val_end = int(total_rows * (self.train_ratio + self.val_ratio))
        
        # 2. Slice Data (Chronological)
        # We use .copy() to ensure these are independent objects (memory extraction)
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        
        # 3. Verify Integrity
        self._verify_splits(train, val, test)
        
        # 4. Report Ranges
        self._print_stats("TRAIN", train)
        self._print_stats("VALIDATION", val)
        self._print_stats("TEST", test)
        
        return train, val, test

    def _verify_splits(self, train, val, test):
        """Internal method to verify no overlap."""
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError("One of the splits is empty! Check dataset size.")
            
        # Check Date Continuity
        train_max = train.index.max()
        val_min = val.index.min()
        val_max = val.index.max()
        test_min = test.index.min()
        
        # Strict Inequality: Train End < Val Start
        if train_max >= val_min:
            raise AssertionError(f"Train/Val Overlap! Train End: {train_max}, Val Start: {val_min}")
            
        if val_max >= test_min:
            raise AssertionError(f"Val/Test Overlap! Val End: {val_max}, Test Start: {test_min}")

    def _print_stats(self, name, df):
        print(f"[{name:<10}] Rows: {len(df):<6} | {df.index.min()} -> {df.index.max()}")
