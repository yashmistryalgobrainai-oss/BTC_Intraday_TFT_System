import pandas as pd
import numpy as np
from src.indicators import TechnicalIndicators

class TemporalFeatureEngineer:
    """
    Advanced Feature Engineer that handles Temporal Validation correctly.
    
    Prevents Data Leakage by:
    1. Calculating rolling windows respecting time order (pandas default).
    2. Storing normalization statistics (mean/std) from the TRAINING set.
    3. Applying saved statistics to the TEST set (never fitting on test).
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_cols = []
        self.is_fitted = False
        
    def _add_technical_indicators(self, df):
        """Generates indicators (RSI, MACD, ATR, BB)."""
        df = df.copy()
        
        # 1. Existing Indicators (Reuse Logic)
        ti = TechnicalIndicators()
        df['rsi'] = ti.get_rsi(df['close'])
        df['macd'], df['macd_signal'] = ti.get_macd(df['close'])
        df['atr'] = ti.get_atr(df)
        df['atr_raw'] = df['atr'] # Preserve for dollar-based risk management
        df = ti.add_time_features(df)
        
        # 2. Bollinger Bands (20, 2)
        # BB works on price, so it's not stationary, but relative position (B% or bandwidth) is.
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        # Feature: Price position relative to BB (0 to 1) - cleaner than raw bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df

    def _add_derived_features(self, df):
        """Generates Lags, Rolling Volatility, and Time features."""
        
        # Returns (Base for volatility)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Time Features (Cyclical)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 2. Rolling Volatility
        # Volatility over 1h (4 candles of 15m) and 4h (16 candles)
        df['volatility_1h'] = df['returns'].rolling(window=4).std()
        df['volatility_4h'] = df['returns'].rolling(window=16).std()
        
        # 3. Lag Features
        # "What happened 15m ago, 30m ago...?"
        features_to_lag = ['close', 'volume', 'rsi', 'returns', 'atr', 'bb_position']
        
        for col in features_to_lag:
            if col not in df.columns: continue
            for lag in [1, 2, 3]:  # Only 3 hours lookback for 1H candles
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
                
        return df

    def fit_transform(self, df):
        """
        Processes training data AND learns scaling parameters.
        
        Args:
            df (pd.DataFrame): Training data.
            
        Returns:
            pd.DataFrame: Processed and normalized training data.
        """
        print("Feature Engineering: FITTING on Training Data...")
        
        # 1. Generate Raw Features
        df = self._add_technical_indicators(df)
        df = self._add_derived_features(df)
        
        # 2. Clean NaNs created by rolling/shifting
        # IMPORTANT: We drop rows here, which changes the shape. 
        # For training, this is fine.
        df.dropna(inplace=True)
        
        # 3. Identify Numeric Columns to Normalize
        # We exclude targets & raw IDs
        exclude_cols = ['target_class', 'forward_return_1h', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_raw']
        # Note: We keep raw OHLV un-scaled usually, OR we scale them. 
        # If model expects scaled, we scale. Let's scale everything valid except targets.
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # 4. Fit & Transform (Z-Score Standardization)
        for col in self.feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            # Store stats
            self.scalers[col] = {'mean': mean, 'std': std}
            
            # Apply
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0 # Handle constant columns
                
        self.is_fitted = True
        print(f"Feature Engineering Complete. Fitted {len(self.feature_cols)} features.")
        return df

    def transform(self, df):
        """
        Processes test/validation data using PRE-LEARNED scaling parameters.
        Does NOT recalculate mean/std (prevents leakage).
        
        Args:
            df (pd.DataFrame): Test data.
            
        Returns:
            pd.DataFrame: Processed and normalized test data.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted on training data first!")
            
        print("Feature Engineering: TRANSFORMING Test Data (Using Train Stats)...")
        
        # 1. Generate Raw Features
        df = self._add_technical_indicators(df)
        df = self._add_derived_features(df)
        
        # 2. Clean NaNs
        df.dropna(inplace=True)
        
        # 3. Apply Saved Scaling
        for col in self.feature_cols:
            if col not in df.columns:
                print(f"Warning: Feature {col} missing in test data. Filling 0.")
                df[col] = 0
                continue
                
            stats = self.scalers.get(col)
            if stats:
                if stats['std'] != 0:
                    df[col] = (df[col] - stats['mean']) / stats['std']
                else:
                    df[col] = 0.0
                    
        return df

    def get_feature_names(self):
        """Returns the list of features trained on."""
        return self.feature_cols
