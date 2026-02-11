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
        """Generates curated list of indicators (Trend, Momentum, Volatility, Volume)."""
        df = df.copy()
        
        # --- 1. TREND (EMA 20/50) ---
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()  # Long-term trend
        df['trend_signal'] = df['ema_20'] - df['ema_50'] # Positive = Uptrend
        
        # --- 2. MOMENTUM (RSI, MACD) ---
        ti = TechnicalIndicators()
        df['rsi'] = ti.get_rsi(df['close'])
        df['macd'], df['macd_signal'] = ti.get_macd(df['close'])
        
        # --- 3. VOLATILITY (ATR, BB) ---
        df['atr'] = ti.get_atr(df)
        df['atr_raw'] = df['atr'] # Preserve for dollar-based risk management
        
        # Bollinger Bands (20, 2)
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] # Normalized width
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # --- 4. VOLUME (OBV, Rel Vol) ---
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Relative Volume (Vol / 20-period Avg Vol)
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
        df['rel_volume'] = df['volume'] / df['vol_ma_20']
        # Replace Inf/NaN in rel_volume logic (start of series)
        df['rel_volume'] = df['rel_volume'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        return df

    def _add_derived_features(self, df):
        """Generates Support/Resistance, Lags, and Time features."""
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # --- 5. STRUCTURE (Support/Resistance) ---
        # 24H High/Low lookback
        df['resistance_24h'] = df['high'].rolling(window=24).max()
        df['support_24h'] = df['low'].rolling(window=24).min()
        
        # Distance to S&R (Normalized)
        df['dist_to_res'] = (df['close'] - df['resistance_24h']) / df['close']
        df['dist_to_sup'] = (df['close'] - df['support_24h']) / df['close']
        
        # 1. Time Features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 2. Lag Features (Reduced set)
        features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'bb_position', 'obv']
        
        for col in features_to_lag:
            if col not in df.columns: continue
            for lag in [1, 2, 3]:  # Only 3 hours lookback
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
        exclude_cols = ['target_class', 'forward_return_1h', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_raw', 'ema_200']
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
