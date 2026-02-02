import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from src.config import Config

def add_lag_features(df):
    """
    Give the model memory.
    Instead of just seeing "Current RSI is 70", 
    it will see "RSI was 60, now 70" (Trend is UP).
    """
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    
    # Add Lag 1, 2, 3 (Past 45 mins context)
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # Add Rolling Features (Volatility context)
    df['rolling_volatility'] = df['returns'].rolling(window=12).std() # 3 hour vol
    
    df.dropna(inplace=True)
    return df

def prepare_data_enhanced(df):
    df = add_lag_features(df)
    
    # --- TARGET DEFINITION ---
    # We lower the bar slightly to 0.15% to get more trade data
    THRESHOLD = 0.0015 
    
    df['target_class'] = 0 
    df.loc[df['target_return'] > THRESHOLD, 'target_class'] = 1   # BUY
    df.loc[df['target_return'] < -THRESHOLD, 'target_class'] = 2  # SELL
    
    # Select all numeric features including new lags
    exclude = ['target_return', 'target_class', 'datetime', 'group_id', 'time_idx']
    features = [c for c in df.columns if c not in exclude]
    
    return df, features

def train_xgb_enhanced():
    print("--- Starting Enhanced XGBoost Training ---")
    
    # 1. Load
    data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    df = pd.read_parquet(data_path)
    
    # 2. Prepare
    print("Engineering Lag Features...")
    df, feature_cols = prepare_data_enhanced(df)
    
    # 3. Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_class']
    X_test = test_df[feature_cols]
    y_test = test_df['target_class']
    
    # 4. COMPUTE SAMPLE WEIGHTS (The Secret Sauce)
    # This automatically calculates that Class 1 (Buy) is rare 
    # and assigns it a higher weight (e.g., 5.0) vs Class 0 (Hold)
    print("Computing Class Weights to fix imbalance...")
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 5. Initialize Model
    # We increase depth slightly to handle the extra features
    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,      # Slower learning for better generalization
        max_depth=8,             # Deeper trees for complex lag interactions
        subsample=0.7,
        colsample_bytree=0.7,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        device='cuda',
        eval_metric='mlogloss',
        early_stopping_rounds=50
    )
    
    # 6. Train with Weights
    print("Training on GPU with Class Weights...")
    model.fit(
        X_train, y_train,
        sample_weight=weights, # <--- KEY CHANGE
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 7. Evaluate
    print("\n--- Evaluation (Enhanced) ---")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=['HOLD', 'BUY', 'SELL']))
    
    # 8. Save
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_enhanced.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgb_enhanced()