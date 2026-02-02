import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from src.config import Config

def prepare_data(df):
    """
    Convert Regression (Continuous Returns) into Classification (Buy/Sell/Hold)
    """
    df = df.copy()
    
    # --- 1. Define The Target ---
    # We want to catch moves bigger than 0.2% (covers 0.1% fee + profit)
    THRESHOLD = 0.002 
    
    df['target_class'] = 0  # Default: HOLD
    df.loc[df['target_return'] > THRESHOLD, 'target_class'] = 1   # BUY
    df.loc[df['target_return'] < -THRESHOLD, 'target_class'] = 2  # SELL
    
    # --- 2. Select Features ---
    # We use the same indicators, but XGBoost handles them raw (no scaling needed)
    features = [
        'open', 'high', 'low', 'close', 'volume', 
        'rsi', 'macd', 'macd_signal', 'atr', 
        'hour_sin', 'hour_cos', 'day_of_week',
        'returns', 'log_returns'
    ]
    
    # Drop rows where target is NaN (end of file)
    df.dropna(subset=['target_class', 'target_return'] + features, inplace=True)
    
    return df, features

def train_xgb():
    print("--- Starting XGBoost Training ---")
    
    # 1. Load Data
    data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Prepare Labels
    df, feature_cols = prepare_data(df)
    
    # 3. Train/Test Split (Time Series Split - No Shuffling!)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_class']
    X_test = test_df[feature_cols]
    y_test = test_df['target_class']
    
    # Check Class Balance
    print("\nTraining Class Distribution:")
    print(y_train.value_counts(normalize=True))
    # If 1 (Buy) is < 5%, we might need to adjust weights, but XGBoost handles this well.

    # 4. Initialize XGBoost Classifier
    # using 'gpu_hist' to utilize your RTX 4060
    model = xgb.XGBClassifier(
        n_estimators=500,        # Number of trees
        learning_rate=0.05,      # Step size
        max_depth=6,             # Tree depth (prevent overfitting)
        subsample=0.8,           # Use 80% of data per tree
        colsample_bytree=0.8,    # Use 80% of features per tree
        objective='multi:softprob', # Multiclass probability
        num_class=3,             # 3 Classes: Hold, Buy, Sell
        tree_method='hist',      # Optimized for speed
        device='cuda',           # ENABLE GPU
        eval_metric='mlogloss',
        early_stopping_rounds=50
    )
    
    # 5. Train
    print("\nTraining Model on GPU...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # 6. Evaluate
    print("\n--- Evaluation on Test Set ---")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=['HOLD', 'BUY', 'SELL']))
    
    # 7. Save
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgb()