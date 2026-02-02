import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from src.config import Config

# --- Fix 1: Add Lags Separate from Data Processor ---
def add_lag_features(df):
    """
    Applied to the dataframe after loading.
    Must handle NaNs created by shifting.
    """
    df = df.copy()
    features_to_lag = ['close', 'volume', 'rsi', 'macd', 'atr', 'returns']
    
    # Lags 1, 2, 3
    for col in features_to_lag:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # Rolling Volatility
    df['rolling_volatility'] = df['returns'].rolling(window=12).std()
    
    df.dropna(inplace=True)
    return df

def train_xgb_enhanced():
    print("--- Starting Enhanced XGBoost Training ---")
    
    # 1. Load Data
    data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Add Lag Features (Feature Engineering)
    # FIX: Applying this BEFORE split, but since it's just lagging past values, it's generally OK 
    # as long as we drop the NaNs at the start.
    print("Adding Lag Features...")
    df = add_lag_features(df)
    
    # --- Fix 3: Target Threshold Adjustment ---
    # We want broader moves. 0.005 = 0.5%.
    THRESHOLD = 0.005
    print(f"Applying Target Threshold: +/- {THRESHOLD*100}%")
    
    # Re-calculate target class based on new threshold if not already done in data_processor_v2
    # Note: data_processor_v2 DOES label it, but we can override here if needed.
    # We assume 'target_class' exists.
    
    # Select Features
    exclude = ['target_class', 'target_return', 'forward_return', 'forward_return_1h', 'datetime', 'group_id', 'time_idx', 'future_close_1h']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
    
    print(f"Training on {len(feature_cols)} features.")

    # 3. Create Splits (Train/Val/Test)
    # Fix 4: Add Validation Set
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_class']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target_class']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target_class']
    
    print(f"Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. Compute Sample Weights
    print("Computing Class Weights...")
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 5. Initialize Model
    # Fix 2: Reduced Complexity
    model = xgb.XGBClassifier(
        n_estimators=150,        # Reduced from 800
        max_depth=4,             # Reduced from 8
        learning_rate=0.05,
        min_child_weight=5,
        gamma=0.1,               # Regularization
        reg_alpha=0.05,          # L1 Reg
        reg_lambda=1.0,          # L2 Reg
        subsample=0.7,
        colsample_bytree=0.7,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        device='cuda',
        eval_metric='mlogloss',
        early_stopping_rounds=20 # Fix 6: Early Stopping
    )
    
    # 6. Train
    print("Training XGBoost Model...")
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # 7. Evaluate
    print("\n--- Test Set Evaluation ---")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=['HOLD', 'BUY', 'SELL']))
    
    # Fix 5: Feature Importance
    print("\n--- Top 10 Features ---")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(importance.head(10))
    
    # Plot Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title("XGBoost Feature Importance")
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'feature_importance.png'))
    print("Feature importance plot saved.")

    # 8. Save
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'btc_xgb_enhanced.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgb_enhanced()