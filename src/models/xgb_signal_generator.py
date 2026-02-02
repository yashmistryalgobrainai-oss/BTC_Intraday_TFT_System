import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from src.config import Config

class TradingSignalXGB:
    """
    XGBoost Classifier tuned for signal generation with strict regularization against overfitting.
    
    Why this config?
    - Low depth (4) prevents memorizing noise.
    - min_child_weight (5) ensures leaves represent significant patterns, not outliers.
    - gamma (0.1) requires a minimum loss reduction to split.
    - subsample/colsample (0.7) adds randomness (bagging) to reduce variance.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def build_model(self):
        """Initialize the model architecture."""
        self.model = xgb.XGBClassifier(
            n_estimators=150,        # Reduced from 800 to prevent overfitting on noise
            max_depth=4,             # Shallow trees force generalization
            learning_rate=0.05,      # Slower learning = better convergence
            min_child_weight=5,      # High value stops splitting on tiny samples
            gamma=0.1,               # Pruning parameter
            reg_alpha=0.05,          # L1 Regularization (Lasso)
            reg_lambda=1.0,          # L2 Regularization (Ridge)
            subsample=0.7,           # Train on random 70% of rows per tree
            colsample_bytree=0.7,    # Train on random 70% of features per tree
            objective='multi:softprob',
            num_class=3,             # Classes: 0 (Hold), 1 (Buy), 2 (Sell)
            eval_metric='mlogloss',
            # Hardware settings
            tree_method='hist',
            device='cuda',           # Use GPU
            verbosity=1
        )
        print("Model initialized with conservative hyperparameters.")
        
    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the model with Class Weights and Early Stopping.
        """
        if self.model is None:
            self.build_model()
            
        print(f"Training on {len(X_train)} samples...")
        self.feature_names = list(X_train.columns)
        
        # 1. Handle Class Imbalance
        # Crypto data often has 90% HOLD, 5% BUY, 5% SELL.
        # Without weights, model will just predict HOLD forever (90% accuracy, 0 profit).
        # Sample weights force it to pay attention to the rare BUY/SELL signals.
        print("Computing Class Weights...")
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # 2. Fit
        self.model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            verbose=20
        )
        
        # 3. Report
        print("\n--- Validation Report ---")
        val_preds = self.model.predict(X_val)
        print(classification_report(y_val, val_preds, target_names=['HOLD', 'BUY', 'SELL']))
        
    def predict_signal(self, X_input):
        """
        Predicts signal for new data (can be single row or dataframe).
        
        Returns:
            dict: {'signal': int, 'confidence': float, 'probs': [p0, p1, p2]}
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Ensure column order matches training
        X_input = X_input[self.feature_names]
        
        probs = self.model.predict_proba(X_input)
        
        # For single row usage (Live Trading)
        if len(probs) == 1:
            p = probs[0]
            predicted_class = np.argmax(p)
            confidence = p[predicted_class]
            
            return {
                'signal': int(predicted_class),    # 0, 1, or 2
                'confidence': float(confidence),
                'probs': {
                    'HOLD': float(p[0]),
                    'BUY': float(p[1]),
                    'SELL': float(p[2])
                }
            }
        
        # For batch usage (Backtesting)
        return probs

    def get_feature_importance(self):
        """Returns feature importance dataframe."""
        if self.model is None:
            return None
            
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return importance

    def save_model(self, filename='btc_xgb_v2.json'):
        path = os.path.join(Config.MODEL_SAVE_PATH, filename)
        print(f"Saving model to {path}...")
        self.model.save_model(path)

    def load_model(self, filename='btc_xgb_v2.json'):
        path = os.path.join(Config.MODEL_SAVE_PATH, filename)
        print(f"Loading model from {path}...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        # Recover feature names if possible (XGBoost JSON stores them)
        try:
            self.feature_names = self.model.get_booster().feature_names
            if not self.feature_names:
                print("Warning: Could not recover feature names from model file.")
        except:
            pass
