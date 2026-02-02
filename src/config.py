import os
import torch

class Config:
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw', 'btc_1m.csv')
    DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')
    
    # --- Data Settings ---
    TIMEFRAME = '15min'  # Primary trading timeframe (can be '30min', '1H')
    INPUT_WINDOW = 60    # Lookback: Past 60 candles (e.g., 15h of context)
    PREDICT_WINDOW = 4   # Forecast: Next 4 candles (e.g., next 1 hour)
    
    # --- Feature Engineering ---
    # Features we will calculate and feed into the AI
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns',
        'rsi', 'macd', 'macd_signal', 'atr',
        'hour_sin', 'hour_cos', 'day_of_week' # Time embeddings
    ]
    TARGET = 'target_return' # We will predict the return of the next candle
    
    # --- Model Hyperparameters (TFT) ---
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.0003
    DROPOUT = 0.1
    HIDDEN_SIZE = 64     # Size of LSTM/Attention layers
    ATTENTION_HEADS = 4
    
    # --- Trading Logic ---
    RISK_REWARD_RATIO = 1.5
    CONFIDENCE_THRESHOLD = 0.75
    
    # --- Hardware ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create directories if they don't exist
os.makedirs(Config.DATA_PROCESSED, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)