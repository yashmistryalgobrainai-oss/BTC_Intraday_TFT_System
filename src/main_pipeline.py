import os
import sys
import pandas as pd
import logging

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor_v2 import DataProcessorV2
from src.data_splitter import TimeSeriesSplitter
from src.feature_engineer_v2 import TemporalFeatureEngineer
from src.models.xgb_signal_generator import TradingSignalXGB
from src.backtest.realistic_backtest import RealisticBacktester
from src.live_trading_assistant import LiveTradingAssistant
from src.config import Config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemPipeline:
    def __init__(self):
        self.output_model_name = 'trading_assistant_v2.json'
        
    def step_1_data_processing(self):
        """Generates the labeled dataset."""
        logger.info("STEP 1: Data Processing...")
        processor = DataProcessorV2()
        # Note: DataProcessorV2.run_pipeline() currently runs the full labeling logic
        # and saves to btc_15min_v2_processed.parquet
        processor.run_pipeline()
        return os.path.join(Config.DATA_PROCESSED, "btc_15min_v2_processed.parquet")

    def step_2_splitting(self, df):
        """Splits Data safely."""
        logger.info("STEP 2: Time-Series Splitting...")
        splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2)
        train, val, test = splitter.split(df)
        logger.info(f"Split Sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    def step_3_feature_engineering(self, train, val, test):
        """Engineers features without leakage."""
        logger.info("STEP 3: Feature Engineering (No Leakage)...")
        engineer = TemporalFeatureEngineer()
        
        # Fit on Train, Transform others
        train_eng = engineer.fit_transform(train)
        val_eng = engineer.transform(val)
        test_eng = engineer.transform(test)
        
        return train_eng, val_eng, test_eng, engineer.feature_cols

    def step_4_training(self, train, val):
        """Trains the model."""
        logger.info("STEP 4: Model Training...")
        
        # Prepare X and y
        feature_cols = [c for c in train.columns if c not in ['target_class', 'forward_return_1h', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        X_train = train[feature_cols]
        y_train = train['target_class']
        X_val = val[feature_cols]
        y_val = val['target_class']
        
        model = TradingSignalXGB()
        model.train(X_train, y_train, X_val, y_val)
        model.save_model(self.output_model_name)
        
        return model

    def step_5_backtesting(self, model, test_df):
        """Runs realistic backtest."""
        logger.info("STEP 5: Realistic Backtesting...")
        
        # Get feature names from model
        feature_cols = model.feature_names
        X_test = test_df[feature_cols]
        
        # CRITICAL FIX: Get raw probabilities array, not the dict wrapper
        probs = model.model.predict_proba(X_test)
        
        # Run simulation with higher threshold
        backtester = RealisticBacktester(initial_capital=10000)
        metrics = backtester.run(test_df, probs, threshold=0.65)
        
        logger.info("\n" + "="*30)
        logger.info("   FINAL BACKTEST METRICS")
        logger.info("="*30)
        for k, v in metrics.items():
            logger.info(f"{k:<20}: {v}")
            
        return metrics

    def run_full_pipeline(self, retrain=True):
        print("\n=== STARTING TRADING SYSTEM PIPELINE ===\n")
        
        # 1. Load Data
        processed_path = self.step_1_data_processing()
        df = pd.read_parquet(processed_path)
        
        # 2. Split
        train, val, test = self.step_2_splitting(df)
        
        # 3. Engineer
        train, val, test, feats = self.step_3_feature_engineering(train, val, test)
        
        # 4. Train
        if retrain:
            model = self.step_4_training(train, val)
        else:
            logger.info("Skipping training, loading existing model...")
            model = TradingSignalXGB()
            model.load_model(self.output_model_name)
            
        # 5. Backtest
        metrics = self.step_5_backtesting(model, test)
        
        # 6. Initialize Assistant
        logger.info("STEP 6: Initializing Live Assistant...")
        assistant = LiveTradingAssistant(self.output_model_name)
        
        print("\n=== PIPELINE COMPLETE ===\n")
        return assistant, metrics

def interactive_session():
    pipeline = TradingSystemPipeline()
    
    # Run Pipeline
    ans = input("Run full retraining pipeline? (y/n): ").lower()
    retrain = True if ans == 'y' else False
    
    assistant, metrics = pipeline.run_full_pipeline(retrain=retrain)
    
    if metrics['Total Trades'] == 0:
        print("\n[!] Warning: No trades in backtest. Model might be too conservative.")
    
    # Live Mode
    print("\n" + "*"*40)
    print("      LIVE TRADING DASHBOARD")
    print("*"*40)
    
    try:
        cap_str = input("Enter Trading Capital ($): ")
        capital = float(cap_str)
    except:
        capital = 1000.0
        print("Invalid input. Defaulting to $1,000.")
        
    print(f"\nAnalyzing current market for ${capital:,.2f} account...")
    assistant.analyze_market(capital)
    
    start_monitor = input("\nStart Continuous Monitoring? (y/n): ").lower()
    if start_monitor == 'y':
        assistant.run_live_monitor(capital, interval=60)
    else:
        print("Exiting.")

if __name__ == "__main__":
    interactive_session()
