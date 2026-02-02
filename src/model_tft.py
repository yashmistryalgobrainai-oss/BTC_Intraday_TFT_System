import pandas as pd
import numpy as np
import lightning.pytorch as pl  # <--- CHANGED THIS LINE
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from src.config import Config

class TFTBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # --- NUCLEAR CLEANUP START ---
        self.df = self.df.astype({'open': 'float32', 'close': 'float32', 'volume': 'float32'})
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        
        # Create IDs
        self.df = self.df.reset_index()
        self.df['time_idx'] = self.df.index
        self.df['group_id'] = 'BTCUSD'
        # --- NUCLEAR CLEANUP END ---

    def create_dataset(self):
        max_encoder_length = Config.INPUT_WINDOW
        max_prediction_length = Config.PREDICT_WINDOW
        
        training_cutoff = self.df["time_idx"].max() - max_prediction_length

        training_dataset = TimeSeriesDataSet(
            self.df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=Config.TARGET,
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2, 
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group_id"],
            time_varying_known_reals=["hour_sin", "hour_cos", "day_of_week", "time_idx"],
            time_varying_unknown_reals=[
                "open", "high", "low", "close", "volume", 
                "returns", "log_returns", "rsi", "macd", "atr"
            ],
            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation=None
            ), 
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, self.df, predict=True, stop_randomization=True
        )
        
        return training_dataset, validation_dataset

    @staticmethod
    def get_model(training_dataset):
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=Config.LEARNING_RATE,
            hidden_size=Config.HIDDEN_SIZE,
            attention_head_size=Config.ATTENTION_HEADS,
            dropout=Config.DROPOUT,
            hidden_continuous_size=Config.HIDDEN_SIZE,
            output_size=7, 
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
        return tft