import os
import pandas as pd
import lightning.pytorch as pl # <--- CHANGED
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping # <--- CHANGED
from lightning.pytorch.loggers import TensorBoardLogger # <--- CHANGED

from src.config import Config
from src.model_tft import TFTBuilder

def train_model():
    print("--- Starting TFT Training Pipeline ---")
    
    # 1. Load Data
    data_path = os.path.join(Config.DATA_PROCESSED, f"btc_{Config.TIMEFRAME}_processed.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run data_processor.py first.")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Build Datasets
    print("Building TimeSeriesDataSet...")
    builder = TFTBuilder(df)
    training_dataset, validation_dataset = builder.create_dataset()
    
    # Create DataLoaders
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=Config.BATCH_SIZE, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=Config.BATCH_SIZE * 2, num_workers=0
    )
    
    # 3. Initialize Model
    print("Initializing Temporal Fusion Transformer...")
    tft = TFTBuilder.get_model(training_dataset)
    
    # 4. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.MODEL_SAVE_PATH,
        filename='btc_tft_best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )

    logger = TensorBoardLogger("logs", name="tft_btc_trading")

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=Config.EPOCHS,
        accelerator=Config.DEVICE, 
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.05, 
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )
    
    # 6. Fit
    print("Starting training...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print(f"Training Complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train_model()