import os
import argparse
import mlflow
import torch.optim as optim
import mlflow.pytorch as mlflow_pt

from src.utils import get_dicts
from src.components.data_loader import LoadData
from src.components.model import CRNN, weights_init
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

def main():
    parser = argparse.ArgumentParser()
    
    # Define required arguments for model training
    parser.add_argument("--data", type=str, help="path to input raw data")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=5)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    mlflow_pt.autolog()

    # Log parameters to MLflow
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)

    char_to_idx, _ = get_dicts(args.data)
    num_chars = len(char_to_idx)

    # Initiate data ingestion
    data_ingestion = DataIngestion(args.data, random_seed=args.random_seed, split_ratio=args.split_ratio)
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Initiate data loader
    loader = LoadData(train_path, test_path, batch_size=args.batch_size)
    train_loader, _ = loader.initiate_data_loader()

    # Create model object
    crnn = CRNN(num_chars, rnn_hidden_size=args.hidden_size, dropout=args.drop_out)
    crnn.apply(weights_init)

    # Create optimizer and scheduler
    optimizer = optim.Adam(crnn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience)

    # Initiate model trainer and save the model
    trainer = ModelTrainer(model=crnn, 
                           train_loader=train_loader, 
                           optimizer=optimizer, 
                           lr_scheduler=lr_scheduler, 
                           raw_data_path=args.data, 
                           epochs=args.epochs,  
                           clip_norm=args.clip_norm)
    
    model = trainer.initiate_model_trainer()
    trainer.save_model()

    # Save model to MLflow
    mlflow_pt.log_model(model, "model")
    
    # End MLflow run
    mlflow.end_run()

if __name__ == "__main__":
    main()
