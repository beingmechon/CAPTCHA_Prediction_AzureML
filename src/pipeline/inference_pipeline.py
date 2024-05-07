import argparse
import mlflow
import mlflow.pytorch as mlflow_pt

from src.utils import decode_predictions, correct_prediction, get_dicts
from src.components.data_ingestion import DataIngestion
from src.components.data_loader import LoadData
from src.components.model_evaluator import ModelEvaluation


def log_parameters(args):
    """Log parameters to MLflow."""
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)


def main():
    parser = argparse.ArgumentParser()
    
    # Define required arguments for model training
    parser.add_argument("--data", type=str, help="path to input raw data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    mlflow_pt.autolog()

    # Log parameters to MLflow
    log_parameters(args)

    # Initiate data ingestion
    data_ingestion = DataIngestion(args.data, random_seed=args.random_seed, split_ratio=args.split_ratio)
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Initiate data loader
    loader = LoadData(train_path, test_path, batch_size=args.batch_size)
    _, test_loader = loader.initiate_data_loader()

    # Evaluate the model
    model_evaluator = ModelEvaluation(test_loader, raw_data_path=args.data)
    model_evaluator.initiate_model_evaluator()
    metrics = model_evaluator.get_metrics()

    mlflow.log_metrics(metrics)

    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
