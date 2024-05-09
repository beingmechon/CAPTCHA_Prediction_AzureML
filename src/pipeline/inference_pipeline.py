import argparse
import sys

import mlflow
from mlflow.models import Model
import mlflow.pytorch as mlflow_pt

sys.path.append("./")

from utils import decode_predictions, correct_prediction, get_dicts
from components.data_ingestion import DataIngestion
from components.data_loader import LoadData
from components.model_evaluator import Evaluator


def log_parameters(args):
    """Log parameters to MLflow."""
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)


def main():
    parser = argparse.ArgumentParser()
    
    # Define required arguments for model training
    parser.add_argument("--data", type=str, help="path to input raw data", required=True)
    parser.add_argument("--model_path", type=str, required=True, help="trained model path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    mlflow_pt.autolog()

    # Evaluate the model
    model_evaluator = Evaluator(args)
    result = model_evaluator.evaluate()
    metrics = model_evaluator.get_metrics(result)

    mlflow.log_metrics(metrics)

    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
