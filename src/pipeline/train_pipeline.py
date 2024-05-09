import sys
import argparse

import mlflow
import mlflow.pytorch as mlflow_pt

# sys.path.append("../")
sys.path.append("./")

from components.model_trainer import Trainer

def main():

    mlflow.start_run()
    # mlflow.set_registry_uri("http://127.0.0.1:5000")
    print(mlflow.get_registry_uri())
    mlflow_pt.autolog()

    parser = argparse.ArgumentParser()
    
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

    trainer = Trainer(args)
    model_info = trainer.train()
    print(model_info.model_uri)
    
    mlflow.end_run()


if __name__ == "__main__":
    main()
