import sys
import torch
import mlflow
import pandas as pd
from tqdm import tqdm
from mlflow.models import Model
import mlflow.pytorch as mlflow_pt
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_curve, 
                             auc)

sys.path.append("./")

from logger import logging
from exception import CustomException
from utils import get_dicts, decode_predictions, correct_prediction
from components.data_loader import LoadData
from components.data_ingestion import DataIngestion


class Evaluator():
    def __init__(self, args):
        self.args = args

    def evaluate(self):        
        for arg, value in vars(self.args).items():
            mlflow.log_param(arg, value)
        
        _, idx_to_char = get_dicts(self.args.data)
        
        data_ingestion = DataIngestion(self.args.data, random_seed=self.args.random_seed, split_ratio=self.args.split_ratio)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        loader = LoadData(train_path, test_path, batch_size=self.args.batch_size)
        _, test_loader = loader.initiate_data_loader()
        
        model = mlflow_pt.load_model(self.args.model_path)
        print("model loaded")
        print(type(model))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        results_test = pd.DataFrame(columns=['actual', 'prediction'])
        try:              
            for image_batch, text_batch in tqdm(test_loader, desc="Batches", leave=False):
                model.eval()
                with torch.no_grad():
                    text_batch_logits = model(image_batch)
                    text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx_to_char)

            df = pd.DataFrame({'actual': text_batch, 'prediction': text_batch_pred})
            results_test = pd.concat([results_test, df], ignore_index=True)

            results_test['prediction_corrected'] = results_test['prediction'].apply(correct_prediction)

        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise CustomException("Error occurred during training", sys)
        
        return results_test
        
    def get_metrics(self, results_test):
        if 'prediction_corrected' not in results_test.columns:
            raise CustomException("prediction_corrected is not found in results_test")
        
        actual = results_test['actual']
        predicted = results_test['prediction_corrected']
        
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, average='weighted', zero_division=1)
        recall = recall_score(actual, predicted, average='weighted', zero_division=1)
        f1 = f1_score(actual, predicted, average='weighted')

        # mlflow.log_params({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

