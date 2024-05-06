import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.components.model import CRNN
from src.exception import CustomException
from src.utils import decode_predictions, correct_prediction, get_dicts
from src.components.data_ingestion import DataIngestion
from src.components.data_loader import LoadData


@dataclass
class ModelEvaluatorConfig:
    trained_model_file: str = os.path.join("artifacts", "model.pth")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelEvaluation:
    def __init__(self, data_loader, raw_data_path) -> None:
        self.config = ModelEvaluatorConfig()
        self.data_loader = data_loader
        _, self.idx_to_char = get_dicts(raw_data_path=raw_data_path)
        self.model = torch.load(self.config.trained_model_file)

    def initiate_model_evaluater(self):
        logging.info("Model Evaluate initiated.")
        self.results_test = pd.DataFrame(columns=['actual', 'prediction'])

        try:
            with torch.no_grad():
                for image_batch, text_batch in tqdm(self.data_loader, desc="Evaluation Progress", leave=True):
                    text_batch_logits = self.model(image_batch.to(self.config.device))
                    text_batch_pred = decode_predictions(text_batch_logits.cpu(), self.idx_to_char)
                    df = pd.DataFrame({'actual': text_batch, 'prediction': text_batch_pred})
                    self.results_test = pd.concat([self.results_test, df], ignore_index=True)

            self.results_test['prediction_corrected'] = self.results_test['prediction'].apply(correct_prediction)
            logging.info("Model evaluation completed.")
            
            return self.results_test
        
        except Exception as e:
            logging.error(f"Error occurred during model evaluation: {e}")
            raise CustomException(f"Error occurred during model evaluation {e}", sys)

    def get_accuracy(self):
        if 'prediction_corrected' not in self.results_test.columns:
            logging.warning("Corrected predictions not found. Initiating model evaluation.")
            self.initiate_model_evaluater()

        return accuracy_score(self.results_test['actual'], self.results_test['prediction_corrected'])


if __name__ == '__main__':
    RAW_DATA = "./data/raw_data"

    dataIngestion = DataIngestion(os.path.join("data", "raw_data"))
    train_path, test_path = dataIngestion.initiate_data_ingestion()

    dataLoader = LoadData(train_path, test_path)
    train_loader, test_loader = dataLoader.initiate_data_loader()

    modelEvaluater = ModelEvaluation(test_loader, raw_data_path=RAW_DATA)
    modelEvaluater.initiate_model_evaluater()
    print(modelEvaluater.get_accuracy())