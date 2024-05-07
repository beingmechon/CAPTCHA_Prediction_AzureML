import os
import sys
from src.exception import CustomException
from src.logger import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass

from src.utils import split_data, count_files

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train_data")
    test_data_path: str = os.path.join("artifacts", "test_data")


class DataIngestion:
    def __init__(self, raw_data_path, random_seed=42, split_ratio=0.8) -> None:
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path = raw_data_path
        self.random_seed = random_seed
        self.split_ratio = split_ratio

    def initiate_data_ingestion(self) -> None:
        logging.info("Data ingestion initiated")
        try:
            # images_path = "../notebook/data/images"
            logging.info(f"Raw data path: %s" % self.raw_data_path)
            logging.info("Data split initiated")
            train_path = self.ingestion_config.train_data_path
            test_path = self.ingestion_config.test_data_path
            logging.info("Train Data Path: %s" % train_path)
            logging.info("Test Data Path: %s" % test_path)

            split_data(self.raw_data_path, train_path, test_path, random_seed=self.random_seed, split_ratio=self.split_ratio)
            logging.info(f"Data ingestion and split completed. Total file in train set: {count_files(train_path)}. Total file in test set: {count_files(test_path)}")
            
            return train_path, test_path
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    dataIngestion = DataIngestion(os.path.join("data", "raw_data"))
    dataIngestion.initiate_data_ingestion()
