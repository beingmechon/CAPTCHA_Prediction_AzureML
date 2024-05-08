import os
import sys

sys.path.append('./')

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from logger import logging
from exception import CustomException
from components.data_ingestion import DataIngestion


class CAPTCHADataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        
    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, index):
        image_fn = os.listdir(self.data_path)[index]
        image_fp = os.path.join(self.data_path, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        text = image_fn.split(".")[0]
        return image, text
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
    


class LoadData:
    def __init__(self, train_path, test_path, batch_size=32):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

    def initiate_data_loader(self):
        try:
            logging.info("Loading data. Training data: {}, Testing data: {}".format(self.train_path, self.test_path))
            train_dataset = CAPTCHADataset(self.train_path)
            test_dataset = CAPTCHADataset(self.test_path)
            logging.info("Dataset Created.")

            logging.info(f"Creating Data Loader with Batch size of {self.batch_size}.")
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
            logging.info("Data Loader Created.")

            return train_loader, test_loader
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    dataIngestion = DataIngestion(os.path.join("data", "raw_data"))
    train_path, test_path = dataIngestion.initiate_data_ingestion()

    loader = LoadData(train_path, test_path)
    train_loader, test_loader = loader.initiate_data_loader()
    print(len(next(iter(train_loader))[0]))
    print(len(next(iter(test_loader))[0]))