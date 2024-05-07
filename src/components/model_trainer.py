import os
import sys
from dataclasses import dataclass

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.components.model import CRNN, weights_init
from src.utils import get_dicts, compute_loss
from src.components.data_loader import LoadData
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file: str = os.path.join("artifacts", "model.pth")
    images_path: str = os.path.join("artifacts", "images")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTrainer:
    def __init__(self, model, train_loader, optimizer, lr_scheduler, raw_data_path, clip_norm=5, epochs=100):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.clip_norm = clip_norm
        self.char_to_idx, _ = get_dicts(raw_data_path=raw_data_path)
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        logging.info("Model trainer initiated with total number of epochs: %d" % self.epochs)

        epoch_losses = []
        iteration_losses = []
        num_updates_epochs = []
        self.model = self.model.to(self.config.device)

        try:
            for epoch in tqdm(range(1, self.epochs+1), desc="Epochs"):
                epoch_loss_list = [] 
                num_updates_epoch = 0
                
                for image_batch, text_batch in tqdm(self.train_loader, desc="Batches", leave=False):
                    self.optimizer.zero_grad()
                    text_batch_logits = self.model(image_batch.to(self.config.device))
                    loss = compute_loss(text_batch, text_batch_logits, self.config.device, criterion=nn.CTCLoss(blank=0), char_to_idx=self.char_to_idx)
                    iteration_loss = loss.item()

                    if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                        continue
                    
                    num_updates_epoch += 1
                    iteration_losses.append(iteration_loss)
                    epoch_loss_list.append(iteration_loss)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()

                    logging.info(f"Epoch: {epoch} | Batch Loss: {iteration_loss}")

                epoch_loss = np.mean(epoch_loss_list)
                logging.info(f"Epoch: {epoch} | Loss: {epoch_loss} | NumUpdates: {num_updates_epoch}")
                epoch_losses.append(epoch_loss)
                num_updates_epochs.append(num_updates_epoch)
                
                if self.lr_scheduler:
                    self.lr_scheduler.step(epoch_loss)
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    logging.info("Current learning rate: {}".format(current_lr))

                # Plot and save the graph
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                ax1.plot(epoch_losses)
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Loss")

                ax2.plot(iteration_losses)
                ax2.set_xlabel("Iterations")
                ax2.set_ylabel("Loss")

                if not os.path.exists(self.config.images_path):
                    os.makedirs(self.config.images_path)

                fig.savefig(os.path.join(self.config.images_path, "Loss.png"))
                plt.show()

        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise CustomException("Error occurred during training", sys)
        
        return self.model

    def save_model(self):
        torch.save(self.model, self.config.trained_model_file)

if __name__ == '__main__':
    RAW_DATA = os.path.join("data", "raw_data")
    HIDDEN_SIZE = 256
    EPOCHS = 1
    LR = 0.001
    WEIGHT_DECAY = 1e-3
    CLIP_NORM = 5
    BATCH_SIZE = 32
    DROP_OUTS = 0.1

    char_to_idx, _= get_dicts(RAW_DATA)
    num_chars = len(char_to_idx)

    # Initiate data ingestion
    data_ingestion = DataIngestion(RAW_DATA)
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Initiate data loader
    loader = LoadData(train_path, test_path, batch_size=BATCH_SIZE)
    train_loader, _ = loader.initiate_data_loader()

    # Create model object
    crnn = CRNN(num_chars, rnn_hidden_size=HIDDEN_SIZE, dropout=DROP_OUTS)
    crnn.apply(weights_init)

    # Create optimizer and scheduler
    optimizer = optim.Adam(crnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Initiate model trainer and save the model
    trainer = ModelTrainer(model=crnn, 
                           train_loader=train_loader, 
                           optimizer=optimizer, 
                           lr_scheduler=lr_scheduler, 
                           raw_data_path=RAW_DATA, 
                           epochs=EPOCHS, 
                           clip_norm=CLIP_NORM)
    model = trainer.initiate_model_trainer()
    trainer.save_model()
