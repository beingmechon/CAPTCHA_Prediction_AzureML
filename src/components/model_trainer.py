import sys
import torch
import mlflow
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
import mlflow.pytorch as mlflow_pt
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# sys.path.append("../")
sys.path.append("./")

from logger import logging
from exception import CustomException
from utils import get_dicts, compute_loss, decode_predictions
from components.data_loader import LoadData
from components.model import CRNN, weights_init
from components.data_ingestion import DataIngestion


class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self):        
        for arg, value in vars(self.args).items():
            mlflow.log_param(arg, value)
        
        char_to_idx, idx_to_char = get_dicts(self.args.data)
        num_chars = len(char_to_idx)
        
        data_ingestion = DataIngestion(self.args.data, random_seed=self.args.random_seed, split_ratio=self.args.split_ratio)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        loader = LoadData(train_path, test_path, batch_size=self.args.batch_size)
        train_loader, _ = loader.initiate_data_loader()
        
        crnn = CRNN(num_chars, rnn_hidden_size=self.args.hidden_size, dropout=self.args.drop_out)
        crnn.apply(weights_init)
        
        optimizer = optim.Adam(crnn.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.args.patience)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        crnn = crnn.to(device)
        criterion = nn.CTCLoss(blank=0)
        
        epoch_losses = []
        iteration_losses = []
        val_losses = []
        num_updates_epochs = []
        
        try:
            for epoch in tqdm(range(1, self.args.epochs+1), desc="Epochs"):
                epoch_loss_list = [] 
                val_epoch_loss_list = []
                num_updates_epoch = 0
                
                for image_batch, text_batch in tqdm(train_loader, desc="Batches", leave=False):
                    crnn.train()
                    text_batch_logits = crnn(image_batch.to(device))
                    loss = compute_loss(text_batch, text_batch_logits, device, criterion, char_to_idx)
                    iteration_loss = loss.item()
                    optimizer.zero_grad()

                    if not torch.isfinite(loss):
                        continue

                    num_updates_epoch += 1
                    iteration_losses.append(iteration_loss)
                    epoch_loss_list.append(iteration_loss)

                    loss.backward()
                    nn.utils.clip_grad_norm_(crnn.parameters(), self.args.clip_norm)
                    optimizer.step()

                    crnn.eval()
                    with torch.no_grad():
                        text_batch_logits = crnn(image_batch)
                        text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx_to_char)

                        val_loss = compute_loss(text_batch, text_batch_logits, device, criterion, char_to_idx)
                        iteration_val_loss = val_loss.item()
                        val_epoch_loss_list.append(iteration_val_loss)


                    accuracy = accuracy_score(text_batch, text_batch_pred)
                    precision = precision_score(text_batch, text_batch_pred, average='weighted')
                    recall = recall_score(text_batch, text_batch_pred, average='weighted')
                    f1 = f1_score(text_batch, text_batch_pred, average='weighted')

                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("F1-Score", f1)
                    mlflow.log_metric("Batch Loss", iteration_loss)

                epoch_loss = torch.tensor(epoch_loss_list).mean().item()
                val_loss = torch.tensor(val_epoch_loss_list).mean().item()

                mlflow.log_metric("Epoch Loss", epoch_loss)
                mlflow.log_metric("Validation Loss", val_loss)
                mlflow.log_metric("Num Updates", num_updates_epoch)

                epoch_losses.append(epoch_loss)
                val_losses.append(val_loss)
                num_updates_epochs.append(num_updates_epoch)
                
                if lr_scheduler:
                    lr_scheduler.step(epoch_loss)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    print("Current learning rate: {}".format(current_lr))
                    # mlflow.log_metric("current_lr", current_lr)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

            ax1.plot(epoch_losses)
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.set_title("Epochs vs Losses")

            ax2.plot(iteration_losses)
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Loss")
            ax2.set_title("Iterations vs Loss")

            ax3.plot(val_losses)
            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Validation Loss")
            ax3.set_title("Epochs vs Validation Loss")

            mlflow.log_figure(fig, "Losses.png")

        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise CustomException("Error occurred during training", sys)
        
        # Signature
        # signature = infer_signature(train_loader)

        model_info = mlflow_pt.log_model(crnn, "model") #, signature=signature)
        # mlflow.end_run()

        return model_info

