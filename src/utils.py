import os
import glob
import torch
import shutil
import random

import torch.nn.functional as F
from src.logger import logging

def split_data(source_folder, train_folder, val_folder, split_ratio=0.8):
    all_files = os.listdir(source_folder)
    random.shuffle(all_files)
    num_train = int(len(all_files) * split_ratio)
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
        logging.info(f"Train folder {train_folder} has been deleted")
    if os.path.exists(val_folder):
        shutil.rmtree(val_folder)
        logging.info(f"Test folder {val_folder} has been deleted")

    # os.makedirs(train_folder, exist_ok=True)
    shutil.copytree(source_folder, train_folder, symlinks=True, ignore=shutil.ignore_patterns(*val_files))

    # os.makedirs(val_folder, exist_ok=True)
    shutil.copytree(source_folder, val_folder, symlinks=True, ignore=shutil.ignore_patterns(*train_files))


def count_files(folder):
    return sum(len(files) for _, _, files in os.walk(folder))


def get_dicts(raw_data_path):
    data_names = glob.glob(os.path.join(raw_data_path, "**/*.png"), recursive = True)
    chars = set(os.path.basename(img).removesuffix(".png") for img in data_names)
    letters = sorted(set("".join(chars)))
    vocab = ["-"] + letters
    char_to_idx = {ch:i for i, ch in enumerate(vocab)}
    idx_to_char = {i:ch for i, ch in enumerate(vocab)}

    return char_to_idx, idx_to_char


def compute_loss(text_batch, text_batch_logits, device, criterion, char_to_idx):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.int32).to(device) # [batch_size] 
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch, char_to_idx)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


def encode_text_batch(text_batch, char_to_idx):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
    
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char_to_idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)
    
    return text_batch_targets, text_batch_targets_lens


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def decode_predictions(text_batch_logits, idx_to_char):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx_to_char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new