import os
import shutil
import random

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
