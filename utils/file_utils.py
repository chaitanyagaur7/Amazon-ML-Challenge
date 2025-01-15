import os
import pandas as pd
import logging

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory created: {path}")

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(data, output_path):
    pd.DataFrame(data).to_csv(output_path, index=False)
