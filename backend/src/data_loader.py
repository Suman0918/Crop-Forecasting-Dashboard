# Market/backend/src/data_loader.py

import pandas as pd
import os

# Directory where your CSVs live:
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))


def load_crop_data(crop_name):
    """
    Loads {crop_name}_price.csv from data/ folder.
    Expects columns: "date", "price".
    Returns a DataFrame sorted by date ascending.
    """
    file_path = os.path.join(DATA_DIR, f"{crop_name}_price.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for crop '{crop_name}' not found at {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
