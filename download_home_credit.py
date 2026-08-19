"""
Download Home Credit Default Risk dataset from Kaggle.
Requires kaggle.json credentials.

Run: python download_home_credit.py
"""

import kagglehub
import shutil
import os
import json

# Load Kaggle credentials
kaggle_json_path = r"d:\DavidBroPrj\kaggle.json"
if os.path.exists(kaggle_json_path):
    with open(kaggle_json_path) as f:
        creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

dest_dir = r"d:\DavidBroPrj\credit-risk-project\data\raw"
os.makedirs(dest_dir, exist_ok=True)

print("Downloading Home Credit Default Risk dataset...")
try:
    path = kagglehub.competition_download("home-credit-default-risk")
    print(f"Downloaded to: {path}")

    print(f"Copying files to {dest_dir}...")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("Home Credit data placed in data/raw/")
except Exception as e:
    print(f"Could not download via competition API: {e}")
    print("Trying dataset download instead...")
    try:
        path = kagglehub.dataset_download("home-credit-default-risk/home-credit-default-risk")
        print(f"Downloaded to: {path}")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Home Credit data placed in data/raw/")
    except Exception as e2:
        print(f"Download failed: {e2}")
        print("\nPlease download manually from:")
        print("  https://www.kaggle.com/competitions/home-credit-default-risk/data")
        print("Place application_train.csv, bureau.csv, bureau_balance.csv in data/raw/")
