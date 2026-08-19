import kagglehub
import shutil
import os
import json

# Load Kaggle credentials from the provided json
with open(r'd:\DavidBroPrj\kaggle.json') as f:
    creds = json.load(f)

# Set environment variables for Kaggle auth
os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY'] = creds['key']

print("Downloading Lending Club dataset via kagglehub...")
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print("Downloaded to:", path)

dest_dir = r"d:\DavidBroPrj\credit-risk-project\data\raw"
print(f"Copying files to {dest_dir}...")

# Copy files from cache to our raw data directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Data successfully placed in data/raw!")
