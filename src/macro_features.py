import pandas as pd
import os
import time

def fetch_macro_features():
    """
    Fetches macro-economic data (e.g., from FRED) to append to our credit risk features.
    This acts as a mock implementation since the API key might not be present.
    """
    processed_data_path = "data/processed/features.parquet"
    if not os.path.exists(processed_data_path):
        print(f"Error: Could not find features at {processed_data_path}")
        return
        
    print("Loading existing features to append macro data...")
    df = pd.read_parquet(processed_data_path)
    
    print("Fetching macro data (Mocking for now)...")
    time.sleep(1)
    
    # We create some mock macro features based on the issue_year and issue_month
    # In reality, this would use pandas-datareader or fredapi
    
    # Mock Unemployment Rate: let's say it fluctuates between 4% and 10% based on year
    df['macro_unemployment_rate'] = 5.0 + (df['issue_year'] - 2007) * 0.2
    
    # Mock Federal Funds Rate
    df['macro_fed_funds_rate'] = 2.0 + (df['issue_month'] % 4) * 0.5
    
    print("Appending macro features to the dataset...")
    # Overwrite the features parquet
    df.to_parquet(processed_data_path, index=False)
    
    print(f"Appended macro features to {processed_data_path}")
    print("Done!")

if __name__ == "__main__":
    fetch_macro_features()
