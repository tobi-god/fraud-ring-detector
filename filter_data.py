import os
import pandas as pd

# Define relative paths
input_path = "data/01_raw/Fraud Detection Dataset.csv"
output_path = "data/01_raw/synthetic_sample.csv"

print(f"Checking if raw file exists at: {input_path}")

if os.path.exists(input_path):
    print("File found! Loading first 50,000 rows...")
    df = pd.read_csv(input_path, nrows=50000)
    
    # Save the sliced sample
    df.to_csv(output_path, index=False)
    print(f"Success! Sliced dataset created with shape: {df.shape}")
    print(f"Saved to: {output_path}")
else:
    print(f"ERROR: Could not find '{input_path}'. Check the exact filename in data/01_raw/")