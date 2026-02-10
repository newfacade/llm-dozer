import pandas as pd
import os

input_path = "data/wikitext-103-raw-v1-train.parquet"
output_path = "data/wikitext-103-raw-v1-train-sampled.parquet"

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    exit(1)

print(f"Loading {input_path}...")
df = pd.read_parquet(input_path)
print(f"Original rows: {len(df)}")

# Randomly sample 1/10 of the data
sampled_df = df.sample(frac=0.1, random_state=42)
print(f"Sampled rows: {len(sampled_df)}")

print(f"Saving to {output_path}...")
sampled_df.to_parquet(output_path)
print("Done.")
