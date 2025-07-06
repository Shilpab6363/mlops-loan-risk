import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv("data/loan_data.csv")

# Simple cleanup
df = df.dropna()

# Save cleaned data
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned.csv", index=False)
