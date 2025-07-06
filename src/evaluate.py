import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score
import os

# Load data
df = pd.read_csv("data/processed/cleaned.csv")

# Drop ID or unnecessary columns
if 'Loan_ID' in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})


# Convert categorical columns
X = pd.get_dummies(X)

# Load expected columns
with open("models/columns.json", "r") as f:
    expected_cols = json.load(f)

# Align feature columns
X = X.reindex(columns=expected_cols, fill_value=0)

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Predict
y_pred = model.predict(X)

# Round predictions for classification
y_pred_class = [1 if p >= 0.5 else 0 for p in y_pred]

# Evaluate
acc = accuracy_score(y, y_pred_class)
print(f"Accuracy: {acc}")


