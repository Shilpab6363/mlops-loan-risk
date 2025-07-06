import pandas as pd
import yaml
import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load hyperparameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
alpha = params["train"]["alpha"]

# Load data
df = pd.read_csv("data/processed/cleaned.csv")

# Drop ID or unnecessary columns
if 'Loan_ID' in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Split data
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# Convert categorical columns
X = pd.get_dummies(X)

# Save feature column order for evaluation
os.makedirs("models", exist_ok=True)
with open("models/columns.json", "w") as f:
    import json
    json.dump(list(X.columns), f)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Log to MLflow
mlflow.start_run()
mlflow.log_param("alpha", alpha)
mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, "ridge-model")
mlflow.end_run()
