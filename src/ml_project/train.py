"""
Production training script.
Trains the final GradientBoosting model,
logs to MLflow, and saves the pipeline artifact
in a Docker-compatible way.

Run:
    python src/ml_project/train.py
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn

# Optional version logging
print("Environment versions:", {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
    "joblib": joblib.__version__,
})

# Paths (robust to execution location)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "train_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "gradient_boosting_pipeline.pkl"

TARGET = "SalePrice"
RANDOM_STATE = 42

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Feature groups
num_cols = X.select_dtypes(exclude="object").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()
ordinal_cols = ["ExterQual", "KitchenQual", "HeatingQC", "BsmtQual", "GarageQual"]

# Ordinal categories
ord_categories = [["None", "Po", "Fa", "TA", "Gd", "Ex"]] * len(ordinal_cols)

# ----------------------
# Pipelines
# ----------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ord_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OrdinalEncoder(categories=ord_categories, dtype=float))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("ord", ord_pipeline, ordinal_cols),
    ("cat", cat_pipeline, [c for c in cat_cols if c not in ordinal_cols])
], remainder="drop")

# ----------------------
# Model
# ----------------------
model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# ----------------------
# MLflow
# ----------------------
mlflow.set_experiment("house_price_production")
os.makedirs(MODEL_DIR, exist_ok=True)

with mlflow.start_run(run_name="GradientBoosting_final"):
    start_time = time.time()

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    elapsed = time.time() - start_time

    # Log params
    mlflow.log_param("model", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("training_time_sec", elapsed)

    # Log model to MLflow
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    # Save pipeline locally (Docker-compatible)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(pipeline, model_path, compress=3)  # compress optional
    print(f"Model saved to: {model_path}")

print(f"Training complete. RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
