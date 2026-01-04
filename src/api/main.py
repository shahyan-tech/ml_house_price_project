from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_pipeline.pkl"


# LOAD PIPELINE
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    raise e

# FASTAPI APP
app = FastAPI(title="House Price Prediction API")


# INPUT MODEL
class InputFeatures(BaseModel):
    # Users can provide any subset of columns
    features: Dict[str, Any]


# PREDICTION ENDPOINT
@app.post("/predict")
def predict_price(features: InputFeatures):
    user_input = features.features

    # Extract the original columns the pipeline was trained on
    # ColumnTransformer stores the names before encoding in 'feature_names_in_'
    original_columns = model.named_steps["preprocessor"].feature_names_in_

    # Create DataFrame and fill missing values:
    df = pd.DataFrame([{
        col: user_input.get(col, np.nan)  # keep NaN so pipeline imputers handle them
        for col in original_columns
    }])

    try:
        pred = model.predict(df)
        return {"prediction": float(pred[0])}
    except Exception as e:
        return {"error": str(e)}


# ROOT ENDPOINT
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running ✅"}
