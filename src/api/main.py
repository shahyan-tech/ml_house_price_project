from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os


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

# MONGODB CONNECTION
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["house_price_db"]
    predictions_collection = db["predictions"]
    print("MongoDB connected successfully!")
except Exception as e:
    mongo_client = None
    predictions_collection = None
    print("MongoDB connection failed:", e)

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
        prediction_value = float(pred[0])

        # Save to MongoDB if connected
        if predictions_collection:
            predictions_collection.insert_one({
                "input": user_input,
                "prediction": prediction_value,
                "timestamp": datetime.utcnow()
            })

        return {"prediction": prediction_value}
    except Exception as e:
        return {"error": str(e)}


# ROOT ENDPOINT
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running ✅"}

# HISTORY ENDPOINT
@app.get("/history")
def get_history():
    if not predictions_collection:
        return {"error": "MongoDB not connected"}

    try:
        records = list(predictions_collection.find({}, {"_id": 0}))
        return {"history": records}
    except Exception as e:
        return {"error": str(e)}
