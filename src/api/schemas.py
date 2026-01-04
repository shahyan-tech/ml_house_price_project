# src/api/schemas.py
#Define Pydantic schemas for request/response validation:
from pydantic import BaseModel
from typing import List

class HouseData(BaseModel):
    # Add your feature names here, example:
    LotArea: float
    OverallQual: int
    YearBuilt: int
    # ... add all features your pipeline expects

class PredictionResponse(BaseModel):
    predictions: List[float]
