import os
import pickle
from typing import List

# Absolute path relative to project root
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gradient_boosting_pipeline.pkl'))

# Load model once when API starts
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def predict(data: List[dict]) -> List[float]:
    """
    data: list of dicts, each dict is one row of features
    """
    import pandas as pd
    df = pd.DataFrame(data)
    preds = model.predict(df)
    return preds.tolist()
