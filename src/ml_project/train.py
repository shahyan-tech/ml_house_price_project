# src/ml_project/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load cleaned data
df = pd.read_csv("data/clean/train_clean.csv")

# Separate features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Optional: define ordinal features (quality ratings)
ordinal_cols = ["ExterQual", "KitchenQual"]
ordinal_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

# Create pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Full pipeline: preprocessing + model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=500, random_state=42))
])

# Train on full dataset
pipeline.fit(X, y)

# Save the trained pipeline
joblib.dump(pipeline, "models/gradientboosting_pipeline.pkl")
print("Model saved as models/gradientboosting_pipeline.pkl")
