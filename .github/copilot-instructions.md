# AI Coding Assistant Instructions

## Project Overview
This is a machine learning project for house price prediction using scikit-learn pipelines, FastAPI for serving, and MLflow for experiment tracking. The project follows a structured ML workflow: data cleaning → model training → API deployment.

## Architecture
- **Data Flow**: Raw data (`data/raw/train.csv`) → Cleaned data (`data/clean/train_clean.csv`) → Trained model (`models/gradientboosting_pipeline.pkl`) → FastAPI predictions
- **Training**: `src/ml_project/train.py` creates a sklearn Pipeline with ColumnTransformer preprocessing + GradientBoostingRegressor
- **API**: `src/api/main.py` serves predictions via FastAPI at `/predict` endpoint
- **Experiments**: MLflow tracks runs in "house_price_production" experiment

## Key Conventions

### Path Handling
Always use robust path resolution for Docker compatibility:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "train_clean.csv"
```

### Model Pipeline Structure
The production model is a sklearn Pipeline with:
- **Preprocessor**: ColumnTransformer with 3 transformers:
  - `num`: Numerical features (median impute + StandardScaler)
  - `ord`: Ordinal features (constant impute "None" + OrdinalEncoder)
  - `cat`: Categorical features (constant impute "None" + OneHotEncoder)
- **Model**: GradientBoostingRegressor(n_estimators=100)

Ordinal features: `["ExterQual", "KitchenQual", "HeatingQC", "BsmtQual", "GarageQual"]`
Ordinal categories: `[["None", "Po", "Fa", "TA", "Gd", "Ex"]]` for each

### API Input/Output
- **Input**: `{"features": {"LotArea": 8450, "OverallQual": 7, ...}}` (dict of feature values)
- **Output**: `{"prediction": 208500.0}` or `{"error": "message"}`
- API handles missing features by setting them to NaN (let pipeline imputers handle)

### Version Pinning
For Docker reproducibility, pin exact versions in `requirements.txt` and validate in training scripts:
```python
REQUIRED_VERSIONS = {"numpy": "1.26.4", "pandas": "2.1.1", "sklearn": "1.7.2"}
for pkg, ver in REQUIRED_VERSIONS.items():
    actual = getattr(__import__(pkg), "__version__")
    if actual != ver:
        raise ValueError(f"{pkg} version mismatch")
```

## Development Workflows

### Training the Model
```bash
cd /path/to/project
python src/ml_project/train.py
```
- Creates `models/gradientboosting_pipeline.pkl`
- Logs metrics (RMSE, MAE, R2) and model to MLflow
- Validates environment versions

### Testing the API
```bash
# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# In another terminal
python test_api.py
```
Uses `test_input.json` with complete feature set for testing.

### Docker Deployment
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```
- Model is trained during Docker build for version compatibility

## Common Patterns

### Feature Engineering
- Numerical: median imputation + standardization
- Ordinal: map quality ratings (None/Po/Fa/TA/Gd/Ex) to 0-5 scale
- Categorical: one-hot encoding with unknown handling

### Error Handling
- API catches exceptions and returns `{"error": str(e)}`
- Training validates versions and paths early

### Model Persistence
```python
import joblib
joblib.dump(pipeline, model_path, compress=3)
model = joblib.load(model_path)
```

## File Structure Reference
- `src/ml_project/train.py`: Production training script
- `src/api/main.py`: FastAPI application
- `notebooks/`: Exploratory analysis and experiments (01_eda.ipynb, 02_cleaning.ipynb, 03_model_experiments.ipynb)
- `data/clean/train_clean.csv`: Preprocessed training data
- `models/`: Saved model artifacts
- `mlruns/`: MLflow experiment tracking
- `reports/experiment_results.csv`: Model comparison results</content>
<parameter name="filePath">d:\Ml_Projects\ml_house_price_project\.github\copilot-instructions.md