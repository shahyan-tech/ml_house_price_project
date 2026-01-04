# ML House Price Prediction Project

A complete end-to-end machine learning project for predicting house prices using scikit-learn, FastAPI, Streamlit, MongoDB, and Kubernetes.

## 🚀 Project Overview

This project demonstrates a production-ready ML pipeline:
- **Data Exploration & Cleaning**: Jupyter notebooks for EDA and preprocessing.
- **Model Training**: Scikit-learn pipeline with GradientBoostingRegressor.
- **API Deployment**: FastAPI for serving predictions with MongoDB persistence.
- **Frontend**: Streamlit UI for user interaction.
- **Containerization**: Docker images for all components.
- **Orchestration**: Kubernetes deployments with CI/CD automation.

## 🏗️ Architecture

```
User → Streamlit Frontend (8501) → FastAPI API (8000) → ML Model → MongoDB (27017)
```

- **Frontend**: Web UI for inputting house features and viewing predictions.
- **API**: RESTful service handling predictions and storing history.
- **Database**: MongoDB for persistent prediction logs.
- **CI/CD**: GitHub Actions for automated testing and deployment to Kubernetes.

## 📁 Project Structure

```
ml_house_price_project/
├── data/
│   ├── raw/          # Original dataset
│   └── clean/        # Processed data
├── models/           # Trained ML models
├── notebooks/        # Jupyter notebooks for EDA/experiments
├── src/
│   ├── ml_project/   # Training scripts
│   └── api/          # FastAPI application
├── frontend/         # Streamlit UI
├── k8s/             # Kubernetes manifests
├── .github/workflows/ # CI/CD pipelines
├── requirements.txt  # Python dependencies
├── Dockerfile        # API container
├── test_api.py       # API tests
└── README.md
```

## 🛠️ Quick Start (Local Development)

### Prerequisites
- Python 3.11
- Docker & Docker Compose
- Minikube (for K8s)
- kubectl

### 1. Clone & Setup
```bash
git clone https://github.com/shahyan-tech/ml_house_price_project.git
cd ml_house_price_project
conda create -n ml-env python=3.11
conda activate ml-env
pip install -r requirements.txt
```

### 2. Run Notebooks (EDA & Training)
```bash
jupyter notebook
# Open notebooks/01_eda.ipynb, 02_cleaning.ipynb, 03_model_experiments.ipynb
# Run cells to explore data and train model
```

### 3. Start API
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# Test: python test_api.py
# Docs: http://localhost:8000/docs
```

### 4. Run Frontend
```bash
docker build -t house-price-frontend:latest frontend/
docker run -p 8501:8501 house-price-frontend:latest
# Access: http://localhost:8501
```

### 5. Full Kubernetes Deployment
```bash
minikube start --driver=docker
minikube docker-env | Invoke-Expression  # Windows PowerShell

# Build images
docker build -t ml-house-price-api:latest .
docker build -t house-price-frontend:latest frontend/

# Deploy
kubectl apply -f k8s/
kubectl port-forward svc/frontend-service 8501:8501
kubectl port-forward svc/ml-api-service 8000:8000

# Access:
# Frontend: http://localhost:8501
# API: http://localhost:8000/docs
```

## 🔄 CI/CD Pipeline

Pushes to `main` trigger GitHub Actions:
- Build & test API/Frontend
- Deploy to kind cluster
- Run integration tests

Check: **GitHub → Actions** tab.

## 📊 API Endpoints

- `POST /predict`: Predict house price
  - Body: `{"features": {"LotArea": 8450, "OverallQual": 7, ...}}`
  - Response: `{"prediction": 208500.0}`
- `GET /history`: Get prediction history
- `GET /docs`: Interactive API docs

## 🗄️ Database

MongoDB stores predictions:
```json
{
  "input": {"LotArea": 8450, ...},
  "prediction": 208500.0,
  "timestamp": "2026-01-04T10:00:00Z"
}
```

## 🧪 Testing

```bash
# API tests
python test_api.py

# K8s tests
kubectl port-forward svc/ml-api-service 8000:8000
python test_api.py
```

## 🚀 Deployment Options

- **Local**: Minikube (above)
- **CI**: GitHub Actions with kind
- **Cloud**: Deploy to AKS/EKS with Helm

## 📈 Model Details

- **Algorithm**: GradientBoostingRegressor
- **Features**: LotArea, OverallQual, OverallCond, YearBuilt, GrLivArea
- **Preprocessing**: StandardScaler, OrdinalEncoder, OneHotEncoder
- **Metrics**: RMSE, MAE, R2 (logged in MLflow)

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. Add tests
4. Submit PR

## 📄 License

MIT License - see LICENSE file.

## 📞 Support

For issues: GitHub Issues
Demo: Run the K8s deployment and access the frontend!
