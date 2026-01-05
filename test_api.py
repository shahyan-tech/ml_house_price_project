# test_api.py

from fastapi.testclient import TestClient  # Make sure this import is correct
from src.api.main import app  # import your FastAPI app
import json

# Initialize the TestClient with the FastAPI app
client = TestClient(app)

# Load test input
with open("test_input.json") as f:
    data = json.load(f)

# Send POST request to /predict
response = client.post("/predict", json=data)

# Print the JSON response
print(response.json())
