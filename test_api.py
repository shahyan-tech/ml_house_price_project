from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
import json

# Mock MongoDB connection for tests
with patch("src.api.main.MongoClient"):
    client = TestClient(app)

with open("test_input.json") as f:
    data = json.load(f)

response = client.post("/predict", json=data)
print(response.json())