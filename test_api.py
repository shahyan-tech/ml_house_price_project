from fastapi.testclient import TestClient
from src.api.main import app
import json

client = TestClient(app)

def test_predict():
    with open("test_input.json") as f:
        payload = json.load(f)

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    print(response.json())

if __name__ == "__main__":
    test_predict()
