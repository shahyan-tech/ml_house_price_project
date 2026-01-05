import requests
import json

url = "http://127.0.0.1:8001/predict"

with open("test_input.json") as f:
    data = json.load(f)

response = requests.post(url, json=data)
print(response.json())
