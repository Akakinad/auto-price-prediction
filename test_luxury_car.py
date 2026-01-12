
import requests
import json

car_data = {
    "year": 2018,
    "mileage": 30000,
    "volume": 3000,
    "make": 11,
    "fuel_type": 1,
    "transmission": 0
}

response = requests.post("http://localhost:5000/predict", json=car_data)
print("🚗 2018 BMW with 30k km:")
print(json.dumps(response.json(), indent=2))
