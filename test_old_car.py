
import requests
import json

car_data = {
    "year": 2005,
    "mileage": 200000,
    "volume": 1400,
    "make": 50,
    "fuel_type": 1,
    "transmission": 1
}

response = requests.post("http://localhost:5000/predict", json=car_data)
print("🚗 2005 old car with 200k km:")
print(json.dumps(response.json(), indent=2))
