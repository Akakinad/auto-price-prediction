import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Example car data
car_data = {
    "year": 2015,
    "mileage": 80000,
    "volume": 1600,
    "make": 82,      # VW
    "fuel_type": 1,  # Petrol
    "transmission": 0 # Auto
}

# Make request
response = requests.post(url, json=car_data)

# Print result
print("🚗 Testing Car Price Prediction API")
print("="*50)
print(f"Input: 2015 VW, 80k km, 1600cc")
print(f"\nResponse:")
print(json.dumps(response.json(), indent=2))