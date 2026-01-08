from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model
model = joblib.load('models/xgboost_model.pkl')

@app.route('/')
def home():
    return jsonify({
        'message': 'Car Price Prediction API',
        'model': 'XGBoost',
        'accuracy': '93.2% (R² Score)',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame with proper column order
        input_df = pd.DataFrame({
            'make': [data.get('make', 50)],
            'model': [data.get('model', 300)],
            'year': [data['year']],
            'condition': [data.get('condition', 1)],
            'mileage(kilometers)': [data['mileage']],
            'fuel_type': [data.get('fuel_type', 1)],
            'volume(cm3)': [data['volume']],
            'color': [data.get('color', 5)],
            'transmission': [data.get('transmission', 0)],
            'drive_unit': [data.get('drive_unit', 1)],
            'segment': [data.get('segment', 2)]
        })
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return result
        return jsonify({
            'predicted_price': float(round(prediction, 2)),
            'currency': 'USD',
            'model': 'XGBoost',
            'confidence': 'R² = 0.9323'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Car Price Prediction API...")
    print("📊 Model: XGBoost (93.2% accuracy)")
    print("🌐 API running at: http://localhost:5000")
    app.run(debug=True, port=5000)