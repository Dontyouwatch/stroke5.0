from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import requests
import os

app = Flask(__name__)

# Model URLs from GitHub
MODEL_URLS = {
    'xgb': 'https://github.com/Dontyouwatch/stroke5.0/raw/main/models/xgboost_model.pkl',
    'ensemble': 'https://github.com/Dontyouwatch/stroke5.0/raw/main/models/ensemble_model.pkl'
}

# Directory to cache downloaded models
MODEL_CACHE_DIR = 'model_cache'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def download_model(model_name):
    """Download model from GitHub if not already cached"""
    cache_path = os.path.join(MODEL_CACHE_DIR, f'{model_name}_model.pkl')
    
    if not os.path.exists(cache_path):
        print(f"Downloading {model_name} model from GitHub...")
        try:
            response = requests.get(MODEL_URLS[model_name])
            response.raise_for_status()
            with open(cache_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading {model_name} model: {str(e)}")
            return None
    
    try:
        return joblib.load(cache_path)
    except Exception as e:
        print(f"Error loading {model_name} model: {str(e)}")
        return None

# Load models at startup with sentence case feature alignment
try:
    xgb_model = download_model('xgb')
    ensemble_model = download_model('ensemble')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    xgb_model = None
    ensemble_model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not xgb_model or not ensemble_model:
        return jsonify({
            'status': 'error',
            'message': 'Models not loaded properly'
        }), 500

    try:
        # 1. Get and validate form data
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400

        # 2. Transform form data to match dataset's sentence case features
        features = {
            'Age': float(data['age']),
            'Sex': 1 if data['sex'] == 'male' else 0,
            'BMI': float(data['bmi']),
            'Cholesterol': float(data['cholesterol']),
            'Hypertension': int(data['hypertension']),
            'Atrial Fibrillation': int(data['atrial_fibrillation']),
            'Diabetes': int(data['diabetes']),
            'Smoking': int(data['smoking']),
            'Previous Stroke': int(data['previous_stroke'])
        }

        # 3. Create DataFrame with exact sentence case feature order
        feature_order = [
            'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension',
            'Atrial Fibrillation', 'Diabetes', 'Smoking', 'Previous Stroke'
        ]
        input_df = pd.DataFrame([features], columns=feature_order)

        # 4. Make predictions with both models
        xgb_prob = xgb_model.predict_proba(input_df)[0][1]
        ensemble_prob = ensemble_model.predict_proba(input_df)[0][1]

        return jsonify({
            'status': 'success',
            'xgb_risk': round(xgb_prob * 100, 1),
            'ensemble_risk': round(ensemble_prob * 100, 1),
            'features': {k.lower().replace(' ', '_'): v for k, v in features.items()}  # Return form-style keys
        })

    except KeyError as e:
        return jsonify({
            'status': 'error',
            'message': f'Missing form field: {str(e)}'
        }), 400
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid value: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
