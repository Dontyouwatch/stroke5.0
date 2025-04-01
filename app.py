from flask import Flask, request, jsonify, send_from_directory
from xgboost import XGBClassifier
import pandas as pd
import joblib
from pathlib import Path
import os
import traceback
import numpy as np
from datetime import datetime

app = Flask(__name__, static_folder='static')

# Configuration
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'strokemodel.json'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

# Initialize model and scaler
model = None
scaler = None

# Define the EXACT feature order expected by the model
# This must match exactly what was used during training
FEATURE_ORDER = [
    'Age', 
    'Sex', 
    'BMI', 
    'Cholesterol', 
    'Hypertension', 
    'Atrial_Fibrillation', 
    'Diabetes', 
    'Smoking', 
    'Previous_Stroke',
    'Hypertension_Age',
    'Cholesterol_BMI',
    'Smoking_Cholesterol',
    'Age_Group_40-59',
    'Age_Group_60+',
    'BMI_Category_Overweight',
    'BMI_Category_Obese'
]

def load_models():
    """Load model and scaler with enhanced error handling"""
    global model, scaler
    
    try:
        # Verify model files exist
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
        # Load model
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        
        print(f"{datetime.now()} - Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"{datetime.now()} - ERROR loading models:")
        print(traceback.format_exc())
        return False

# Load models at startup
if not load_models():
    print(f"{datetime.now()} - CRITICAL: Models failed to load")

def prepare_features(input_data):
    """
    Prepare features in EXACTLY the same way as during training
    This must match your training pipeline exactly
    """
    # Create base features
    features = {
        'Age': input_data['age'],
        'Sex': 1 if input_data['sex'] == 'male' else 0,
        'BMI': input_data['bmi'],
        'Cholesterol': input_data['cholesterol'],
        'Hypertension': input_data['hypertension'],
        'Atrial_Fibrillation': input_data['atrial_fibrillation'],
        'Diabetes': input_data['diabetes'],
        'Smoking': input_data['smoking'],
        'Previous_Stroke': input_data['previous_stroke']
    }
    
    # Create interaction terms (must match training)
    features['Hypertension_Age'] = features['Hypertension'] * features['Age']
    features['Cholesterol_BMI'] = features['Cholesterol'] * features['BMI']
    features['Smoking_Cholesterol'] = features['Smoking'] * features['Cholesterol']
    
    # Create age groups (must match training)
    features['Age_Group_40-59'] = 1 if 40 <= features['Age'] < 60 else 0
    features['Age_Group_60+'] = 1 if features['Age'] >= 60 else 0
    
    # Create BMI categories (must match training)
    features['BMI_Category_Overweight'] = 1 if 25 <= features['BMI'] < 30 else 0
    features['BMI_Category_Obese'] = 1 if features['BMI'] >= 30 else 0
    
    # Create DataFrame with EXACT feature order
    input_df = pd.DataFrame([features])[FEATURE_ORDER]
    
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    start_time = datetime.now()
    try:
        # Check if models are loaded
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction service unavailable (models not loaded)',
                'timestamp': str(datetime.now())
            }), 503

        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data received',
                'timestamp': str(datetime.now())
            }), 400

        # Required fields with validation
        required_fields = {
            'age': (float, 'Age must be a number'),
            'sex': (str, 'Sex must be specified'),
            'bmi': (float, 'BMI must be a number'),
            'cholesterol': (float, 'Cholesterol must be a number'),
            'hypertension': (int, 'Hypertension must be 0, 1, or 2'),
            'atrial_fibrillation': (int, 'Atrial fibrillation must be 0 or 1'),
            'diabetes': (int, 'Diabetes must be 0 or 1'),
            'smoking': (int, 'Smoking must be 0 or 1'),
            'previous_stroke': (int, 'Previous stroke must be 0 or 1')
        }

        input_data = {}
        for field, (field_type, error_msg) in required_fields.items():
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}',
                    'timestamp': str(datetime.now())
                }), 400
            
            try:
                input_data[field] = field_type(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    'status': 'error',
                    'message': error_msg,
                    'timestamp': str(datetime.now())
                }), 400

        # Prepare features in exact expected order
        input_df = prepare_features(input_data)
        
        # Scale features
        numeric_features = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Age', 'Cholesterol_BMI', 'Smoking_Cholesterol']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make prediction
        stroke_prob = float(model.predict_proba(input_df)[0][1])
        
        return jsonify({
            'status': 'success',
            'risk_percentage': round(stroke_prob * 100, 1),
            'timestamp': str(datetime.now()),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        })
        
    except Exception as e:
        print(f"{datetime.now()} - Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': str(datetime.now())
        }), 500

@app.route('/healthcheck')
def healthcheck():
    return jsonify({
        'status': 'ok' if model and scaler else 'service_unavailable',
        'model_loaded': bool(model),
        'scaler_loaded': bool(scaler),
        'timestamp': str(datetime.now())
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
