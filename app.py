from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Model paths
MODEL_PATHS = {
    'ensemble': 'ensemble_model.pkl',
    'xgboost': 'xgboost_model.pkl'
}

# Load the ensemble model first, fall back to XGBoost if needed
try:
    model = joblib.load(MODEL_PATHS['ensemble'])
    print("Loaded ensemble model")
    model_type = 'ensemble'
except Exception as e:
    print(f"Failed to load ensemble model: {str(e)}")
    try:
        model = joblib.load(MODEL_PATHS['xgboost'])
        print("Loaded XGBoost model")
        model_type = 'xgboost'
    except Exception as e:
        raise Exception(f"Failed to load any model: {str(e)}")

@app.route('/')
def home():
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input data
        data = request.get_json()
        if not data:
            raise ValueError("No input data provided")
        
        # Prepare features in EXACTLY the same order as training (9 features)
        required_features = [
            'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension',
            'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke'
        ]
        
        # Map input data to model features
        input_data = {
            'Age': _map_age(data['age']),
            'Sex': 1 if data['sex'] == 'male' else 0,
            'BMI': _map_bmi(data['bmi']),
            'Cholesterol': _map_cholesterol(data['cholesterol']),
            'Hypertension': int(data['hypertension']),
            'Atrial_Fibrillation': int(data['atrial_fibrillation']),
            'Diabetes': int(data['diabetes']),
            'Smoking': int(data['smoking']),
            'Previous_Stroke': int(data['previous_stroke'])
        }
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([input_data], columns=required_features)
        
        # Debug: Verify features before prediction
        print("Features being sent to model:")
        print(df.columns.tolist())
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(df)[0][1]  # Probability of class 1 (stroke)
        else:
            prediction = model.predict(df)[0]  # Class prediction
        
        # Convert to percentage (0-100%)
        risk_percentage = min(100, max(0, round(float(prediction) * 100, 2)))
        
        return jsonify({
            'risk_percentage': risk_percentage,
            'status': 'success',
            'model_used': model_type
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Ensure you are sending exactly 9 features (excluding Stroke)'
        }), 400

# Mapping functions (must match training preprocessing)
def _map_age(age):
    age = float(age)
    if age < 40: return 1
    elif age < 60: return 2
    else: return 3

def _map_bmi(bmi):
    bmi = float(bmi)
    if bmi < 18.5: return 1
    elif bmi < 25: return 2
    elif bmi < 30: return 3
    else: return 4

def _map_cholesterol(chol):
    chol = float(chol)
    if chol < 200: return 0
    elif chol < 240: return 1
    else: return 2

if __name__ == '__main__':
    # Verify at least one model exists
    if not any(os.path.exists(p) for p in MODEL_PATHS.values()):
        raise FileNotFoundError("No model files found. Please ensure either ensemble_model.pkl or xgboost_model.pkl exists")
    
    app.run(host='0.0.0.0', port=5000)
