from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder='static')

# Configure model paths
MODEL_PATHS = {
    'xgboost_json': 'models/strokemodel.json',
    'xgboost_pkl': 'models/xgboost_model.pkl',
    'ensemble': 'models/ensemble_model.pkl',
    'scaler': 'models/scaler.pkl'
}

# Load models
try:
    # Try loading XGBoost from JSON first
    model = xgb.Booster()
    model.load_model(MODEL_PATHS['xgboost_json'])
    print("Loaded XGBoost model from JSON")
    model_type = 'xgboost_json'
except:
    try:
        # Fall back to pickle version
        model = joblib.load(MODEL_PATHS['xgboost_pkl'])
        print("Loaded XGBoost model from pickle")
        model_type = 'xgboost_pkl'
    except:
        try:
            # Final fallback to ensemble
            model = joblib.load(MODEL_PATHS['ensemble'])
            print("Loaded ensemble model")
            model_type = 'ensemble'
        except Exception as e:
            raise Exception(f"Could not load any model: {str(e)}")

# Load scaler if exists
try:
    scaler = joblib.load(MODEL_PATHS['scaler'])
    print("Loaded scaler")
except:
    scaler = None
    print("No scaler found")

@app.route('/')
def home():
    # Serve the HTML file from static folder
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Map the form data to match your model's expected input
        input_data = {
            'Age': _map_age(data['age']),
            'Sex': 1 if data['sex'] == 'male' else 0,
            'BMI': _map_bmi(data['bmi']),
            'Cholesterol': _map_cholesterol(data['cholesterol']),
            'Hypertension': data['hypertension'],
            'Atrial_Fibrillation': data['atrial_fibrillation'],
            'Diabetes': data['diabetes'],
            'Smoking': data['smoking'],
            'Previous_Stroke': data['previous_stroke']
        }
        
        # Convert to DataFrame with correct column order
        df = pd.DataFrame([input_data], columns=[
            'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension',
            'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke'
        ])
        
        # Scale features if scaler exists
        if scaler is not None:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
        # Convert to DMatrix for XGBoost JSON model
        if model_type == 'xgboost_json':
            dmatrix = xgb.DMatrix(df)
            prediction = model.predict(dmatrix)[0]
        else:
            # For sklearn-style models
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(df)[0][1]  # Probability of stroke (class 1)
            else:
                prediction = model.predict(df)[0]
        
        # Convert to percentage (0-100%) with clipping
        risk_percentage = min(100, max(0, round(float(prediction) * 100, 2)))
        
        # Return result
        return jsonify({
            'risk_percentage': risk_percentage,
            'status': 'success',
            'model_used': model_type,
            'features_used': input_data
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# Helper functions to map form values to model expected values
def _map_age(age_value):
    age = float(age_value)
    if age < 40:
        return 1  # Young
    elif 40 <= age < 60:
        return 2  # Middle-aged
    else:
        return 3  # Elderly

def _map_bmi(bmi_value):
    bmi = float(bmi_value)
    if bmi < 18.5:
        return 1  # Underweight
    elif 18.5 <= bmi < 25:
        return 2  # Normal
    elif 25 <= bmi < 30:
        return 3  # Overweight
    else:
        return 4  # Obese

def _map_cholesterol(chol_value):
    chol = float(chol_value)
    if chol < 200:
        return 0  # Normal
    elif 200 <= chol < 240:
        return 1  # Borderline high
    else:
        return 2  # High

if __name__ == '__main__':
    # Check if models exist
    if not any(os.path.exists(path) for path in MODEL_PATHS.values()):
        raise FileNotFoundError("No trained model found. Please run the training script first.")
    
    app.run(debug=True)
