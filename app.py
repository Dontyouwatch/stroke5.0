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

# Initialize model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler with enhanced error handling"""
    global model, scaler
    
    try:
        model_dir = Path(__file__).parent / 'models'
        model_path = model_dir / 'heartstroke.json'
        scaler_path = model_dir / 'scaler.pkl'
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model files not found")
        
        # Load XGBoost model from JSON
        model = XGBClassifier()
        model.load_model(model_path)
        
        # Load scaler from pickle
        scaler = joblib.load(scaler_path)
        
        print(f"{datetime.now()} - Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"{datetime.now()} - Error loading model or scaler: {str(e)}")
        print(traceback.format_exc())
        return False

# Load model at startup
if not load_model_and_scaler():
    print(f"{datetime.now()} - WARNING: Failed to load model - service will not function properly")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = datetime.now()
    try:
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction service not available',
                'timestamp': str(datetime.now())
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data received',
                'timestamp': str(datetime.now())
            }), 400

        # Validate and convert input data
        required_fields = ['age', 'sex', 'bmi', 'cholesterol', 'hypertension', 
                          'atrial_fibrillation', 'diabetes', 'smoking', 'previous_stroke']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}',
                    'timestamp': str(datetime.now())
                }), 400

        # Prepare input data
        input_data = {
            'Age': float(data['age']),
            'Sex': 1 if data['sex'] == 'male' else 0,
            'BMI': float(data['bmi']),
            'Cholesterol': float(data['cholesterol']),
            'Hypertension': int(data['hypertension']),
            'Atrial_Fibrillation': int(data['atrial_fibrillation']),
            'Diabetes': int(data['diabetes']),
            'Smoking': int(data['smoking']),
            'Previous_Stroke': int(data['previous_stroke'])
        }

        # Create DataFrame and apply feature engineering
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        input_df['Age_Group'] = pd.cut(input_df['Age'], 
                                     bins=[0, 40, 60, 80, np.inf], 
                                     labels=['Young', 'Middle_Aged', 'Senior', 'Elderly'])
        input_df['BMI_Category'] = pd.cut(input_df['BMI'], 
                                        bins=[0, 18.5, 24.9, 29.9, np.inf], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        input_df['Hypertension_Age'] = input_df['Hypertension'] * input_df['Age']
        input_df['Cholesterol_BMI'] = input_df['Cholesterol'] * input_df['BMI']
        input_df['Smoking_Cholesterol'] = input_df['Smoking'] * input_df['Cholesterol']

        # Convert categoricals to dummy variables
        input_df = pd.get_dummies(input_df, columns=['Age_Group', 'BMI_Category'], drop_first=True)

        # Ensure all expected columns exist
        expected_columns = [
            'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension', 
            'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke',
            'Hypertension_Age', 'Cholesterol_BMI', 'Smoking_Cholesterol',
            'Age_Group_Middle_Aged', 'Age_Group_Senior', 'Age_Group_Elderly',
            'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese'
        ]
        
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Scale features
        numeric_features = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Age', 'Cholesterol_BMI', 'Smoking_Cholesterol']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make prediction
        stroke_prob = float(model.predict_proba(input_df[expected_columns])[0][1])
        
        response = {
            'status': 'success',
            'risk_percentage': round(stroke_prob * 100, 1),
            'timestamp': str(datetime.now()),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        print(f"Successful prediction: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error_details': str(e),
            'timestamp': str(datetime.now()),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/healthcheck')
def healthcheck():
    return jsonify({
        'status': 'ok' if model and scaler else 'service_unavailable',
        'model_loaded': bool(model),
        'scaler_loaded': bool(scaler),
        'timestamp': str(datetime.now())
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
