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
    """Load the trained model (JSON) and scaler (pkl)"""
    global model, scaler
    
    try:
        model_dir = Path(__file__).parent / 'models'
        model_path = model_dir / 'strokemodel.json'
        scaler_path = model_dir / 'scaler.pkl'
        
        # Load XGBoost model from JSON
        model = XGBClassifier()
        model.load_model(model_path)
        
        # Load scaler from pickle
        scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        print(traceback.format_exc())
        return False

# Load model at startup
if not load_model_and_scaler():
    print("Failed to load model - check error messages above")


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

        # Prepare features
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

        # Create DataFrame
        input_df = pd.DataFrame([features])

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

        # Convert categoricals
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
