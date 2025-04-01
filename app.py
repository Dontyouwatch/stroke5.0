from flask import Flask, request, jsonify, send_from_directory
from xgboost import XGBClassifier
import pandas as pd
import joblib
from pathlib import Path
import os
import traceback
import numpy as np

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
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error',
                'message': 'Prediction service unavailable'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data received',
                'status': 'error'
            }), 400

        # Convert form data to model input format
        input_data = {
            'Age': float(data.get('age')),
            'Sex': 1 if data.get('sex') == 'male' else 0,
            'BMI': float(data.get('bmi')),
            'Cholesterol': float(data.get('cholesterol')),
            'Hypertension': int(data.get('hypertension')),
            'Atrial_Fibrillation': int(data.get('atrial_fibrillation')),
            'Diabetes': int(data.get('diabetes')),
            'Smoking': int(data.get('smoking')),
            'Previous_Stroke': int(data.get('previous_stroke'))
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply feature engineering (same as in your Python model)
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

        # Ensure we have all expected columns (fill missing with 0)
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

        # Scale the features
        numeric_features = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Age', 'Cholesterol_BMI', 'Smoking_Cholesterol']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make prediction
        stroke_prob = float(model.predict_proba(input_df[expected_columns])[0][1])
        
        return jsonify({
            'risk_percentage': round(stroke_prob * 100, 1),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
