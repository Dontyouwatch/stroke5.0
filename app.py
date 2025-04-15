from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model_path = 'stroke_model.pkl'  # Update with your actual model path
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Serve the HTML file
    with open('index.html', 'r') as f:
        return f.read()

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
        
        # Make prediction
        prediction = model.predict_proba(df)[0][1]  # Probability of stroke
        
        # Convert to percentage (0-100%)
        risk_percentage = round(prediction * 100, 2)
        
        # Return result
        return jsonify({
            'risk_percentage': risk_percentage,
            'status': 'success'
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
    app.run(debug=True)
