from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
MODELS_DIR = Path('models')
MODEL_PATHS = {
    'ensemble': MODELS_DIR / 'ensemble_model.pkl',
    'xgboost': MODELS_DIR / 'xgboost_model.pkl'
}

# Load model with fallback
model = None
model_type = None

for name, path in MODEL_PATHS.items():
    try:
        if path.exists():
            model = joblib.load(path)
            model_type = name
            print(f"Successfully loaded {name} model from {path}")
            break
    except Exception as e:
        print(f"Error loading {name} model: {str(e)}")

if model is None:
    raise Exception("Failed to load any model. Please ensure at least one model exists in the models directory.")

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No input data provided")
        
        # Prepare features in correct order
        features = [
            'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension',
            'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke'
        ]
        
        # Validate all required fields are present
        if not all(field in data for field in ['age', 'sex', 'bmi', 'cholesterol', 
                                             'hypertension', 'atrial_fibrillation',
                                             'diabetes', 'smoking', 'previous_stroke']):
            raise ValueError("Missing required fields in input data")
        
        # Map input data
        input_data = {
            'Age': _map_age(data['age']),
            'Sex': 1 if data['sex'].lower() == 'male' else 0,
            'BMI': _map_bmi(data['bmi']),
            'Cholesterol': _map_cholesterol(data['cholesterol']),
            'Hypertension': int(data['hypertension']),
            'Atrial_Fibrillation': int(data['atrial_fibrillation']),
            'Diabetes': int(data['diabetes']),
            'Smoking': int(data['smoking']),
            'Previous_Stroke': int(data['previous_stroke'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data], columns=features)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            risk = model.predict_proba(df)[0][1]  # Probability of stroke
        else:
            risk = model.predict(df)[0]
        
        risk_percentage = min(100, max(0, round(float(risk) * 100, 2)))
        
        return jsonify({
            'risk_percentage': risk_percentage,
            'status': 'success',
            'model_used': model_type
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Please check your input data and try again'
        }), 400

# Mapping functions
def _map_age(age):
    try:
        age = float(age)
        if age < 40: return 1
        elif age < 60: return 2
        return 3
    except:
        return 2  # Default to middle-aged if invalid

def _map_bmi(bmi):
    try:
        bmi = float(bmi)
        if bmi < 18.5: return 1
        elif bmi < 25: return 2
        elif bmi < 30: return 3
        return 4
    except:
        return 2  # Default to normal if invalid

def _map_cholesterol(chol):
    try:
        chol = float(chol)
        if chol < 200: return 0
        elif chol < 240: return 1
        return 2
    except:
        return 0  # Default to normal if invalid

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Check for models
    if not any(path.exists() for path in MODEL_PATHS.values()):
        print("Warning: No model files found in models directory")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
