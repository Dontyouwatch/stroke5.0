from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load both models
ensemble_model = joblib.load('ensemble_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Map form data to model features
        features = {
            'age': float(data['age']),
            'sex': 1 if data['sex'] == 'male' else 0,
            'bmi': float(data['bmi']),
            'cholesterol': float(data['cholesterol']),
            'hypertension': int(data['hypertension']),
            'atrial_fibrillation': int(data['atrial_fibrillation']),
            'diabetes': int(data['diabetes']),
            'smoking': int(data['smoking']),
            'previous_stroke': int(data['previous_stroke'])
        }
        
        # Create DataFrame with the same feature order as training data
        feature_order = [
            'age', 'sex', 'bmi', 'cholesterol', 'hypertension', 
            'atrial_fibrillation', 'diabetes', 'smoking', 'previous_stroke'
        ]
        input_df = pd.DataFrame([features], columns=feature_order)
        
        # Make predictions with both models
        try:
            # Ensemble model prediction
            ensemble_prob = ensemble_model.predict_proba(input_df)[0][1]
            ensemble_risk = round(ensemble_prob * 100, 1)
            
            # XGBoost model prediction (from the pipeline)
            xgb_prob = xgb_model.predict_proba(input_df)[0][1]
            xgb_risk = round(xgb_prob * 100, 1)
            
            # Return both predictions
            return jsonify({
                'status': 'success',
                'ensemble_risk': ensemble_risk,
                'xgb_risk': xgb_risk,
                'features': features
            })
            
        except Exception as model_error:
            return jsonify({
                'status': 'error',
                'message': f'Model prediction failed: {str(model_error)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid request data: {str(e)}'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
