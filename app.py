from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# --- Load preprocessor and XGBoost model for price prediction ---
try:
    preprocessor = joblib.load('preprocessor.pkl')
except Exception as e:
    preprocessor = None
    preprocessor_error = str(e)
try:
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model('Xgboost_price_predictor.json')
except Exception as e:
    xgb_reg = None
    xgb_reg_error = str(e)

# --- Load recommender pipeline and reference data ---
try:
    recommender_model = joblib.load('Recommender_knn_model.pkl')
except Exception as e:
    recommender_model = None
    recommender_model_error = str(e)
try:
    df_ref = pd.read_csv('Large_Mobile_Datset_with_Name.csv')
except Exception as e:
    df_ref = None
    df_ref_error = str(e)

REQUIRED_FEATURES = [
    'Processor', 'Brand', 'Ram_GB', 'Rom_GB',
    'Battery_Capacity', 'Display_Quality', 'Version'
]

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict_price():
    if xgb_reg is None or preprocessor is None:
        return jsonify({'error': f'Model or preprocessor not loaded: {xgb_reg_error if xgb_reg is None else preprocessor_error}'}), 500
    data = request.json
    missing = [feat for feat in REQUIRED_FEATURES if data.get(feat) is None]
    if missing:
        return jsonify({'error': f"Missing required fields: {', '.join(missing)}. Please select all required fields."}), 400
    input_data = {feat: data[feat] for feat in REQUIRED_FEATURES}
    input_df = pd.DataFrame([input_data])
    try:
        X_transformed = preprocessor.transform(input_df)
        price = xgb_reg.predict(X_transformed)[0]
        return jsonify({'predicted_price': float(price)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/recommend', methods=['POST'])
def recommend_phones():
    if recommender_model is None or preprocessor is None or df_ref is None:
        return jsonify({'error': f'Model, preprocessor, or reference data not loaded: {recommender_model_error if recommender_model is None else preprocessor_error if preprocessor is None else df_ref_error}'}), 500
    data = request.json
    missing = [feat for feat in REQUIRED_FEATURES if data.get(feat) is None]
    if missing:
        return jsonify({'error': f"Missing required fields: {', '.join(missing)}. Please select all required fields."}), 400
    input_data = {feat: data[feat] for feat in REQUIRED_FEATURES}
    input_df = pd.DataFrame([input_data])
    try:
        user_encoded = preprocessor.transform(input_df)
        distances, indices = recommender_model.kneighbors(user_encoded)
        similar_phones = df_ref.iloc[indices[0]].copy()
        return jsonify({'similar_phones': similar_phones.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 