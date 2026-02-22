from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import threading
import requests

app = Flask(__name__, template_folder='.')
CORS(app)

data_lock = threading.Lock()
latest_sensor_data = {
    "n": 0, "p": 0, "k": 0,
    "temperature": 0, "humidity": 0,
    "ph": 0, "rainfall": 0,
    "soil_moisture": 0 
}

try:
    model = joblib.load('crop_recommendation_model.pkl')
except FileNotFoundError:
    model = None

try:
    production_df = pd.read_csv('crop_production_india.csv')
    fertilizer_df = pd.read_csv('fertilizer_recommendation.csv')
    production_df.dropna(subset=['Production'], inplace=True)
    production_df['Yield'] = production_df['Production'] / (production_df['Area'] + 1e-6)
    yield_data = production_df.groupby(['State_Name', 'Crop'])['Yield'].mean().reset_index()
    ALL_STATES = sorted(production_df['State_Name'].unique().tolist())
except FileNotFoundError:
    yield_data, fertilizer_df, ALL_STATES = None, None, []

def recommend_fertilizer(crop, soil, moisture):
    if fertilizer_df is None: return "General Purpose (20-20)"
    crop_lower, soil_lower = crop.lower(), soil.lower()
    filtered = fertilizer_df[(fertilizer_df['Crop Type'].str.lower() == crop_lower) & (fertilizer_df['Soil Type'].str.lower() == soil_lower)]
    if not filtered.empty:
        filtered['moisture_diff'] = abs(filtered['Soil Moisture'] - moisture)
        return filtered.loc[filtered['moisture_diff'].idxmin()]['Fertilizer Name']
    if moisture < 30: return "Urea (for dry soil)"
    if moisture > 60: return "DAP (for wet soil)"
    return "14-35-14 (General)"

def get_expected_yield(crop, state):
    if yield_data is None: return "Yield data not available."
    yield_result = yield_data[(yield_data['State_Name'] == state) & (yield_data['Crop'].str.lower() == crop.lower())]
    if not yield_result.empty: return f"{yield_result['Yield'].iloc[0]:.2f} tonnes/hectare"
    return "Yield data not available."

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/get_initial_data', methods=['GET'])
def get_initial_data():
    if ALL_STATES: return jsonify({'states': ALL_STATES})
    return jsonify({'error': 'Could not load state data.'}), 500

@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    global latest_sensor_data
    data = request.get_json()
    with data_lock:
        latest_sensor_data = {
            "n": data.get('n', 0), "p": data.get('p', 0), "k": data.get('k', 0),
            "temperature": data.get('temperature', 0), "humidity": data.get('humidity', 0),
            "ph": data.get('ph', 0), "rainfall": data.get('rainfall', 0), "soil_moisture": data.get('soil_moisture', 0) 
        }
    return jsonify({"status": "success"})

@app.route('/api/get_recommendation', methods=['POST'])
def get_recommendation():
    if model is None: return jsonify({'error': 'Model not loaded'}), 500
    request_data = request.get_json()
    state, soil = request_data.get('state'), request_data.get('soil')
    if not state or not soil: return jsonify({'error': 'State/Soil missing'}), 400

    with data_lock: current_data = latest_sensor_data.copy()

    try:
        features = pd.DataFrame([{'N': current_data['n'], 'P': current_data['p'], 'K': current_data['k'], 'temperature': current_data['temperature'], 'humidity': current_data['humidity'], 'ph': current_data['ph'], 'rainfall': current_data['rainfall']}])
        crop_prediction = model.predict(features)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    fertilizer_prediction = recommend_fertilizer(crop_prediction, soil, current_data['soil_moisture'])
    yield_prediction = get_expected_yield(crop_prediction, state)

    return jsonify({
        'sensor_data': current_data,
        'recommendation': {'crop': crop_prediction.capitalize(), 'fertilizer': fertilizer_prediction, 'yield': yield_prediction}
    })

# --- REAL ML ENDPOINT ---
@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    
    # The WORKING Hugging Face Model
    HF_API_URL = "https://api-inference.huggingface.co/models/dima806/plant_disease_detection"
    
    # REPLACE THE TEXT BELOW WITH YOUR ACTUAL TOKEN! Keep the word "Bearer " and the quotes.
    HF_HEADERS = {"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"} 
    
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, data=image_bytes)
        
        if response.status_code == 200:
            predictions = response.json()
            
            # Extract the best prediction safely
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], list):
                    best_prediction = predictions[0][0]
                else:
                    best_prediction = predictions[0]
            else:
                return jsonify({'error': 'Unexpected response from AI'}), 500

            # Format the disease name beautifully
            raw_label = best_prediction.get('label', 'Unknown')
            disease_name = raw_label.replace('_', ' ').title()
            confidence = round(best_prediction.get('score', 0.0) * 100, 1)
            
            # Generate dynamic cause/cure text
            if "Healthy" in disease_name or "Background" in disease_name:
                cause = "N/A"
                cure = "Looking great! Maintain optimal NPK and moisture levels."
            else:
                cause = "Fungal, Viral, or Bacterial Infection"
                cure = "Isolate the plant, remove infected leaves, and apply appropriate fungicide/bactericide."
                
            return jsonify({
                'disease': disease_name,
                'cause': cause,
                'cure': cure,
                'confidence': confidence
            })
        else:
             return jsonify({'error': 'ML API Failed', 'details': response.text}), response.status_code
             
    except Exception as e:
         return jsonify({'error': 'Request to ML API failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
