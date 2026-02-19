from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import threading
import time

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

# --- MOCK ML DEMO ENDPOINT ---
@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename.lower()
    
    # Simulate AI thinking time
    time.sleep(1.5)
    
    # 12 Mock Diseases Database
    diseases_db = {
        'spot': {'name': 'Bacterial Leaf Spot', 'cause': 'Bacteria (Xanthomonas)', 'cure': 'Apply copper-based bactericide and avoid overhead watering.'},
        'blight': {'name': 'Early Blight', 'cause': 'Fungus (Alternaria solani)', 'cure': 'Prune lower leaves and apply chlorothalonil fungicide.'},
        'rust': {'name': 'Common Rust', 'cause': 'Fungus (Puccinia sorghi)', 'cure': 'Apply mancozeb-based fungicide early in the season.'},
        'mildew': {'name': 'Powdery/Downy Mildew', 'cause': 'Fungal Infection', 'cure': 'Spray with neem oil or sulfur-based fungicide.'},
        'anthracnose': {'name': 'Anthracnose', 'cause': 'Colletotrichum Fungus', 'cure': 'Remove infected parts and spray copper fungicides.'},
        'mosaic': {'name': 'Mosaic Virus', 'cause': 'Virus (Aphid Transmitted)', 'cure': 'No cure. Uproot infected plants and use insecticidal soap.'},
        'wilt': {'name': 'Fusarium Wilt', 'cause': 'Soil-borne Fungus', 'cure': 'Solarize soil, ensure crop rotation, use resistant seeds.'},
        'canker': {'name': 'Citrus Canker', 'cause': 'Bacteria', 'cure': 'Prune infected branches and spray liquid copper fungicide.'},
        'scab': {'name': 'Plant Scab', 'cause': 'Fungus (Venturia)', 'cure': 'Apply captan or sulfur-based fungicides during leafing.'},
        'rot': {'name': 'Root Rot', 'cause': 'Water mold (Phytophthora)', 'cure': 'Improve soil drainage and reduce watering immediately.'},
        'curl': {'name': 'Leaf Curl', 'cause': 'Pest damage / Virus', 'cure': 'Control whitefly population and remove severely curled leaves.'},
        'healthy': {'name': 'Healthy Plant', 'cause': 'N/A', 'cure': 'Looking great! Maintain optimal NPK and moisture levels.'}
    }

    # Default to Healthy if no keyword is found in the filename
    result = diseases_db['healthy']
    
    # Scan the filename for keywords
    for key, data in diseases_db.items():
        if key in filename:
            result = data
            break
            
    return jsonify({
        'disease': result['name'],
        'cause': result['cause'],
        'cure': result['cure'],
        'confidence': 98.7  # Fake high confidence for the presentation!
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
