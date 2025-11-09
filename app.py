from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import threading

# --- Initialize Flask App ---
# We tell Flask to look for the 'dashboard.html' file in the same folder ('.')
app = Flask(__name__, template_folder='.')
CORS(app)

# --- Global Variable for Sensor Data ---
data_lock = threading.Lock()
latest_sensor_data = {
    "n": 0, "p": 0, "k": 0,
    "temperature": 0, "humidity": 0,
    "ph": 0, "rainfall": 0,
    "soil_moisture": 0  # Added soil moisture
}

# --- Load Models and Datasets ---
try:
    model = joblib.load('crop_recommendation_model.pkl')
except FileNotFoundError:
    print("Error: Model file 'crop_recommendation_model.pkl' not found.")
    print("Please make sure it is in the same folder as app.py")
    model = None

try:
    production_df = pd.read_csv('crop_production_india.csv')
    fertilizer_df = pd.read_csv('fertilizer_recommendation.csv')
    
    production_df.dropna(subset=['Production'], inplace=True)
    production_df['Yield'] = production_df['Production'] / (production_df['Area'] + 1e-6)
    yield_data = production_df.groupby(['State_Name', 'Crop'])['Yield'].mean().reset_index()
    
    ALL_STATES = sorted(production_df['State_Name'].unique().tolist())
except FileNotFoundError as e:
    print(f"Error: Data file not found - {e.filename}")
    print("Please make sure all .csv files are in the same folder as app.py")
    yield_data, fertilizer_df, ALL_STATES = None, None, []

# --- Helper Functions ---

def recommend_fertilizer(crop, soil, moisture):
    """Smarter fertilizer logic using soil moisture."""
    if fertilizer_df is None: return "General Purpose (20-20)"
    
    crop_lower = crop.lower()
    soil_lower = soil.lower()
    
    # Filter by Crop and Soil
    filtered = fertilizer_df[
        (fertilizer_df['Crop Type'].str.lower() == crop_lower) &
        (fertilizer_df['Soil Type'].str.lower() == soil_lower)
    ]
    
    if not filtered.empty:
        # If we have matches, find the one closest to the current moisture
        # Calculate absolute difference in moisture
        filtered['moisture_diff'] = abs(filtered['Soil Moisture'] - moisture)
        # Get the row with the smallest difference
        best_match = filtered.loc[filtered['moisture_diff'].idxmin()]
        return best_match['Fertilizer Name']
    else:
        # Fallback if no specific match
        if moisture < 30: return "Urea (for dry soil)"
        if moisture > 60: return "DAP (for wet soil)"
        return "14-35-14 (General)"

def get_expected_yield(crop, state):
    if yield_data is None: return "Yield data not available."
    
    yield_result = yield_data[
        (yield_data['State_Name'] == state) & 
        (yield_data['Crop'].str.lower() == crop.lower())
    ]
    if not yield_result.empty:
        return f"{yield_result['Yield'].iloc[0]:.2f} tonnes/hectare"
    return "Yield data not available."

# --- API Endpoints ---

@app.route('/')
def dashboard():
    """
    This is the main route. When you go to http://127.0.0.1:5000/,
    this function runs and tries to send 'dashboard.html' to your browser.
    """
    return render_template('dashboard.html')

@app.route('/get_initial_data', methods=['GET'])
def get_initial_data():
    """This provides the states to the dropdown list."""
    if ALL_STATES:
        return jsonify({'states': ALL_STATES})
    else:
        return jsonify({'error': 'Could not load state data from server.'}), 500

@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    """Receives data FROM the ESP32."""
    global latest_sensor_data
    data = request.get_json()
    
    with data_lock:
        latest_sensor_data = {
            "n": data.get('n', 0),
            "p": data.get('p', 0),
            "k": data.get('k', 0),
            "temperature": data.get('temperature', 0),
            "humidity": data.get('humidity', 0),
            "ph": data.get('ph', 0),
            "rainfall": data.get('rainfall', 0),
            "soil_moisture": data.get('soil_moisture', 0) 
        }
    print(f"Received from Arduino: {latest_sensor_data}")
    return jsonify({"status": "success"})

@app.route('/api/get_recommendation', methods=['POST'])
def get_recommendation():
    """Provides data TO the dashboard."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    request_data = request.get_json()
    state = request_data.get('state')
    soil = request_data.get('soil')

    if not state or not soil:
        return jsonify({'error': 'State or Soil not provided'}), 400

    with data_lock:
        current_data = latest_sensor_data.copy()

    # --- 1. Crop Recommendation ---
    try:
        features = pd.DataFrame([{
            'N': current_data['n'], 'P': current_data['p'], 'K': current_data['k'],
            'temperature': current_data['temperature'], 'humidity': current_data['humidity'],
            'ph': current_data['ph'], 'rainfall': current_data['rainfall']
        }])
        crop_prediction = model.predict(features)[0]
    except Exception as e:
        return jsonify({'error': f'Error in crop prediction: {str(e)}'}), 500

    # --- 2. Fertilizer & Yield ---
    fertilizer_prediction = recommend_fertilizer(
        crop_prediction, 
        soil,
        current_data['soil_moisture'] 
    )
    yield_prediction = get_expected_yield(crop_prediction, state)

    # --- Return sensor data AND recommendations ---
    return jsonify({
        'sensor_data': current_data,
        'recommendation': {
            'crop': crop_prediction.capitalize(),
            'fertilizer': fertilizer_prediction,
            'yield': yield_prediction
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')