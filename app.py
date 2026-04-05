from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import threading
import time
import os
import json
import google.generativeai as genai
from PIL import Image
import io

# --- Configure Gemini API ---
# Just hardcode it directly as a string for your review tomorrow to be safe!
GEMINI_API_KEY = "AIzaSyDCoLHsWfYy61z4BhuT4qRLkE50-503gQQ"
genai.configure(api_key=GEMINI_API_KEY)

# Use the 1.5 Flash model for fast multimodal tasks
vision_model = genai.GenerativeModel('gemini-2.5-flash')

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

# New Endpoint: Just get sensors for the dashboard UI
@app.route('/api/get_live_sensors', methods=['GET'])
def get_live_sensors():
    with data_lock:
        return jsonify(latest_sensor_data)

# --- GEMINI CROP RECOMMENDATION ---
@app.route('/api/get_recommendation', methods=['POST'])
def get_recommendation():
    request_data = request.get_json()
    state = request_data.get('state')
    soil = request_data.get('soil')
    
    if not state or not soil: 
        return jsonify({'error': 'Please select a State and Soil Type.'}), 400

    # Grab the exact sensor readings at the moment the button was clicked
    with data_lock: 
        current_data = latest_sensor_data.copy()

    try:
        # We tell Gemini exactly what role to play and feed it the live sensor data
        prompt = f"""
        You are an expert agricultural AI in India. Based on the following live farm conditions, recommend the single best crop to plant, a suitable fertilizer, and the expected yield.

        FARM CONDITIONS:
        - Location State: {state}
        - Soil Type: {soil}
        - Nitrogen (N): {current_data['n']}
        - Phosphorus (P): {current_data['p']}
        - Potassium (K): {current_data['k']}
        - Temperature: {current_data['temperature']}°C
        - Humidity: {current_data['humidity']}%
        - Soil pH: {current_data['ph']}
        - Rainfall: {current_data['rainfall']} mm
        - Soil Moisture: {current_data['soil_moisture']}%

        Respond strictly with a valid JSON object (no markdown formatting, no code blocks) containing exactly these keys:
        - "crop": The name of the recommended crop (e.g., "Wheat", "Rice", "Cotton").
        - "fertilizer": The specific recommended fertilizer (e.g., "Urea", "DAP", "NPK 14-35-14").
        - "yield": The expected yield as a string (e.g., "4.5 tonnes/hectare").
        """
        
        # Call Gemini 2.5 Flash
        response = vision_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up Markdown JSON blocks if Gemini adds them
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text.strip())

        # Send it back to the dashboard!
        return jsonify({
            'recommendation': {
                'crop': result['crop'], 
                'fertilizer': result['fertilizer'], 
                'yield': result['yield']
            }
        })

    except Exception as e:
        print(f"Gemini Rec Error: {e}")
        return jsonify({'error': f'AI Prediction failed. Details: {str(e)}'}), 500

# --- AI PLANT DISEASE DETECTION ---
@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    
    try:
        # Load image for AI
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """
        You are an expert plant pathologist AI. Analyze the provided leaf image and identify the disease based ONLY on the following list:
        1. Apple: Healthy, Apple scab, Black rot, Cedar apple rust.
        2. Grape: Healthy, Black rot, Esca (black measles), Leaf Blight.
        3. Peach: Healthy, Bacterial spot.
        4. Potato: Healthy, Early blight, Late blight.
        5. Tomato: Healthy, Early blight, Late blight, Bacterial spot, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Tomato mosaic virus, Tomato yellow leaf curl virus.
        6. Pepper: Healthy, Bacterial spot.
        7. Orange: Healthy, Huanglongbing (Citrus greening).

        Respond strictly with a valid JSON object (no markdown formatting, no code blocks) containing exactly these keys:
        - "disease": The specific plant and disease name (e.g., "Tomato - Early Blight") or "Healthy Plant".
        - "cause": A brief description of the pathogen causing it.
        - "cure": A brief recommended agricultural treatment.
        - "confidence": A number representing your confidence percentage (e.g., 95.5).
        """
        
        response = vision_model.generate_content([prompt, img])
        response_text = response.text.strip()
        
        # Clean up Markdown JSON blocks if AI adds them
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text.strip())
        return jsonify(result)
        
    except Exception as e:
        # I added `str(e)` here so the dashboard will tell you EXACTLY why it failed!
        print(f"AI Error: {e}")
        return jsonify({'error': f'AI processing failed. Details: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
