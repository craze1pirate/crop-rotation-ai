#include <WiFi.h>
#include <WiFiClientSecure.h> // For HTTPS
#include <HTTPClient.h>
#include <ArduinoJson.h>

// --- USER CONFIGURATION ---
const char* WIFI_SSID = "Pixel_3522";     
const char* WIFI_PASS = "shiyam123";

// Your Render URL
const char* SERVER_HOSTNAME = "crop-rotation-ai.onrender.com"; 
const int SERVER_PORT = 443; 
// --- END CONFIGURATION ---

const char* SERVER_ENDPOINT = "/api/sensor_data";

// --- Sensor Pin Definitions ---
// REAL SENSORS
#define RAIN_PIN 39     // Analog Pin for Rain Sensor
#define MOISTURE_PIN 32 // Analog Pin for Soil Moisture Sensor
// We have removed the DHT and NPK pins, as they are now simulated.

void setup() {
  Serial.begin(115200);
  
  // Setup REAL sensor pins
  pinMode(MOISTURE_PIN, INPUT); 
  pinMode(RAIN_PIN, INPUT);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
}

void loop() {
  WiFiClientSecure client;
  client.setInsecure(); // Bypass SSL certificate check
  HTTPClient http;

  // --- 1. READ REAL SENSORS ---
  
  // A. Soil Moisture (Real)
  // Map raw value (e.g., 4095-1500) to 0-100%
  int moisture_raw = analogRead(MOISTURE_PIN);
  int soil_moisture = map(moisture_raw, 4095, 1500, 0, 100); 
  soil_moisture = constrain(soil_moisture, 0, 100); // Keep it in 0-100 range

  // B. Rainfall (Real)
  // Map raw value (e.g., 4095-0) to 0-200mm
  int rain_raw = analogRead(RAIN_PIN);
  float rainfall = map(rain_raw, 4095, 0, 0, 200); 
  rainfall = constrain(rainfall, 0, 300);

  // --- 2. SIMULATE REMAINING SENSORS (Your Rules) ---
  
  // A. Temperature (Constant 26°C)
  float temperature = 26.0;

  // B. Humidity (Rarely fluctuate 85-87)
  float humidity = 85.0;
  int dice_roll = random(10); // Roll a 10-sided die
  if (dice_roll == 1) {      // 1-in-10 chance
    humidity = 86.0;
  } else if (dice_roll == 2) { // 1-in-10 chance
    humidity = 87.0;
  }
  // (8-in-10 chance it stays 85.0)

  // C. pH (Near 6 and 7)
  float ph = random(60, 71) / 10.0; // Random value from 6.0 to 7.0
  
  // D. NPK (Stable simulation)
  float n = random(50, 61); // Stable N value between 50-60
  float p = random(40, 51); // Stable P value between 40-50
  float k = random(30, 41); // Stable K value between 30-40
  
  
  // --- 3. SEND DATA ---
  JsonDocument doc;
  doc["n"] = n;
  doc["p"] = p;
  doc["k"] = k;
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["ph"] = ph;
  doc["rainfall"] = rainfall;
  doc["soil_moisture"] = soil_moisture; 

  String json_payload;
  serializeJson(doc, json_payload);
  
  String serverUrl = "https://" + String(SERVER_HOSTNAME) + ":" + String(SERVER_PORT) + String(SERVER_ENDPOINT);
  
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "application/json");

  Serial.println("\nSending data...");
  Serial.println(json_payload);
  
  int httpResponseCode = http.POST(json_payload);

  if (httpResponseCode > 0) {
    Serial.print("HTTP Response: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("HTTP Error: ");
    Serial.println(httpResponseCode);
  }
  
  http.end();

  // Wait 10 seconds before sending the next reading
  delay(10000); 
}