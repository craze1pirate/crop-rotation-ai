#include <WiFi.h>
#include <WiFiClientSecure.h> // For HTTPS
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h> 

// --- USER CONFIGURATION ---
const char* WIFI_SSID = "Pixel_3522";     
const char* WIFI_PASS = "shiyam123"; 

// Get this from your Render.com dashboard after deploying
// Example: "crop-ai-project.onrender.com"
const char* SERVER_HOSTNAME = "crop-rotation-ai.onrender.com"; // <-- REPLACE THIS
const int SERVER_PORT = 443; // HTTPS port
// --- END CONFIGURATION ---

const char* SERVER_ENDPOINT = "/api/sensor_data";

// --- Sensor Pin Definitions ---
#define DHT_PIN 4      // Digital pin connected to the DHT sensor
#define DHT_TYPE DHT22   // Using a DHT22 sensor
#define PH_PIN 36      // Analog pin for pH sensor
#define RAIN_PIN 39    // Analog pin for rain sensor
#define NPK_PIN 34     // Analog pin for the (simulated) NPK/EC sensor
#define MOISTURE_PIN 32 // Added Soil Moisture Pin

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(MOISTURE_PIN, INPUT); // Set pin for moisture sensor

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Create a secure WiFi client
  WiFiClientSecure client;
  // This is insecure, but fine for our prototype. It bypasses SSL certificate validation.
  client.setInsecure(); 
  
  HTTPClient http;

  // --- 1. Read all sensors (Simulating ALL unconnected sensors) ---
  
  // We simulate these because the sensors weren't fully connected or were "floating"
  float humidity = random(60, 85);     
  float temperature = random(20, 30);  
  float soil_moisture = random(30, 70); // Simulate 30-70%
  float ph = random(55, 75) / 10.0;     // Simulate pH 5.5 - 7.5
  float rainfall = random(40, 120);     // Simulate rainfall 40mm - 120mm

  // We simulate NPK as well, as discussed
  float n = random(40, 90);    
  float p = random(30, 60);    
  float k = random(20, 50);    
  
  // --- 2. Create JSON packet ---
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
  
  // --- 3. Send data to server via HTTPS ---
  
  // Build the full HTTPS URL
  String serverUrl = "https://" + String(SERVER_HOSTNAME) + ":" + String(SERVER_PORT) + String(SERVER_ENDPOINT);
  
  // Begin the HTTPS request
  http.begin(client, serverUrl); // Use the secure client
  http.addHeader("Content-Type", "application/json");

  Serial.println("\nSending data to server...");
  Serial.println(serverUrl); // Print the URL we are trying to reach
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