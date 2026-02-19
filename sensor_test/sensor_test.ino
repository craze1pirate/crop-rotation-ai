#include <DHT.h>

// --- PINS ---
#define DHT_PIN 4
#define DHT_TYPE DHT22
#define RAIN_PIN 39
#define MOISTURE_PIN 32

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(RAIN_PIN, INPUT);
  pinMode(MOISTURE_PIN, INPUT);
  
  Serial.println("--- HARDWARE TEST TOOL STARTING ---");
  delay(2000);
}

void loop() {
  Serial.println("\n--------------------------------");
  
  // 1. Test DHT22
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  
  Serial.print("DHT22 Temp: ");
  if (isnan(t)) Serial.print("ERROR"); else Serial.print(t);
  Serial.print(" °C  |  Humidity: ");
  if (isnan(h)) Serial.print("ERROR"); else Serial.print(h);
  Serial.println(" %");

  // 2. Test Soil Moisture
  // Usually: 4095 (Air) -> ~1500 (Water)
  int raw_moisture = analogRead(MOISTURE_PIN);
  Serial.print("Soil Moisture RAW: ");
  Serial.print(raw_moisture);
  
  if (raw_moisture > 4000) Serial.print(" (Likely DRY/AIR)");
  else if (raw_moisture < 2000) Serial.print(" (Likely WET/WATER)");
  Serial.println();

  // 3. Test Rain Sensor
  // Usually: 4095 (Dry) -> Lower values (Rain)
  int raw_rain = analogRead(RAIN_PIN);
  Serial.print("Rain Sensor RAW:   ");
  Serial.print(raw_rain);
  
  if (raw_rain > 4000) Serial.print(" (NO RAIN)");
  else if (raw_rain < 3000) Serial.print(" (DETECTING RAIN)");
  Serial.println();

  delay(2000);
}