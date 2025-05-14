#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <EEPROM.h>
#include <ArduinoOTA.h>

// Configuration
#define EEPROM_SIZE 96
#define RELAY_PIN 5            // GPIO5 for relay
#define LED_PIN 2              // GPIO2 for onboard LED (onboard LED)
#define LOCAL_PORT 12345       // Port for receiving UDP packets

WiFiUDP udp;
String ssid, password;

// Function to save Wi-Fi credentials to EEPROM
void saveCredentials(const String& ssid, const String& password) {
  EEPROM.begin(EEPROM_SIZE);
  for (int i = 0; i < ssid.length(); i++) {
    EEPROM.write(i, ssid[i]);
  }
  EEPROM.write(ssid.length(), '\0');  // Null-terminate SSID

  for (int i = 0; i < password.length(); i++) {
    EEPROM.write(32 + i, password[i]);
  }
  EEPROM.write(32 + password.length(), '\0');  // Null-terminate password

  EEPROM.commit();
  Serial.println("Wi-Fi credentials saved to EEPROM.");
}

// Function to load Wi-Fi credentials from EEPROM
bool loadCredentials(String& ssid, String& password) {
  EEPROM.begin(EEPROM_SIZE);

  char ssidBuf[32];
  char passwordBuf[64];

  for (int i = 0; i < 32; i++) {
    ssidBuf[i] = EEPROM.read(i);
  }
  ssidBuf[31] = '\0';

  for (int i = 0; i < 64; i++) {
    passwordBuf[i] = EEPROM.read(32 + i);
  }
  passwordBuf[63] = '\0';

  ssid = String(ssidBuf);
  password = String(passwordBuf);

  return ssid.length() > 0 && password.length() > 0;
}

// Function to erase Wi-Fi credentials
void eraseCredentials() {
  EEPROM.begin(EEPROM_SIZE);
  for (int i = 0; i < EEPROM_SIZE; i++) {
    EEPROM.write(i, 0);
  }
  EEPROM.commit();
  Serial.println("Wi-Fi credentials erased.");
}

// Function to start OTA updates
void startOTA() {
  ArduinoOTA.onStart([]() {
    Serial.println("Start OTA update");
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA Update Complete");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("OTA Error[%u]: ", error);
  });
  ArduinoOTA.begin();
  Serial.println("OTA Ready");
}

// Function to connect to Wi-Fi
void connectToWiFi(const String& ssid, const String& password) {
  Serial.printf("Connecting to '%s'...\n", ssid.c_str());
  WiFi.begin(ssid.c_str(), password.c_str());

  int retries = 20;
  while (WiFi.status() != WL_CONNECTED && retries > 0) {
    delay(500);
    Serial.print(".");
    retries--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to Wi-Fi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    startOTA();
    udp.begin(LOCAL_PORT);
    Serial.printf("Listening for UDP packets on port %d\n", LOCAL_PORT);
  } else {
    Serial.println("\nFailed to connect to Wi-Fi.");
  }
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Relay OFF
  digitalWrite(LED_PIN, HIGH);  // LED OFF (inverted logic)

  Serial.begin(115200);
  Serial.println("\nStarting up...");

  WiFi.mode(WIFI_STA);

  if (loadCredentials(ssid, password)) {
    connectToWiFi(ssid, password);
  } else {
    Serial.println("No Wi-Fi credentials found. Please update the firmware to configure.");
  }
}

void loop() {
  ArduinoOTA.handle();

  // UDP Communication
  char packetBuffer[255];
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = '\0';
    }
    String command = String(packetBuffer);
    Serial.printf("Received packet: %s\n", command.c_str());

    if (command == "ON") {
      digitalWrite(RELAY_PIN, HIGH);  // Turn relay ON
      digitalWrite(LED_PIN, LOW);    // Turn LED ON (inverted logic)
      Serial.println("Relay and LED turned ON.");
    } else if (command == "OFF") {
      digitalWrite(RELAY_PIN, LOW);  // Turn relay OFF
      digitalWrite(LED_PIN, HIGH);   // Turn LED OFF (inverted logic)
      Serial.println("Relay and LED turned OFF.");
    } else if (command == "ERASE") {
      eraseCredentials();
      Serial.println("Credentials erased. Restarting...");
      delay(1000);
      ESP.restart();
    }
  }
}