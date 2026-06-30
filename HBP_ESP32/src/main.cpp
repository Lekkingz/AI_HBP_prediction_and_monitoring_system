#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MLX90614.h>
#include "MAX30105.h"

// =====================================
// CONFIGURATION
// =====================================

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define SAMPLE_SIZE 875
#define MAX_RETRIES 5
#define RESULT_SCREEN_DURATION 3000
#define LED_BLINK_INTERVAL 500
#define FINGER_THRESHOLD 50000
#define IR_THRESHOLD_REMOVE 40000
#define WIFI_TIMEOUT 20000
#define HTTP_TIMEOUT 60000
#define WIFI_RECONNECT_INTERVAL 10000
#define UPLOAD_RETRY_WAIT 15000

// ===== NEW ===== 
// Heart rate detection configuration
#define MIN_HEART_RATE 40
#define MAX_HEART_RATE 200
#define PEAK_MIN_DISTANCE 20

// Respiratory rate detection configuration
#define RESPIRATORY_FILTER_WINDOW 21
#define MIN_RESPIRATORY_RATE 10
#define MAX_RESPIRATORY_RATE 24

// =====================================
// PIN DEFINITIONS
// =====================================

#define LED_GREEN 26
#define LED_YELLOW 27
#define LED_BLUE 14
#define LED_WHITE 12
#define LED_ORANGE 13
#define LED_RED 33
#define BUZZER_PIN 25
#define BUZZER_CHANNEL 0
#define BUZZER_FREQUENCY 2700
#define BUZZER_RESOLUTION 8
#define BUZZER_DUTY 128

#define I2C_SDA 21
#define I2C_SCL 22

// =====================================
// WIFI CREDENTIALS
// =====================================

const char* ssid = "Lekk 2.4g";
const char* password = "Cotlin355.";

// =====================================
// API CONFIGURATION
// =====================================

const char* serverURL = "https://ai-hbp-prediction-and-monitoring-system.onrender.com/predict";
const char* liveDataURL = "https://ai-hbp-prediction-and-monitoring-system.onrender.com/live-data";

// =====================================
// HARDWARE INSTANCES
// =====================================

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_MLX90614 mlx;
MAX30105 particleSensor;

// =====================================
// STATE MACHINE DEFINITIONS
// =====================================

enum SystemState {
    STATE_STARTUP,
    STATE_WIFI_CONNECT,
    STATE_WAITING_FINGER,
    STATE_COLLECTING_SAMPLES,
    STATE_SAMPLES_COMPLETE,
    STATE_UPLOADING,
    STATE_SHOWING_RESULTS,
    STATE_ERROR
};

// =====================================
// GLOBAL VARIABLES
// =====================================

SystemState currentState = STATE_STARTUP;
SystemState previousState = STATE_STARTUP;

float ppgSignal[SAMPLE_SIZE];
int sampleIndex = 0;

float bodyTemp = 0.0;
float predictedBP = 0.0;
float bpm = 75.0;
int respRate = 16;

String riskLevel = "WAITING";
float riskScore = 0.0;

// Timing variables
unsigned long stateStartTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long lastDisplayUpdate = 0;
unsigned long wifiConnectStart = 0;
unsigned long httpTimeout = 0;
unsigned long resultScreenStart = 0;

int currentResultScreen = 0;
int uploadRetries = 0;
bool ledBlinkState = false;
bool fingerDetected = false;

String errorMessage = "";

// ===== NEW =====
// Heart rate and respiratory rate buffers for analysis
float filteredSignal[SAMPLE_SIZE];
bool peakArray[SAMPLE_SIZE];

// =====================================
// LED FUNCTIONS
// =====================================

void ledInit() {
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    pinMode(LED_WHITE, OUTPUT);
    pinMode(LED_ORANGE, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_BLUE, LOW);
    digitalWrite(LED_WHITE, LOW);
    digitalWrite(LED_ORANGE, LOW);
    digitalWrite(LED_RED, LOW);
}

void ledTest() {
    digitalWrite(LED_GREEN, HIGH);
    delay(200);
    digitalWrite(LED_GREEN, LOW);
    delay(100);
    
    digitalWrite(LED_YELLOW, HIGH);
    delay(200);
    digitalWrite(LED_YELLOW, LOW);
    delay(100);
    
    digitalWrite(LED_BLUE, HIGH);
    delay(200);
    digitalWrite(LED_BLUE, LOW);
    delay(100);
    
    digitalWrite(LED_WHITE, HIGH);
    delay(200);
    digitalWrite(LED_WHITE, LOW);
    delay(100);
    
    digitalWrite(LED_ORANGE, HIGH);
    delay(200);
    digitalWrite(LED_ORANGE, LOW);
    delay(100);
    
    digitalWrite(LED_RED, HIGH);
    delay(200);
    digitalWrite(LED_RED, LOW);
    delay(100);
}

void ledAllOff() {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_BLUE, LOW);
    digitalWrite(LED_WHITE, LOW);
    digitalWrite(LED_ORANGE, LOW);
    digitalWrite(LED_RED, LOW);
}

void updateRiskLEDs() {
    ledAllOff();
    
    if (riskLevel == "LOW") {
        digitalWrite(LED_GREEN, HIGH);
    } else if (riskLevel == "MODERATE") {
        digitalWrite(LED_ORANGE, HIGH);
    } else if (riskLevel == "HIGH") {
        unsigned long currentTime = millis();
        if ((currentTime / 500) % 2 == 0) {
            digitalWrite(LED_RED, HIGH);
        } else {
            digitalWrite(LED_RED, LOW);
        }
    }
}

void updateStatusLED() {
    unsigned long currentTime = millis();

    ledAllOff();

    switch (currentState) {
        case STATE_WIFI_CONNECT:
            if ((currentTime - lastBlinkTime) > LED_BLINK_INTERVAL) {
                ledBlinkState = !ledBlinkState;
                digitalWrite(LED_GREEN, ledBlinkState ? HIGH : LOW);
                lastBlinkTime = currentTime;
            }
            break;

        case STATE_COLLECTING_SAMPLES:
            digitalWrite(LED_YELLOW, HIGH);
            break;

        case STATE_UPLOADING:
            if ((currentTime - lastBlinkTime) > LED_BLINK_INTERVAL) {
                ledBlinkState = !ledBlinkState;
                digitalWrite(LED_BLUE, ledBlinkState ? HIGH : LOW);
                lastBlinkTime = currentTime;
            }
            break;

        case STATE_SHOWING_RESULTS:
            updateRiskLEDs();
            break;

        case STATE_ERROR:
            if ((currentTime - lastBlinkTime) > LED_BLINK_INTERVAL) {
                ledBlinkState = !ledBlinkState;
                digitalWrite(LED_RED, ledBlinkState ? HIGH : LOW);
                lastBlinkTime = currentTime;
            }
            break;

        default:
            break;
    }
}

// =====================================
// BUZZER FUNCTIONS
// =====================================

void buzzerInit() {
    ledcSetup(BUZZER_CHANNEL, BUZZER_FREQUENCY, BUZZER_RESOLUTION);
    ledcAttachPin(BUZZER_PIN, BUZZER_CHANNEL);
}

void beep(int duration = 120) {
    ledcWriteTone(BUZZER_CHANNEL, 2800);
    delay(duration);
    ledcWriteTone(BUZZER_CHANNEL, 0);
    delay(20);
}

void beepDouble() {
    beep(120);
    delay(80);
    beep(120);
}

void beepLong() {
    beep(400);
}

void beepTriple() {
    beep(110);
    delay(90);
    beep(110);
    delay(90);
    beep(110);
}

void beepWarning() {
    beep(180);
    delay(80);
    beep(180);
}

void beepSamplesComplete() {
    beepLong();
}

// =====================================
// SERIAL LOGGING
// =====================================

void logSeparator() {
    Serial.println("========================");
}

void logStartup() {
    logSeparator();
    Serial.println("AI HBP MONITOR");
    Serial.println("Firmware v2.2");
    logSeparator();
}

void logWiFiConnected() {
    Serial.println("WiFi Connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

void logWaitingFinger() {
    Serial.println("Waiting Finger");
}

void logCollecting(int samples) {
    Serial.print("Collecting: ");
    Serial.print(samples);
    Serial.print("/");
    Serial.println(SAMPLE_SIZE);
}

void logCollectingComplete() {
    Serial.println("875/875");
}

void logUploading() {
    Serial.println("Uploading");
}

void logHTTPCode(int code) {
    Serial.print("HTTP ");
    Serial.println(code);
}

void logPredictionReceived() {
    Serial.println("Prediction Received");
}

void logResults() {
    Serial.print("BP: ");
    Serial.println(predictedBP, 1);
    Serial.print("Heart Rate: ");
    Serial.println(bpm, 0);
    Serial.print("Temperature: ");
    Serial.println(bodyTemp, 1);
    Serial.print("Respiratory Rate: ");
    Serial.println(respRate);
    Serial.print("Risk: ");
    Serial.println(riskLevel);
    Serial.print("Risk Score: ");
    Serial.println(riskScore, 2);
}

void logWaitingNext() {
    Serial.println("Waiting Next Reading");
    logSeparator();
}

void logError(String message) {
    Serial.print("ERROR: ");
    Serial.println(message);
}

// ===== NEW =====
// Heart rate and respiratory rate calculation logging
void logCalculatedMetrics() {
    Serial.println();
    Serial.println("--- CALCULATED METRICS ---");
    Serial.print("Calculated BPM: ");
    Serial.println(bpm, 1);
    Serial.print("Estimated Respiratory Rate: ");
    Serial.println(respRate);
    Serial.println();
}

// =====================================
// TEMPERATURE SENSOR
// =====================================

float getAverageTemp() {
    float sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += mlx.readObjectTempC();
        delay(20);
    }
    return sum / 10.0;
}

// =====================================
// FINGER DETECTION
// =====================================

bool isFingerPresent(long irValue) {
    return irValue > FINGER_THRESHOLD;
}

bool isFingerRemoved(long irValue) {
    return irValue < IR_THRESHOLD_REMOVE;
}

// ===== NEW =====
// Calculate heart rate from PPG signal using peak detection algorithm
float calculateBPM() {
    // Step 1: Normalize the signal (remove DC offset and scale)
    float minVal = ppgSignal[0];
    float maxVal = ppgSignal[0];
    
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        if (ppgSignal[i] < minVal) minVal = ppgSignal[i];
        if (ppgSignal[i] > maxVal) maxVal = ppgSignal[i];
    }
    
    float range = maxVal - minVal;
    if (range == 0) return 75.0; // Fallback if no signal variation
    
    // Step 2: Normalize signal to 0-100 range and create filtered version
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        filteredSignal[i] = ((ppgSignal[i] - minVal) / range) * 100.0;
    }
    
    // Step 3: Simple moving average filter (5-point window) to smooth noise
    for (int i = 2; i < SAMPLE_SIZE - 2; i++) {
        filteredSignal[i] = (filteredSignal[i-2] + filteredSignal[i-1] + 
                            filteredSignal[i] + filteredSignal[i+1] + 
                            filteredSignal[i+2]) / 5.0;
    }
    
    // Step 4: Peak detection - find local maxima
    int peakCount = 0;
    for (int i = PEAK_MIN_DISTANCE; i < SAMPLE_SIZE - PEAK_MIN_DISTANCE; i++) {
        bool isPeak = true;
        
        // Check if current point is greater than surrounding points
        for (int j = 1; j < PEAK_MIN_DISTANCE; j++) {
            if (filteredSignal[i] <= filteredSignal[i-j] || 
                filteredSignal[i] <= filteredSignal[i+j]) {
                isPeak = false;
                break;
            }
        }
        
        if (isPeak) {
            peakArray[peakCount] = true;
            peakCount++;
        }
    }
    
    // Step 5: Calculate BPM from peak count
    if (peakCount < 2) {
        return 75.0; // Fallback if insufficient peaks detected
    }
    
    // Estimate BPM based on peak density
    // Assuming 875 samples collected at roughly 100 Hz = 8.75 seconds
    float timeWindow = (float)SAMPLE_SIZE / 100.0; // time in seconds
    float peaksPerSecond = (float)peakCount / timeWindow;
    float calculatedBPM = peaksPerSecond * 60.0;
    
    // Constrain to realistic range
    if (calculatedBPM < MIN_HEART_RATE) {
        calculatedBPM = MIN_HEART_RATE;
    } else if (calculatedBPM > MAX_HEART_RATE) {
        calculatedBPM = MAX_HEART_RATE;
    }
    
    return calculatedBPM;
}

// ===== NEW =====
// Calculate respiratory rate from PPG signal using low-frequency modulation detection
int calculateRespiratoryRate() {
    // Step 1: Normalize the signal
    float minVal = ppgSignal[0];
    float maxVal = ppgSignal[0];
    
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        if (ppgSignal[i] < minVal) minVal = ppgSignal[i];
        if (ppgSignal[i] > maxVal) maxVal = ppgSignal[i];
    }
    
    float range = maxVal - minVal;
    if (range == 0) return 16; // Fallback
    
    // Step 2: Create normalized signal
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        filteredSignal[i] = ((ppgSignal[i] - minVal) / range) * 100.0;
    }
    
    // Step 3: Apply low-pass filter to extract respiratory component
    // Use weighted moving average (windowed sinc-like filter)
    float filterWindow = RESPIRATORY_FILTER_WINDOW;
    float respiratory[SAMPLE_SIZE];
    
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        respiratory[i] = 0;
        float weightSum = 0;
        
        int halfWindow = (int)(filterWindow / 2);
        for (int j = -halfWindow; j <= halfWindow; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < SAMPLE_SIZE) {
                // Hamming window weight
                float weight = 0.54 - 0.46 * cos(2.0 * 3.14159 * (j + halfWindow) / filterWindow);
                respiratory[i] += filteredSignal[idx] * weight;
                weightSum += weight;
            }
        }
        if (weightSum > 0) {
            respiratory[i] /= weightSum;
        }
    }
    
    // Step 4: Find peaks in respiratory signal (slower frequency = larger spacing)
    int respPeakCount = 0;
    int minPeakDistance = 40; // Minimum distance between respiratory peaks
    
    for (int i = minPeakDistance; i < SAMPLE_SIZE - minPeakDistance; i++) {
        bool isPeak = true;
        
        for (int j = 1; j < minPeakDistance; j++) {
            if (respiratory[i] <= respiratory[i-j] || 
                respiratory[i] <= respiratory[i+j]) {
                isPeak = false;
                break;
            }
        }
        
        if (isPeak) {
            respPeakCount++;
        }
    }
    
    // Step 5: Calculate respiratory rate
    if (respPeakCount < 1) {
        return 16; // Fallback if no peaks detected
    }
    
    // Time window in seconds (875 samples at ~100 Hz = ~8.75 seconds)
    float timeWindow = (float)SAMPLE_SIZE / 100.0;
    float peaksPerSecond = (float)respPeakCount / timeWindow;
    int calculatedRespRate = (int)(peaksPerSecond * 60.0 + 0.5); // Round to nearest int
    
    // Constrain to realistic range
    if (calculatedRespRate < MIN_RESPIRATORY_RATE) {
        calculatedRespRate = MIN_RESPIRATORY_RATE;
    } else if (calculatedRespRate > MAX_RESPIRATORY_RATE) {
        calculatedRespRate = MAX_RESPIRATORY_RATE;
    }
    
    return calculatedRespRate;
}

// =====================================
// OLED DISPLAY FUNCTIONS
// =====================================

void displayInit() {
    display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.display();
}

void showStartupScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.println("AI HBP");
    display.println("MONITOR");
    display.setTextSize(1);
    display.setCursor(0, 50);
    display.println("Firmware v2.2");
    display.display();
}

void showWiFiConnectingScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.println("Connecting");
    display.println("WiFi...");
    display.display();
}

void showWiFiConnectedScreen() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("WiFi Connected");
    display.setCursor(0, 20);
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.display();
}

void showWaitingFingerScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.println("Place Finger");
    display.setTextSize(1);
    display.setCursor(0, 50);
    display.println("Waiting...");
    display.display();
}

void drawProgressBar(int x, int y, int width, int height, int percentage) {
    display.drawRect(x, y, width, height, SSD1306_WHITE);
    int filledWidth = (width - 2) * percentage / 100;
    display.fillRect(x + 1, y + 1, filledWidth, height - 2, SSD1306_WHITE);
}

void showCollectingSamplesScreen() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Collecting Samples");
    
    int percentage = (sampleIndex * 100) / SAMPLE_SIZE;
    
    display.setCursor(0, 20);
    display.print("Sample: ");
    display.print(sampleIndex);
    display.print("/");
    display.println(SAMPLE_SIZE);
    
    display.setCursor(0, 35);
    display.print("Progress: ");
    display.print(percentage);
    display.println("%");
    
    drawProgressBar(0, 50, 128, 8, percentage);
    
    display.display();
}

void showSamplesCompleteScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.println("Samples");
    display.println("Complete");
    display.setTextSize(1);
    display.setCursor(0, 50);
    display.println("Uploading...");
    display.display();
}

void showUploadingScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.println("Uploading...");
    display.display();
}

void showServerErrorScreen() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Server Error");
    display.setCursor(0, 20);
    display.println("Retry Later");
    display.setCursor(0, 40);
    display.print("Retry: ");
    display.print(uploadRetries);
    display.print("/");
    display.println(MAX_RETRIES);
    display.display();
}

void showParseErrorScreen() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.println("Parse");
    display.println("Error");
    display.display();
}

void showFingerRemovedScreen() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Finger Removed");
    display.setCursor(0, 20);
    display.println("Restarting...");
    display.display();
}

void showResultScreen1() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Predicted BP");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print(predictedBP, 1);
    display.println(" mmHg");
    display.display();
}

void showResultScreen2() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Heart Rate");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print(bpm, 0);
    display.println(" BPM");
    display.display();
}

void showResultScreen3() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Temperature");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print(bodyTemp, 1);
    display.println(" C");
    display.display();
}

void showResultScreen4() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Respiratory Rate");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print(respRate);
    display.println(" breaths/min");
    display.display();
}

void showResultScreen5() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Risk Level");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.println(riskLevel);
    display.display();
}

void showResultScreen6() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Risk Score");
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print(riskScore, 2);
    display.display();
}

void showResultScreens() {
    unsigned long currentTime = millis();
    
    if ((currentTime - resultScreenStart) > RESULT_SCREEN_DURATION) {
        currentResultScreen = (currentResultScreen + 1) % 6;
        resultScreenStart = currentTime;
    }
    
    switch (currentResultScreen) {
        case 0:
            showResultScreen1();
            break;
        case 1:
            showResultScreen2();
            break;
        case 2:
            showResultScreen3();
            break;
        case 3:
            showResultScreen4();
            break;
        case 4:
            showResultScreen5();
            break;
        case 5:
            showResultScreen6();
            break;
    }
}

// =====================================
// WIFI CONNECTION
// =====================================

bool connectWiFi() {
    logStartup();
    
    stateStartTime = millis();
    wifiConnectStart = millis();
    
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED) {
        unsigned long elapsedTime = millis() - wifiConnectStart;
        
        if (elapsedTime > WIFI_TIMEOUT) {
            logError("WiFi Connection Timeout");
            return false;
        }
        
        delay(500);
    }
    
    logWiFiConnected();
    beepDouble();
    
    return true;
}

// ===== MODIFIED =====
// Updated to include respiratory_rate in JSON payload
String createPayload() {
    String payload;
    payload.reserve(7000);

    payload = "{\"ppg_signal\":[";

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        if (i > 0) {
            payload += ",";
        }

        payload += String(ppgSignal[i], 0);
    }

    payload += "],\"heart_rate\":";
    payload += String(bpm, 0);
    payload += ",\"temperature\":";
    payload += String(bodyTemp, 2);
    payload += ",\"respiratory_rate\":";
    payload += String(respRate);
    payload += "}";

    return payload;
}

// =====================================
// HTTP COMMUNICATION
// =====================================

bool sendToAI() {
    // ===== MODIFIED =====
    // Added calculated metrics logging before upload
    logCalculatedMetrics();
    
    String payload = createPayload();
    bool uploadSuccess = false;

    for (int attempt = 1; attempt <= MAX_RETRIES && !uploadSuccess; attempt++) {
        logSeparator();
        Serial.print("Attempt ");
        Serial.print(attempt);
        Serial.print("/");
        Serial.println(MAX_RETRIES);
        
        WiFiClientSecure client;
        client.setInsecure();
        client.setTimeout(HTTP_TIMEOUT);
        
        HTTPClient http;
        http.setTimeout(HTTP_TIMEOUT);
        
        if (!http.begin(client, serverURL)) {
            Serial.println("Connection failed");
            http.end();
            
            if (attempt < MAX_RETRIES) {
                delay(UPLOAD_RETRY_WAIT);
            }
            continue;
        }
        
        http.addHeader("Content-Type", "application/json");
        http.addHeader("Connection", "close");
        
        int httpCode = http.POST(payload);
        String response = http.getString();
        
        Serial.print("HTTP Code: ");
        Serial.println(httpCode);
        
        if (httpCode == HTTP_CODE_OK) {
            JsonDocument result;
            DeserializationError err = deserializeJson(result, response);
            http.end();
            
            if (!err) {
                predictedBP = result["predicted_systolic_bp"] | 0.0;
                riskLevel = String((const char*)(result["risk_level"] | "WAITING"));
                riskScore = result["risk_score"] | 0.0;
                bpm = result["heart_rate"] | bpm;
                bodyTemp = result["temperature"] | bodyTemp;
                respRate = result["respiratory_rate"] | respRate;
                
                uploadSuccess = true;
                
                Serial.println("Prediction received successfully");
                logResults();
                
                beep(100);
                digitalWrite(LED_WHITE, HIGH);
                delay(200);
                digitalWrite(LED_WHITE, LOW);
                
                if (riskLevel == "HIGH") {
                    beepTriple();
                }
            } else {
                Serial.println("JSON Parse Error");
            }
        } else {
            Serial.print("HTTP Error Response: ");
            Serial.println(response);
            http.end();
            
            if (attempt < MAX_RETRIES) {
                delay(UPLOAD_RETRY_WAIT);
            }
        }
    }
    
    if (!uploadSuccess) {
        Serial.println("Upload failed after all retries");
        beepWarning();
        beepWarning();
        return false;
    }
    
    return true;
}

// =====================================
// STATE MACHINE HANDLERS
// =====================================

void handleStateStartup() {
    if (previousState != STATE_STARTUP) {
        showStartupScreen();
        stateStartTime = millis();
        previousState = STATE_STARTUP;
        
        Serial.println("\n");
        logStartup();
        ledTest();
        beep(200);
    }
    
    if (millis() - stateStartTime > 2000) {
        currentState = STATE_WIFI_CONNECT;
    }
}

void handleStateWiFiConnect() {
    if (previousState != STATE_WIFI_CONNECT) {
        showWiFiConnectingScreen();
        previousState = STATE_WIFI_CONNECT;
        stateStartTime = millis();
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        showWiFiConnectedScreen();
        digitalWrite(LED_GREEN, HIGH);
        beepDouble();
        delay(2000);
        currentState = STATE_WAITING_FINGER;
        logWaitingFinger();
    } else if (millis() - stateStartTime > WIFI_TIMEOUT) {
        currentState = STATE_ERROR;
        errorMessage = "WiFi Timeout";
        logError(errorMessage);
    }
}

void handleStateWaitingFinger() {
    if (previousState != STATE_WAITING_FINGER) {
        showWaitingFingerScreen();
        previousState = STATE_WAITING_FINGER;
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_GREEN, HIGH);
        sampleIndex = 0;
    }
    
    long irValue = particleSensor.getIR();
    
    if (isFingerPresent(irValue)) {
        fingerDetected = true;
        digitalWrite(LED_YELLOW, HIGH);
        currentState = STATE_COLLECTING_SAMPLES;
    }
}

void handleStateCollectingSamples() {
    if (previousState != STATE_COLLECTING_SAMPLES) {
        showCollectingSamplesScreen();
        previousState = STATE_COLLECTING_SAMPLES;
        stateStartTime = millis();
        logCollecting(sampleIndex);
    }
    
    long irValue = particleSensor.getIR();
    
    if (isFingerRemoved(irValue)) {
        currentState = STATE_WAITING_FINGER;
        previousState = STATE_COLLECTING_SAMPLES;
        showFingerRemovedScreen();
        beepWarning();
        beepWarning();
        sampleIndex = 0;
        delay(2000);
        return;
    }
    
    if (irValue > FINGER_THRESHOLD) {
        ppgSignal[sampleIndex++] = (float)irValue;
        
        if ((millis() - lastDisplayUpdate) > 100) {
            showCollectingSamplesScreen();
            lastDisplayUpdate = millis();
        }
        
        if (sampleIndex >= SAMPLE_SIZE) {
            logCollectingComplete();
            currentState = STATE_SAMPLES_COMPLETE;
        }
    }
}

// ===== MODIFIED =====
// Calculate BPM and respiratory rate when samples complete
void handleStateSamplesComplete() {
    if (previousState != STATE_SAMPLES_COMPLETE) {
        beepSamplesComplete();
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_BLUE, HIGH);
        showSamplesCompleteScreen();
        previousState = STATE_SAMPLES_COMPLETE;
        stateStartTime = millis();
        
        // ===== NEW =====
        // Calculate real BPM and respiratory rate from collected samples
        Serial.println();
        Serial.println("--- ANALYZING PPG SIGNAL ---");
        
        bpm = calculateBPM();
        respRate = calculateRespiratoryRate();
        
        Serial.print("Heart Rate Analysis: ");
        Serial.print(bpm, 1);
        Serial.println(" BPM detected");
        Serial.print("Respiratory Analysis: ");
        Serial.print(respRate);
        Serial.println(" breaths/min estimated");
        Serial.println();
    }
    
    if (millis() - stateStartTime > 1500) {
        uploadRetries = 0;
        currentState = STATE_UPLOADING;
    }
}

void handleStateUploading() {
    if (previousState != STATE_UPLOADING) {
        showUploadingScreen();
        previousState = STATE_UPLOADING;
        stateStartTime = millis();
        uploadRetries = 0;
    }
    
    if (sendToAI()) {
        digitalWrite(LED_BLUE, LOW);
        resultScreenStart = millis();
        currentResultScreen = 0;
        currentState = STATE_SHOWING_RESULTS;
        logWaitingNext();
    } else {
        uploadRetries++;
        
        if (uploadRetries >= MAX_RETRIES) {
            currentState = STATE_ERROR;
            errorMessage = "Upload Failed";
            logError(errorMessage);
        } else {
            showServerErrorScreen();
            delay(3000);
        }
    }
}

void handleStateShowingResults() {
    if (previousState != STATE_SHOWING_RESULTS) {
        resultScreenStart = millis();
        currentResultScreen = 0;
        previousState = STATE_SHOWING_RESULTS;
    }
    
    long irValue = particleSensor.getIR();
    
    if (isFingerRemoved(irValue)) {
        currentState = STATE_WAITING_FINGER;
        digitalWrite(LED_YELLOW, LOW);
        logWaitingFinger();
    } else {
        showResultScreens();
    }
}

void handleStateError() {
    if (previousState != STATE_ERROR) {
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_BLUE, LOW);
        digitalWrite(LED_YELLOW, LOW);
        previousState = STATE_ERROR;
        stateStartTime = millis();
        display.clearDisplay();
        display.setTextSize(1);
        display.setCursor(0, 0);
        display.println("ERROR");
        display.println(errorMessage);
        display.display();
    }
    
    if (millis() - stateStartTime > 5000) {
        if (WiFi.status() == WL_CONNECTED) {
            currentState = STATE_WAITING_FINGER;
            logWaitingFinger();
        } else {
            currentState = STATE_WIFI_CONNECT;
        }
    }
}

void updateDisplay() {
    // Empty implementation for compatibility
}

void update() {
    switch (currentState) {
        case STATE_STARTUP:
            handleStateStartup();
            break;
        case STATE_WIFI_CONNECT:
            handleStateWiFiConnect();
            break;
        case STATE_WAITING_FINGER:
            handleStateWaitingFinger();
            break;
        case STATE_COLLECTING_SAMPLES:
            handleStateCollectingSamples();
            break;
        case STATE_SAMPLES_COMPLETE:
            handleStateSamplesComplete();
            break;
        case STATE_UPLOADING:
            handleStateUploading();
            break;
        case STATE_SHOWING_RESULTS:
            handleStateShowingResults();
            break;
        case STATE_ERROR:
            handleStateError();
            break;
    }
    
    updateStatusLED();
}

// =====================================
// SETUP
// =====================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    ledInit();
    buzzerInit();
    
    Wire.begin(I2C_SDA, I2C_SCL);
    
    displayInit();
    
    if (!mlx.begin()) {
        Serial.println("MLX90614 Failed");
    }
    
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        Serial.println("MAX30102 Failed");
    }
    
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    
    connectWiFi();
    
    currentState = STATE_STARTUP;
    previousState = STATE_STARTUP;
}

// =====================================
// MAIN LOOP
// =====================================

void loop() {
    bodyTemp = getAverageTemp();
    
    long irValue = particleSensor.getIR();
    
    update();
    
    delay(10);
}


