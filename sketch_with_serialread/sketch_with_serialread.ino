// Syringe Infusion Pump with AI Pain Detection
// Controls stepper motor based on AI model output (0 = No Pain, 1 = Pain)

// Pin definitions for A4988 driver
#define STEP_PIN 3    // STEP pin for A4988
#define DIR_PIN 2     // DIR pin for A4988
#define ENABLE_PIN 5  // ENABLE pin for A4988 (optional, to enable/disable driver)

// Motor parameters
#define STEPS_PER_REVOLUTION 200  // Steps per revolution for NEMA 17 (1.8°/step)
#define MOTOR_SPEED_RPM 60        // Motor speed in RPM (adjust as needed)
#define STEP_DELAY_US 6000        // Microseconds between steps (calculated for RPM)

// Variables
int aiOutput = 0;  // AI model output (0 = No Pain, 1 = Pain)
bool motorRunning = false;  // Motor state

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);
  
  // Set pin modes for A4988
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  
  // Initialize pins
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);  // Set direction (HIGH = clockwise, adjust if needed)
  digitalWrite(ENABLE_PIN, LOW); // Enable driver (LOW = enabled)
  
  // Wait for serial to stabilize
  delay(1000);
}

void loop() {
  // Check if serial data is available
  if (Serial.available() > 0) {
    char input = Serial.read();  // Read incoming byte
    
    // Convert char to integer (expecting '0' or '1')
    if (input == '0') {
      aiOutput = 0;  // No Pain
    } else if (input == '1') {
      aiOutput = 1;  // Pain
    } else {
      // Ignore invalid inputs
      return;
    }
    
    // Control motor based on AI output
    if (aiOutput == 0 && !motorRunning) {
      // Start motor for No Pain
      digitalWrite(ENABLE_PIN, LOW); // Ensure driver is enabled
      motorRunning = true;
    } else if (aiOutput == 1 && motorRunning) {
      // Stop motor for Pain
      digitalWrite(ENABLE_PIN, HIGH); // Disable driver
      motorRunning = false;
    }
  }
  
  // Run motor if enabled
  if (motorRunning) {
    // Generate step pulse
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(10);  // Short pulse
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(STEP_DELAY_US);  // Delay between steps
  }
}