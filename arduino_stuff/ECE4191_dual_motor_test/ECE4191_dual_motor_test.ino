// define speed and direction pins for motors
// Might need PWM pins for speed
// Pins 2-13 and 44-46
// NAME THESE MOTORS LEFT AND RIGHT EVENTUALLY
int motor1dir = 31;
int motor1speed = 30;
int motor2dir = 33;
int motor2speed = 32;

// define encoder pins for motors
// Interrupt pins are 2-3 and 18-21
int encoder1 = 2;
int encoder1in = 18;
int encoder2 = 3;
int encoder2in = 19;

// define variables for encoder counts
volatile long encoder1Count = 0;
volatile long encoder2Count = 0;

// define variables for PID control
long previousTime = 0;
float ePrevious = 0;
float eIntegral = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(motor1dir, OUTPUT);
  pinMode(motor1speed, OUTPUT);
  pinMode(motor2dir, OUTPUT);
  pinMode(motor2speed, OUTPUT);
  pinMode(encoder1, OUTPUT);
  pinMode(encoder2, OUTPUT);

  // Interrupts
  attachInterrupt(digitalPinToInterrupt(encoder1), handleEncoder1, RISING);
  attachInterrupt(digitalPinToInterrupt(encoder2), handleEncoder2, RISING);

}

void loop() {
  // put your main code here, to run repeatedly:
  /*
  digitalWrite(motor1dir, HIGH);
  digitalWrite(motor1speed, LOW);
  digitalWrite(motor2dir, HIGH);
  digitalWrite(motor2speed, LOW);
  delay(1000);

  digitalWrite(motor1dir, LOW);
  digitalWrite(motor1speed, HIGH);
  digitalWrite(motor2dir, LOW);
  digitalWrite(motor2speed, HIGH);
  delay(1000);
  */

  // Set desired setpoint for motor 2
  int target = encoder1Count;

  // Move motor 1
  digitalWrite(motor1dir, 1);
  analogWrite(motor1speed, 50);
  
  // PID gains and computation
  float kp = 2.0;
  float kd = 0.0;
  float ki = 0.0;
  float u = pidController(target, kp, kd, ki);

  // Control motor 2 based on PID
  moveMotor(motor2dir, motor2speed, u);

  // Print statements for debugging
  Serial.print(encoder1Count);
  Serial.print(", ");
  Serial.println(encoder2Count);

}

// Functions called during interrupts

void handleEncoder1() {
  encoder1Count++;
}
void handleEncoder2() {
  encoder2Count++;
}

void moveMotor(int dirPin, int pwmPin, float u) {
  // Maximum motor speed
  float speed = fabs(u);
  if (speed > 255) {
    speed = 255;
  }
  // Stop the motor during overshoot
  if (encoder2Count > encoder1Count) {
    speed = 0;
  }
  // Control the motor
  int direction = 0;
  digitalWrite(dirPin, direction);
  analogWrite(pwmPin, speed);
}

float pidController(int target, float kp, float kd, float ki) {
  // Measure the time elapsed since the last iteration
  long currentTime = micros();
  float deltaT = ((float)(currentTime - previousTime))/1.0e6;

  // Compute the error, derivative and integral
  int e = encoder2Count - target;
  float eDerivative = (e - ePrevious)/deltaT;
  eIntegral = eIntegral + e*deltaT;

  // Compute the PID control signal
  float u = (kp*e) + (kd*eDerivative) + (ki*eIntegral);

  // Update variables for the next iteration
  previousTime = currentTime;
  ePrevious = e;

  return u;
}
