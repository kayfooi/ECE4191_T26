// define constants
#define PI 3.1415926535897932384626433832795

void decodeSerial(String serialInput);

// define speed and direction pins for motors
// Might need PWM pins for speed
// Pins 2-13 and 44-46
// NAME THESE MOTORS LEFT AND RIGHT EVENTUALLY
int motorRightA = 35;
int motorRightB = 34;
int motorLeftA = 37;
int motorLeftB = 36;

// define encoder pins for motors
// Interrupt pins are 2-3 and 18-21
int encoderRight = 3;
int encoderRightin = 18;
int encoderLeft = 2;
int encoderLeftin = 19 ;

// define variables for encoder counts
volatile long encoderRightCount = 0;
volatile long encoderLeftCount = 0;

// define variables for PID control
long previousTime = 0;
float ePrevious = 0;
float eIntegral = 0;

String serialInput;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("serial test");
  pinMode(motorRightA, OUTPUT);
  pinMode(motorRightB, OUTPUT);
  pinMode(motorLeftA, OUTPUT);
  pinMode(motorLeftB, OUTPUT);
  
  pinMode(encoderRight, INPUT);
  pinMode(encoderRightin, INPUT);
  pinMode(encoderLeft, INPUT);
  pinMode(encoderLeftin, INPUT);

  // Interrupts
  attachInterrupt(digitalPinToInterrupt(encoderRight), handleEncoder1, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderLeft), handleEncoder2, RISING);


}

void loop() {
//  // put your main code here, to run repeatedly:

//          Test when connected to RPI
//  if (Serial.available() > 0 ) {
//    String input = Serial.readStringUntil('\n')
//  }
  String testSerialMessage = "T_-200";
  decodeSerial(testSerialMessage);
  delay(10000);

}

// Functions called during interrupts

void handleEncoder1() {
  encoderRightCount++;
}
void handleEncoder2() {
  encoderLeftCount++;
}



void moveMotor(int dirPin, int pwmPin, float u) {
  // Maximum motor speed
  float speed = fabs(u);
  if (speed > 255) {
    speed = 255;
  }
  // Stop the motor during overshoot
  if (encoderLeftCount > encoderRightCount) {
    speed = 0;
  }
  // Control the motor
  int direction = 1; // direction set to 1 because right motor mounted in reverse
  digitalWrite(dirPin, direction);
  analogWrite(pwmPin, speed);
}

float pidController(int target, float kp, float kd, float ki) {
  // Measure the time elapsed since the last iteration
  long currentTime = micros();
  float deltaT = ((float)(currentTime - previousTime))/1.0e6;

  // Compute the error, derivative and integral
  int e = encoderLeftCount - target;
  float eDerivative = (e - ePrevious)/deltaT;
  eIntegral = eIntegral + e*deltaT;

  // Compute the PID control signal
  float u = (kp*e) + (kd*eDerivative) + (ki*eIntegral);

  // Update variables for the next iteration
  previousTime = currentTime;
  ePrevious = e;

  return u;
}


// Distance to encoder count translation function (helper function)
int DistanceToEncoderCount(float distance) { 
  Serial.println("in disttoencoder");

  // define number of encoder counts per rotation
  int constant_count = 900;
  // define wheel diameter
  int diameter = 54; // [mm]
  // distance travelled by wheel in one rotation
  float dist_per_rotation = diameter*PI;  // [m*1000] = [mm]
  // distance travelled per encoder count
  float dist_per_count = dist_per_rotation/constant_count;
  Serial.println(diameter);
  // encoder counts required to travel specified distance
  float encoder_counts_req = distance/dist_per_count;

  return round(encoder_counts_req);

}

// Straight line movement function
void MoveStraight(int direction) {
  
  // set motor rotation direction
  // 1 indicates forward, -1 indicates reverse, 0 indicates stop
  // 2 indicates Right Forward, 3 indicates Left Forward
  // -2 indicates Right Backward, -3 indicates Left Backward
  if (direction == 1) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightB, HIGH);
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftB, HIGH);
  }
  if (direction == -1) {
    digitalWrite(motorRightA, HIGH);
    digitalWrite(motorRightB, LOW);
    digitalWrite(motorLeftA, HIGH);
    digitalWrite(motorLeftB, LOW);
  }
   if (direction == 0) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightB, LOW);
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftB, LOW);
  }
    if (direction == 2) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightB, HIGH);
  }
      if (direction == 3) {
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftB, HIGH);
  }

      if (direction == -2) {
    digitalWrite(motorRightA, HIGH);
    digitalWrite(motorRightB, LOW);
  }
  
      if (direction == -3) {
    digitalWrite(motorLeftA, HIGH);
    digitalWrite(motorLeftB, LOW);
  }


}

void MoveRotate(int direction) {

  // direction = 1 = clockwise, direction = -1 = anticlockwise
  if (direction == 1) {
    digitalWrite(motorRightA, HIGH);
    digitalWrite(motorRightB, LOW);
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftB, HIGH);
  }
  if (direction == -1) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightB, HIGH);
    digitalWrite(motorLeftA, HIGH);
    digitalWrite(motorLeftB, LOW);
  }
}

void DistanceToStraight(int distance, int direction){
// Function converts distance input into motor movement
  int encoderCountsToDist = DistanceToEncoderCount(distance);
  Serial.println("entered correct if statement");
//  Serial.println(encoderCountsToDist);
//  Serial.println(distance);
  if(direction == 1){
    Serial.println(encoderLeftCount);
     while(encoderLeftCount < encoderCountsToDist){
      MoveStraight(1);
     }
  }
  if(direction == -1){
    Serial.println(encoderLeftCount);
     while(encoderLeftCount < encoderCountsToDist){
      MoveStraight(-1);
     }
  }     
   MoveStraight(0);
   delay(1000);
   encoderLeftCount = 0; 
//   return;
  }


void AngleToRotate(int angle, int direction){ // direction = 1 = clockwise, direction = -1 = anticlockwise
// Function converts input angle into motor movement
  if (direction == 1) {
    Serial.println(encoderLeftCount);
    while (encoderLeftCount < 200*angle){
      MoveStraight(3);
      MoveStraight(-2);
    }
  
//    if (encoderLeftCount >= 200*angle){
    MoveStraight(0);
    Serial.println(encoderLeftCount);
    delay(1000);
    encoderLeftCount = 0;
//    return;
//    }  
  }
  if (direction == -1) {
    while (encoderLeftCount < 200*angle){
      MoveStraight(-3);
      MoveStraight(2);
      Serial.println(encoderLeftCount);
    }
  
//    if (encoderLeftCount >= 200*angle){
    MoveStraight(0);
    Serial.println(encoderLeftCount);
    delay(1000);
    encoderLeftCount = 0;
//    return;
//    }  
  }  
}



void decodeSerial(String serialInput){
  int distance_angle = serialInput.substring(2).toInt();
//  Serial.println(serialInput);
//  Serial.println(distance_angle);
//  Serial.println(serialInput.charAt(0));
  if(serialInput.charAt(0) == 'R'){ // Rotational input
    if(distance_angle < 0){
      AngleToRotate(abs(distance_angle), -1);
    }
    else if(distance_angle > 0){
      AngleToRotate(abs(distance_angle), 1);
    }
  }
  if(serialInput.charAt(0) == 'T'){ // Translate / Straight line input 
//    Serial.println("entered translate");
    if(distance_angle < 0){
      DistanceToStraight(abs(distance_angle), -1);
//      Serial.println("entered correct if statement");
    }
    else if(distance_angle > 0){
      DistanceToStraight(abs(distance_angle), 1);
    }    
  }
}



// functions for ultrasonic

// 1 - rotate until ball found
// 2 - go forward until ball is within certain distance
// 3 - make correctional movements to ensure ball is closest to middle sensor 
