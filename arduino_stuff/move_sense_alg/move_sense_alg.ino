// movement algorithm
/*
Split into 2 parts: ultrasonic sensing & motor behaviour based on sensors
- if left or right sensors detect less distance compared to middle sensor,
  robot turns so that middle sensor has the least distance to the object
- when ball within distance = 100, LED lights up and outputs "ball detected",
  then robot goes forward to touch robot

What we need to code:
- after ball detected, find way back to start
*/
#include <NewPing.h>
#include <L298NX2.h>

// Colour sensor Constants
#define S0 4
#define S1 5
#define S2 6
#define S3 7
#define sensorOut 8

// Ultrasonic constants
#define MAX_DISTANCE 1500
#define SONAR_NUM 3 // Number of sensors

// Sensor Constants

int frequency[] = {0, 0, 0};
int prev_frequ[] = {0, 0, 255}; //default blue for blue court
int new_frequ[] = {0, 0, 0};
int stop = 0;

const int trigPins[] = {49, 51, 53};
const int echoPins[] = {48, 50, 52};
const int LEDpin = 47;

// for milestone 1
const int stop_distance = 20;
 
  NewPing sonar[SONAR_NUM] = {
    NewPing(trigPins[0], echoPins[0], MAX_DISTANCE),
    NewPing(trigPins[1], echoPins[1], MAX_DISTANCE),
    NewPing(trigPins[2], echoPins[2], MAX_DISTANCE)
  };

// Motor Constants
const float BASE_SPEED = 300.0; //to check & change based on robot
const int MIN_SPEED = 30; // so motors don't stall
const float MOTOR_FACTOR = BASE_SPEED / 100;

// time between pings
const unsigned long u_wait = 50; //ultrasonic
const unsigned long m_wait = 20; //motor

//memory variables
unsigned long u_Cms; //ultrasonic current
unsigned long u_Pms; //ultrasonic previous

unsigned long m_Cms; //motor current
unsigned long m_Pms; //motor previous

//Motor Instances
const unsigned int EN_A = 3;
const unsigned int IN1_A = 34;
const unsigned int IN2_A = 35;

const unsigned int IN1_B = 36;
const unsigned int IN2_B = 37;
const unsigned int EN_B = 9;

// Initialize both motors
L298NX2 motors(EN_A, IN1_A, IN2_A, EN_B, IN1_B, IN2_B);

  int d_left = 0;
  int d_mid = 0;
  int d_right = 0;
  int ball_detected = 0;
  int initial = 0;


void setup() {
  // put your setup code here, to run once:

  // ! Ultrasonic Sensing !
  for (uint8_t i = 0; i < SONAR_NUM; i++){
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  };

  // ! Colour Sensing !
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(sensorOut, INPUT);
  
  // Setting frequency-scaling to 20%
  digitalWrite(S0,HIGH);
  digitalWrite(S1,LOW);

  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:

  while (stop == 0) { //stops whole thing at last sequence when white detected

// ! Distance Sensing !
  ReadCm();
  delay(10);

// ! Motor Control !

  motorControl();

  Serial.println();

  }

}

// Functions

void ReadCm() {
  u_Cms = millis();
  if (u_Cms > u_Pms + u_wait) {

  
  // Clears trigger pins
  for (uint8_t j = 0; j < SONAR_NUM; j++){
    digitalWrite(trigPins[j], LOW);
  }
  delayMicroseconds(2);

  // Gets distance from sensors
  int d_left = sonar[0].ping_cm();
  delay(30);
  int d_mid = sonar[1].ping_cm();
  delay(50);
  int d_right = sonar[2].ping_cm();
  delay(30);

  if (d_left == 0) { //if sensors output 0, means nothing is there, so put max distance
    d_left = MAX_DISTANCE+1;
  }
  else if (d_mid == 0) {
    d_mid = MAX_DISTANCE+1;
  }
  else if (d_right == 0) {
    d_right = MAX_DISTANCE+1;
  }

  u_Pms = u_Cms;

  }
};

// motor instructions
void motorStop() {
  digitalWrite(motorRightA, LOW);
  digitalWrite(motorRightB, LOW);
  digitalWrite(motorLeftA, LOW);
  digitalWrite(motorLeftB, LOW);
}

void MoveRotate(int direction) { //add in angle

  // direction = 1 = clockwise, direction = -1 = anticlockwise
  if (direction == 1) { //right
    digitalWrite(motorRightA, HIGH);
    digitalWrite(motorRightB, LOW);
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftB, HIGH);

    //add in angle 10 degrees

  }
  if (direction == -1) { //left
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightB, HIGH);
    digitalWrite(motorLeftA, HIGH);
    digitalWrite(motorLeftB, LOW);

    //add in angle 10 degrees
  }
}

// motor behaviour

void motorControl() {

  //initial move forward
  if (initial == 0) { // to move into centre of court
    motors.forward();
    delay(5000);
    initial == 1;
  };

  //if ball within reach
  if (d_left || d_mid || d_right <= 100) {
    Serial.print("Ball detected! Yippeee");
    digitalWrite(LEDpin, HIGH); // turns on LED
    motors.forward();
    delay(3000); //delays 3 seconds
    motors.stop();
    ball_detected = 1;
    delay(3000);
  }

  //if smth detected
  else if (d_left || d_mid || d_right <= MAX_DISTANCE) {
    if (d_left <= d_mid) { //turn left
      direction = -1;
      MoveRotate(direction);
      delay(30);
      direction = 0;
    }
    else if (d_right <= d_mid) { //turn right
      direction = 1;
      MoveRotate(direction);
      delay(30);
      direction = 0;
    }
    else {
      motors.forward();
    }
  }

  //if nothing is detected
  if (d_left || d_mid || d_right >= MAX_DISTANCE) {
    //spin around 360 till something is there
    direction = 1;
    MoveRotate(direction);
    delay(30);
    direction = 0;
    //or spin 90 degrees and go forward a bit
    //if encoder picks up 2 rotations then turn & move forward for a bit
  }


  //if ball is detected
  if (ball_detected == 1) { //to get back to start point

    //back away & turn around

    //move forward till line detencted (use colour sensor)
    ColourSense();
    // follow line till colour sensor picks up tape
  }
}

// Colour sensor stuff
void ColourSense() {
  //Setting red diodes to be read
  digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);

  // Read output frequency
  frequency[0] = pulseIn(sensorOut, LOW);
  new_frequ[0] = map(frequency[0], 25, 70, 255, 0);

  // Setting Green filtered photodiodes to be read
  digitalWrite(S2,HIGH);
  digitalWrite(S3,HIGH);

  // Reading the output frequency
  frequency[1] = pulseIn(sensorOut, LOW);
  new_frequ[1] = map(frequency[0], 25, 70, 255, 0);

  // Setting Blue filtered photodiodes to be read
  digitalWrite(S2,LOW);
  digitalWrite(S3,HIGH);

  // Reading the output frequency
  frequency[2] = pulseIn(sensorOut, LOW);
  new_frequ[2] = map(frequency[0], 25, 70, 255, 0);

  // Compare previous data to detect blue or white
  if ((new_frequ[0] -100) >= prev_frequ[0] && new_frequ[1] -100) >= prev_frequ[1]) {
    Serial.print("White detected! Yippee");
    stop = 1; //stops robot completely
  }

  for (uint8_t k = 0; k < 3; k++) { //replace memory with new data
    prev_frequ[k] = frequency[k];
  }

}
