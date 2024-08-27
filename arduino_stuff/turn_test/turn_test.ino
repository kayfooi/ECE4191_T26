//turning test
// 2 ultrasonic pair sensors
#include <NewPing.h>
#include <L298NX2.h>


#define MAX_DISTANCE 4000
#define SONAR_NUM 3 // Number of sensors

// Sensor Constants

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
const unsigned int motorRightA = 34;
const unsigned int motorLeftA = 35;

const unsigned int motorRightB = 36;
const unsigned int motorLeftB = 37;
const unsigned int EN_B = 9;

// Initialize both motors
//L298NX2 motors(EN_A, IN1_A, IN2_A, EN_B, IN1_B, IN2_B);

  int d_left = 0;
  int d_mid = 0;
  int d_right = 0;
  int ball_detected = 0;
  int initial = 0;

  int direction = 0;


void setup() {
  // put your setup code here, to run once:
  for (uint8_t i = 0; i < SONAR_NUM; i++){
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  };

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:

  // ! Distance Sensing !
  //ReadCm();

  // int d_left = sonar[0].ping_cm();
  // delay(30);
  // int d_right = sonar[2].ping_cm();
  // delay(50);
  int d_middle = sonar[1].ping_cm();
  delay(30);

  // Serial.print("Left: ");
  // Serial.print(d_left);
  // Serial.println();
  Serial.print("Middle: ");
  Serial.print(d_mid);
  Serial.println();
  // Serial.print("Right: ");
  // Serial.print(d_right);
  // Serial.println();

  // digitalWrite(motorRightA, LOW);
  // digitalWrite(motorRightB, HIGH);
  // digitalWrite(motorLeftA, HIGH);
  // digitalWrite(motorLeftB, LOW);

  // delay(3000);


  // if (d_left <= d_mid) { //left
  //   direction = 1;
  // }
  // else if (d_right <= d_mid) { //right
  //   direction = -1;
  // }

  // MoveRotate(direction);

}

void motorStop() {
  digitalWrite(motorRightA, LOW);
  digitalWrite(motorRightB, LOW);
  digitalWrite(motorLeftA, LOW);
  digitalWrite(motorLeftB, LOW);
}



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
  int d_right = sonar[2].ping_cm();
  delay(50);
  int d_mid = sonar[1].ping_cm();
  delay(30);

  if (d_left == 0) {
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
