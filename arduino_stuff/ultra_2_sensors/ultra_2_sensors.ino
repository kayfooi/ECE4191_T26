// 2 ultrasonic pair sensors
#include <NewPing.h>
#include <L298NX2.h>
//#include <Adafruit_MotorShield.h>
//#include <Pololu3piPlus32U4.h>
//#include <AFMotor.h>
// using namespace Pololu3piPlus32U4;

// Buzzer buzzer;
// Motors motors;

#define MAX_DISTANCE 4000
#define SONAR_NUM 3 // Number of sensors

// Sensor Constants

const int trigPins[] = {2, 3, 4};
const int echoPins[] = {5, 6, 7};

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
const unsigned int IN1_A = 5;
const unsigned int IN2_A = 6;

const unsigned int IN1_B = 7;
const unsigned int IN2_B = 8;
const unsigned int EN_B = 9;

// Initialize both motors
L298NX2 motors(EN_A, IN1_A, IN2_A, EN_B, IN1_B, IN2_B);

// Initialize motors (M1 and M2)
// Adafruit_DCMotor motorLeft(1);  // M1 on the shield
// Adafruit_DCMotor motorRight(2); // M2 on the shield

  int d_left = 0;
  int d_mid = 0;
  int d_right = 0;
  int ball_detected = 0;
  int initial = 0;

void setup() {
  // put your setup code here, to run once:
  for (uint8_t i = 0; i < SONAR_NUM; i++){
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  };

  // pinMode(, OUTPUT);
  // pinMode(trigPins[1], OUTPUT);
  // pinMode(ECHO_PIN_1, INPUT);
  // pinMode(ECHO_PIN_2, INPUT);


  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:

// ! Distance Sensing !
  ReadCm();

// ! Motor Control !

  motorControl();

  // Clear trigger pin
  // for (uint8_t j = 0; j < SONAR_NUM; j++){
  //   digitalWrite(trigPins[j], LOW);
  // }
  // delayMicroseconds(2);

  // int d_left = sonar[0].ping_cm();
  // int d_mid = sonar[1].ping_cm();
  // int d_right = sonar[2].ping_cm();




  // for (uint8_t i = 0; i < SONAR_NUM; i++) { //loops thru each sensor
  //   delay(50); //waits 50ms b/w pings
  //   Serial.print(i);
  //   Serial.print("=");
  //   int distance = sonar[i].ping_cm();
  //   Serial.print(distance);

  // }

  Serial.println();

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
  int d_mid = sonar[1].ping_cm();
  int d_right = sonar[2].ping_cm();

  u_Pms = u_Cms;

  }
};

void moveForward(int speed) {
  motors.forward();
  // motorLeft.setSpeed(speed);
  // motorRight.setSpeed(speed);

  // motorLeft.run(BACKWARD);
  // motorRight.run(FORWARD);
};

void moveBackward(int speed) {
  // motorLeft.setSpeed(speed);
  // motorRight.setSpeed(speed);

  // motorLeft.run(FORWARD);
  // motorRight.run(BACKWARD);
  motors.backward();
};

void turn(int dir) {
  if (dir == 1) { //left
    // motorLeft.setSpeed(0);
    // motorRight.setSpeed(BASE_SPEED);

    // motorLeft.run(FORWARD);
    // motorRight.run(FORWARD);
    motors.backwardB();
  }
  else { //right
    // motorLeft.setSpeed(BASE_SPEED);
    // motorRight.setSpeed(0);

    // motorLeft.run(BACKWARD);
    // motorRight.run(BACKWARD);
    motors.backwardA();
  }
}

void stopMoving() {
  // motorLeft.run(RELEASE);
  // motorRight.run(RELEASE);
  motors.stop();
}

void setMotors() {
  m_Cms = millis();
  if (m_Cms > m_Pms + m_wait) {
    //starts at base speed
    float leftSpeed = BASE_SPEED;
    float rightSpeed = BASE_SPEED;

    //
  }
}

void motorControl() {

  //initial move forward
  if (initial == 0) {
    moveForward(BASE_SPEED);
    delay(5000);
    init == 1;
  };

  //if smth detected
  if (d_left || d_mid || d_right <= MAX_DISTANCE) {
    if (d_left >= d_mid) {
      turn(1);
    }
    else if (d_right >= d_mid) {
      turn(0);
    }
    else {
      moveForward(BASE_SPEED);
    }
  }
  //if ball within reach
  if (d_mid = 20) {
    Serial.print("Ball detected! Yippeee");
    ball_detected = 1;
    moveForward(BASE_SPEED/2);
    delay(3000); //delays 3 seconds
  }

  //if nothing is detected
  if (d_left || d_mid || d_right >= MAX_DISTANCE) {
    //spin around 360 till something is there
    turn(1);
    //or spin 90 degrees and go forward a bit
    //if encoder picks up 2 rotations then turn & move forward for a bit
  }


  //if ball is detected
  if (ball_detected == 1) {
    //move forward to touch ball

    //back away & turn around

    //move forward till line detencted

    // follow line till colour sensor picks up tape
  }
}
