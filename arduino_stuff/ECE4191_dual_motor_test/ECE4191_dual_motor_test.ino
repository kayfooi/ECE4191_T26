// define constants
#define PI 3.1415926535897932384626433832795;

// define speed and direction pins for motors
// Might need PWM pins for speed
// Pins 2-13 and 44-46
// NAME THESE MOTORS LEFT AND RIGHT EVENTUALLY
int motorRightA = 30;
int motorRightB = 4;
int motorLeftA = 31;
int motorLeftB = 5;

// define encoder pins for motors
// Interrupt pins are 2-3 and 18-21
int encoderRight = 2;
int encoderRightin = 18;
int encoderLeft = 3;
int encoderLeftin = 19;

// define variables for encoder counts
volatile long encoderRightCount = 0;
volatile long encoderLeftCount = 0;

// define variables for PID control
long previousTime = 0;
float ePrevious = 0;
float eIntegral = 0;

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
  // put your main code here, to run repeatedly:
  
  // TEST


  MoveStraight(1);
  delay(10000);
  MoveStraight(-1);
  delay(10000);


/*
  MoveRotate(1);
  delay(10000);
  MoveRotate(-1);
  delay(10000);
  MoveStraight(0);
  delay(300000);
*/


  /*
  digitalWrite(motorRightdir, HIGH);
  digitalWrite(motorRightspeed, HIGH);
  digitalWrite(motorLeftdir, HIGH);
  digitalWrite(motorLeftspeed, HIGH);
  delay(3000);
  */

  // right motor reverse
  //digitalWrite(motorRightA, HIGH);
  //digitalWrite(motorRightB, LOW);
  // right motor forward
  //digitalWrite(motorRightA, LOW);
  //digitalWrite(motorRightB, HIGH);

  // left motor reverse
  //digitalWrite(motorLeftA, HIGH);
  //digitalWrite(motorLeftB, LOW);
  // left motor forward
  //digitalWrite(motorLeftA, LOW);
  //digitalWrite(motorLeftB, HIGH);

  /*
  // move forward
  digitalWrite(motorRightA, LOW);
  digitalWrite(motorRightB, HIGH);
  digitalWrite(motorLeftA, LOW);
  digitalWrite(motorLeftB, HIGH);
  */

  // spin both motors forward
  /*
  digitalWrite(motorRightdir, LOW);
  digitalWrite(motorRightspeed, 0);
  digitalWrite(motorLeftdir, LOW);
  digitalWrite(motorLeftspeed, 0);
  delay(1000);
  /*

  //digitalWrite(motorRightdir, LOW);
  //digitalWrite(motorRightspeed, HIGH);
  //digitalWrite(motorLeftdir, LOW);
  //digitalWrite(motorLeftspeed, HIGH);
  //delay(1000);
  
/*
  // Set desired setpoint for motor 2
  int target = encoderRightCount;

  // Move motor 1
  //digitalWrite(motorRightdir, 0);
  //analogWrite(motorRightspeed, 50);

  digitalWrite(motorRightdir, 0);
  analogWrite(motorRightspeed, 255);
  
  // PID gains and computation
  float kp = 2.0;
  float kd = 0.0;
  float ki = 0.0;
  float u = pidController(target, kp, kd, ki);

  // Control motor 2 based on PID
  moveMotor(motorLeftdir, motorLeftspeed, u);

*/

  // Print statements for debugging
  Serial.print(encoderRightCount);
  Serial.print(", ");
  Serial.println(encoderLeftCount);
  delay(1000);

/*
  analogWrite(motorRightspeed, 0);
  analogWrite(motorLeftspeed, 0);

  while (true);

  */


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

  // define number of encoder counts per rotation
  int constant_count = 900;
  // define wheel diameter
  int diameter = 54/1000; // [m] = [mm/1000]
  // distance travelled by wheel in one rotation
  float dist_per_rotation = diameter*PI;  // [m]
  // distance travelled per encoder count
  float dist_per_count = dist_per_rotation/constant_count;

  // encoder counts required to travel specified distance
  float encoder_counts_req = distance/dist_per_count;

  return round(encoder_counts_req);

}

/*
// Straight line movement function
void MoveStraight(int direction, float distance) {
  
  if (direction == 0) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorLeftA, LOW);
    digitalWrite(motorLeftA, LOW);
  }

  // set motor rotation direction
  // 1 indicates forward, 0 indicates reverse
  if (direction == 1) {
    digitalWrite(motorRightA, LOW);
    digitalWrite(motorLeftA, LOW);
  }
  if (direction == 0) {
    digitalWrite(motorRightA, HIGH);
    digitalWrite(motorLeftA, HIGH);
  }
  
  // call function to determine encoder counts required
  int dist_travelled = DistanceToEncoderCount(distance);
  // move motors while encoder count is less than this value
  while (encoderRightCount < dist_travelled) {
    // rotate motors at maximum speed
    digitalWrite(motorRightA, );
    digitalWrite(motorLeftA, );
  }

}
*/

void MoveStraight(int direction) {
  
  // set motor rotation direction
  // 1 indicates forward, -1 indicates reverse, 0 indicates stop
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
 
  /*
  // call function to determine encoder counts required
  int dist_travelled = DistanceToEncoderCount(distance);
  // move motors while encoder count is less than this value
  while (encoderRightCount < dist_travelled) {
    // rotate motors at maximum speed
    digitalWrite(motorRightB, speed);
    digitalWrite(motorLeftB, speed);
  }
  */

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
