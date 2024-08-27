typedef struct {
    int Trig;
    int Echo;
} ultrasonic_sensor;

typedef struct {
    int In1;
    int In2;
    int Enable;
} motor;

ultrasonic_sensor UltrasonicSensor = {};
motor MotorL = {};
motor MotorR = {};

void
setup() {
    Serial.begin(9600);
    
    //////////////////////////////////////
    // ULTRASONIC SENSOR
    //////////////////////////////////////
  UltrasonicSensor.Trig = 12;
  UltrasonicSensor.Echo = 13;
    pinMode(UltrasonicSensor.Trig, ...);
    pinMode(UltrasonicSensor.Echo, ...);
    
    //////////////////////////////////////
    // MOTORS
    //////////////////////////////////////
    MotorL.In1 = 2;
    MotorL.In2 = 4;
    MotorL.Enable = 5;
    pinMode(MotorL.In1,    ...);
    pinMode(MotorL.In2,    ...);
    pinMode(MotorL.Enable, ...);
    
    MotorR.In1 = 7;
    MotorR.In2 = 8;
    MotorR.Enable = 3;
    pinMode(MotorR.In1,    ...);
    pinMode(MotorR.In2,    ...);
    pinMode(MotorR.Enable, ...);
}

void
loop() {
    /*
    // Clears the trigPin
    digitalWrite(UltrasonicSensor.Trig, LOW);
    delayMicroseconds(2);
    
  // Sends a pulse
    digitalWrite(UltrasonicSensor.Trig, HIGH);
    delayMicroseconds(10);
    digitalWrite(UltrasonicSensor.Trig, LOW);
    
    // Reads the echoPin, returns the sound wave round trip time in microseconds
    float Duration = pulseIn(UltrasonicSensor.Echo, HIGH);
    
    // Calculates the distance based on the sound wave round trip time
    int Distance = Duration*0.034f/2;
    */
    
    
    // Drive forwards
    int MotorLSpeed = 100;
    int MotorRSpeed = 100;

    digitalWrite(MotorL.In1, HIGH);
    digitalWrite(MotorL.In2, LOW);
    analogWrite(MotorL.Enable, MotorLSpeed);
    digitalWrite(MotorR.In1, HIGH);
    digitalWrite(MotorR.In2, LOW);
    analogWrite(MotorR.Enable, MotorRSpeed);
    
    // Drive backwards
    digitalWrite(MotorL.In1, LOW);
    digitalWrite(MotorL.In2, HIGH);
    analogWrite(MotorL.Enable, Motor LSpeed);
    digitalWrite(MotorR.In1, LOW);
    digitalWrite(MotorR.In2, HIGH);
    analogWrite(MotorR.Enable, MotorRSpeed);

    // Turn left
      //Left wheel turns back
    digitalWrite(MotorL.In1, LOW);
    digitalWrite(Motor.In2, HIGH);
    analogWrite(Motor.Enable, MotorLSpeed);
      //Right wheel turns forward
    digitalWrite(MotorR.In1, HIGH);
    digitalWrite(MotorR.In2, LOW);
    analogWrite(MotorR.Enable, MotorRSpeed);

    // Turn right
      //Left wheel turns forward
    digitalWrite(MotorL.In1, HIGH);
    digitalWrite(Motor.In2, LOW);
    analogWrite(Motor.Enable, MotorLSpeed);
      //Right wheel turns back
    digitalWrite(MotorR.In1, LOW);
    digitalWrite(MotorR.In2, HIGH);
    analogWrite(MotorR.Enable, MotorRSpeed);

    delay(10);


}