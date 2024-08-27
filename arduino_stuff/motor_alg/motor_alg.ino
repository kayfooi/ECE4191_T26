//Arduinos & Movement Algs
#include <NewPing.h>
 
#define TRIGGER_PIN  12
#define ECHO_PIN     11
#define MAX_DISTANCE 200
#define S0 4
#define S1 5
#define S2 6
#define S3 7
#define sensorOut 8

  int frequency = 0;
 
  NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

  // defines pins numbers
  const int trigPin = 10;
  const int echoPin = 9;

  // defines variables
  long duration;
  int distance;

void setup() { //runs code once

//! Distance Sensing !
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input

  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

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

void loop() { //runs code repeatedly

// Wait for RPi to initialise

while()

// ! Distance Sensing

  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  

  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculating the distance in cm
  distance= duration*0.034/2;

  

  // Prints the duration on the Serial Monitor
  Serial.print("Duration: ");
  Serial.println(duration);

  // digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on 
  // Serial.println("LED is ON!");
  // delay(500);                        // wait for half a second

  // digitalWrite(LED_BUILTIN, LOW);    // turn the LED off 
  // Serial.println("LED is OFF!");
  // delay(500);                        // wait for half a second


// ! Colour Sensing !

  digitalWrite(S2,LOW);
  digitalWrite(S3,LOW);

  // Reading the output frequency
  frequency = pulseIn(sensorOut, LOW);

  // Printing the value on the serial monitor
  Serial.print("R= ");         //printing name
  Serial.print(frequency);     //printing RED color frequency
  Serial.print("  ");
  delay(100);

  // Setting Green filtered photodiodes to be read
  digitalWrite(S2,HIGH);
  digitalWrite(S3,HIGH);

  // Reading the output frequency
  frequency = pulseIn(sensorOut, LOW);

  // Printing the value on the serial monitor
  Serial.print("G= ");          //printing name
  Serial.print(frequency);      //printing GREEN color frequency
  Serial.print("  ");
  delay(100);

  // Setting Blue filtered photodiodes to be read
  digitalWrite(S2,LOW);
  digitalWrite(S3,HIGH);

  // Reading the output frequency
  frequency = pulseIn(sensorOut, LOW);

  // Printing the value on the serial monitor
  Serial.print("B= ");          //printing name
  Serial.print(frequency);      //printing RED color frequency
  Serial.println("  ");
  delay(100);

  map(int value, int old_min, int old_max, int new_min, int new_max)
  // maps 25->255, 70->0 and interpolates between. 
  // we invert the final range to make the dominant colour have the HIGHEST value (by default it's the LOWEST value).
  // this line must run continuously inside the loop() function of your code.
  int new_frequency = map(frequency, 25, 70, 255, 0)


  
}

