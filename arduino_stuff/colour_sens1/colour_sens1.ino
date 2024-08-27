//colour sensor test
#define COLOUROUT A5
#define COLOURS2 A4
#define COLOURS3 A3
void setup() {
 Serial.begin(115200);
 pinMode(COLOUROUT,INPUT);
 pinMode(COLOURS2,OUTPUT);
 pinMode(COLOURS3,OUTPUT);
}
void loop() {
 colourred();
 Serial.print("RED:");
 Serial.print(getintensity());
 colourgreen();
 Serial.print("GREEN:");
 Serial.print(getintensity());
 colourblue();
 Serial.print("BLUE:");
 Serial.println(getintensity());
 delay(200);
}
void colourred(){ //select red
 digitalWrite(COLOURS2,LOW);
 digitalWrite(COLOURS3,LOW);
}
void colourblue(){ //select blue
 digitalWrite(COLOURS2,LOW);
 digitalWrite(COLOURS3,HIGH);
}
void colourwhite(){ //select white
 digitalWrite(COLOURS2,HIGH);
 digitalWrite(COLOURS3,LOW);
}
void colourgreen(){ //select green
 digitalWrite(COLOURS2,HIGH);
 digitalWrite(COLOURS3,HIGH);
}
int getintensity(){ //measure intensity with oversampling
 int a=0;
 int b=255;
 for(int i=0;i<10;i++){a=a+pulseIn(COLOUROUT,LOW);}
 if(a>9){b=2550/a;}
 return b;
}