const int ledPin = 13;  // pin number with built-in LED

void setup() {
    Serial.begin(9600); // baud rate must align with RPi
    pinMode(ledPin, OUTPUT);
    digitalWrite(ledPin, HIGH);
}

void loop() {
    // Recevie stuff from the PI
    if (Serial.available()) {
        String buffer = Serial.readStringUntil('\n');
        // Read instruction values
        char instruction; // either R for rotation or T for translation
        int value_int;
        int value_decimal;
        sscanf(buffer.c_str(), "%c_%d.%d", &instruction, &value_int, &value_decimal);
        
        // Amount to rotate or translate
        float value = value_int + (value_decimal / 1000.0);
        digitalWrite(ledPin, LOW);
        
        switch (instruction) {
            case 'R':
                // TODO: Rotate `value` degrees anti-clockwise
                // Note that value can be negative
                break;
            case 'T':
                // TODO: Translate `value` meters forward
                // Note that value can be negative
                break;
            default:
                Serial.print("Invalid instruction ");
                Serial.println(instruction);
                break;
            }


        delay(100);
        
        digitalWrite(ledPin, HIGH);
        // Return successful message (or error msg of some kind)
        Serial.print(instruction);
        Serial.print("_");
        Serial.print(value, 3);
        Serial.println("_COMPLETE");
    }

    Serial.println("LOOP");
    delay(1000);
}
