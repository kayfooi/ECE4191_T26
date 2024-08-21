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
        // Serial.print("Arduino received: ");
        // Serial.println(buffer);
        // Read instruction values
        char instruction;
        int value_int;
        int value_decimal;
        sscanf(buffer.c_str(), "%c_%d.%d", &instruction, &value_int, &value_decimal);
        float value = value_int + (value_decimal / 1000.0);
        digitalWrite(ledPin, LOW);
        // TODO: implement instruction
        delay(1000);
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
