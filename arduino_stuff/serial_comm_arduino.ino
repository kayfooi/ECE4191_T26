bool received = false;
String buffer;

void setup() {
    Serial.begin(9600); // baud rate must align with RPi
}

void loop() {
    // Send stuff to the PI
    Serial.println("Hello RPi, from Arduino");
    bool received;

    if (received) {
        // Read instruction values
        char instruction;
        float value;
        sscanf(buffer, "%c_%.2f", &instruction, &value);

        // TODO: implement instruction
        delay(150)
        
        // Return successful message (or error msg of some kind)
        Serial.print(instruction);
        Serial.print("_");
        Serial.print(value, 2);
        Serial.println("_COMPLETE");
        
    }

    received = false
    String buffer;
    // Recevie stuff from the PI
    if (Serial.available() > 0) {
        buffer = Serial.readStringUntil('\n');
        Serial.print("Arduino received: ")
        Serial.println(buffer);
        received = true
    }

    delay(350);
}