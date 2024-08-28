import serial
import time

# Configure serial connection
ser = serial.Serial('/dev/serial0', 9600, timeout=0.5)
# Replace with your serial port

while True:
    ser.write(b'Hello from Raspberry Pi!\n')
    time.sleep(1)
