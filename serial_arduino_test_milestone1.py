import serial
import time

# Configure serial connection
ser = serial.Serial('/dev/ttyS0', 9600, timeout=0.5)
# Replace with your serial port

while True:
    ser.write(b'T_100\n')
    time.sleep(3) #sleep time in seconds 
    ser.write(b'R_-50\n')
    time.sleep(3)
    ser.write(b'T_-100\n')
    time.sleep(3) #sleep time in seconds 
    ser.write(b'R_50\n')
    time.sleep(10)
