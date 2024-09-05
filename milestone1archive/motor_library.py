from gpiozero import Motor
from time import sleep

# import time
# import RPi.GPIO

motor1 = Motor(4, 14, 12)
motor2 = Motor(17,27, 13)

# motor calibration --> forward, back, turn 45 degrees left & right

# drives motor forward
motor1.forward()
motor2.backward()
sleep(2)

# drives motor back


#drives halfspeed
motor1.forward(0.5)
sleep(2)

# reverse
while True:
    sleep(5)
    motor1.reverse()
    motor2.reverse()

# stops motors
motor1.stop()
motor2.stop()




