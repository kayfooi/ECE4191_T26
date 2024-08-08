from gpiozero import Motor

motor1 = Motor(4, 14)
motor2 = Motor(17,27)

# drives motor forward
motor1.forward()

# drives motor back
motor2.backward()

#drives halfspeed
motor1.forward(0.5)

# reverse
while True:
    sleep(5)
    motor1.reverse()
    motor2.reverse()

# stops motors
motor1.stop()
motor2.stop()

