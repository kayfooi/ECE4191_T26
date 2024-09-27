# Raspberry Pi + MG90S Servo PWM Control Python Code
#
#
import RPi.GPIO as GPIO
import time

# setup the GPIO pin for the servo
servo_pin1 = 18 # GPIO pin 18, rPi pin 12
# servo_pin2 = 15
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin1,GPIO.OUT)
# GPIO.setup(servo_pin2,GPIO.OUT)

# setup PWM process
pwm = GPIO.PWM(servo_pin1,50) # 50 Hz (20 ms PWM period)
# pwm1 = GPIO.PWM(servo_pin2,50)

pwm.start(7) # start PWM by rotating to 90 degrees

def ServoAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin1, True)
    # GPIO.output(servo_pin2, True)
    pwm.ChangeDutyCycle(duty)
    # pwm1.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servo_pin1, False)
    # GPIO.output(servo_pin2, False)
    pwm.ChangeDutyCycle(0)

for ii in range(0,3):
    # pwm.ChangeDutyCycle(2.0) # rotate to 0 degrees
    ServoAngle(0)
    time.sleep(0.5)
    # pwm.ChangeDutyCycle(12.0) # rotate to 180 degrees
    ServoAngle(180)
    time.sleep(0.5)
    # pwm.ChangeDutyCycle(7.0) # rotate to 90 degrees
    ServoAngle(90)
    time.sleep(0.5)

pwm.ChangeDutyCycle(0) # this prevents jitter
# pwm1.ChangeDutyCycle(0)
pwm.stop() # stops the pwm on 13
# pwm1.stop()
GPIO.cleanup() # good practice when finished using a pin