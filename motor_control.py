import time
import RPi.GPIO as GPIO

# Assign pins to GPIOs (For demo only, different GPIOs can be used in practice)
motor_A_in1 = 18
motor_A_in2 = 23
motor_A_en = 13

motor_B_in2 = 24
motor_B_in1 = 25
motor_B_en = 19

# Set GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_A_in1, GPIO.OUT)
GPIO.setup(motor_A_in2, GPIO.OUT)
GPIO.setup(motor_B_in1, GPIO.OUT)
GPIO.setup(motor_B_in2, GPIO.OUT)
GPIO.setup(motor_A_en, GPIO.OUT)
GPIO.setup(motor_B_en, GPIO.OUT)

## Motor Direction Control
start_time = time.time()

# Rotate both motor L and R anticlockwise for 3 seconds
while time.time() - start_time < 3:
    GPIO.output(motor_A_en, GPIO.HIGH)
    GPIO.output(motor_B_en, GPIO.HIGH)

    GPIO.output(motor_A_in1, GPIO.LOW)
    GPIO.output(motor_A_in2, GPIO.HIGH)

    GPIO.output(motor_B_in1, GPIO.LOW)
    GPIO.output(motor_B_in2, GPIO.HIGH)

GPIO.cleanup()


## Motor Speed Control w/ PWM

# Create PWM instance with a frequency of 100 Hz
pwm_L = GPIO.PWM(motor_A_en, 100)
pwm_R = GPIO.PWM(motor_B_en, 100)

# Start PWM with a duty cycle of 0%
pwm_L.start(0)
pwm_R.start(0)

# Change PWM duty cycle to 20%
pwm_L.ChangeDutyCycle(20)
pwm_R.ChangeDutyCycle(20)

start_time = time.time()

# Rotate both motor L and R anticlockwise for 3 seconds with 20% of the full motor speed
while time.time() - start_time < 3:
    GPIO.output(motor_A_in1, GPIO.LOW)
    GPIO.output(motor_A_in2, GPIO.HIGH)

    GPIO.output(motor_B_in1, GPIO.LOW)
    GPIO.output(motor_B_in2, GPIO.HIGH)

pwm_L.stop()
pwm_R.stop()
GPIO.cleanup()