### Based off code provided in ECE4191 Github file called L298N_H-bridge_ICs.ipynb
### Theoretically, we could just make this into big object named Bot or something.
# Author: Edric Lay, 28/07
# Last edited: Edric Lay, 30/07
import time
import RPi.GPIO as GPIO 

# Assign pins to GPIOs
# motor_A = Left
motor_A_in1 = 18
motor_A_in2 = 23
motor_A_en = 13

# motor_B = Right
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

def motorDirection(motor: int = 0, direction: int = 0):
    """Decides what direction for the motor to rotate.

    Args:
        motor (int, optional): Chooses which motor to set rotation direction for
            0: Left motor
            1: Right motor
        direction (int, optional): Chooses direction of motor.
            0: None
            1: Clockwise
            2: Anticlockwise
            3+: N/A
    """
    # Error Checking :)
    if motor not in [0,1]:
        raise ValueError

    if direction not in [0,1,2]:
        raise ValueError

    # Motor Direction Selection
    if motor == 0:
        if direction == 0:
            GPIO.output(motor_A_in1, GPIO.LOW)
            GPIO.output(motor_A_in2, GPIO.LOW)

        elif direction == 1:
            GPIO.output(motor_A_in1, GPIO.LOW)
            GPIO.output(motor_A_in2, GPIO.HIGH)

        else: # direction == 2
            GPIO.output(motor_A_in1, GPIO.LOW)
            GPIO.output(motor_A_in2, GPIO.HIGH)
    
    elif motor == 1:
        if direction == 0:
            GPIO.output(motor_B_in1, GPIO.LOW)
            GPIO.output(motor_B_in2, GPIO.LOW)

        elif direction == 1:
            GPIO.output(motor_B_in1, GPIO.LOW)
            GPIO.output(motor_B_in2, GPIO.HIGH)

        else: # direction == 2
            GPIO.output(motor_B_in1, GPIO.LOW)
            GPIO.output(motor_B_in2, GPIO.HIGH)

def motorControl(dTime: int, direction: int, dutyL: int = 20, dutyR: int = 20, freq: int = 100):
    """Provides control of the entire motor system of the bot.

    Args:
        dTime (int): How long to drive the motors for. 
        direction (int): Direction to send the bot.
            0: Forwards
            1: Left
            2: Right
            3: Backwards
        duty (int, optional): % of max_speed. Defaults to 20.
        freq (int, optional): Frequency of PWM. Defaults to 100.
    """
    # Error Checking :)
    if dTime < 0:
        raise ValueError
    
    if direction not in [0,1,2,3]:
        raise ValueError

    if dutyL<0 or dutyL>100:
        raise ValueError
    
    if dutyR<0 or dutyR>100:
        raise ValueError
    
    if freq<=0:
        raise ValueError
    
    # Motor Speed Selection
    pwm_L = GPIO.PWM(motor_A_en, freq)
    pwm_R = GPIO.PWM(motor_B_en, freq)

    # Start PWM with a duty cycle of 0%
    pwm_L.start(0)
    pwm_R.start(0)

    # Change PWM duty cycle to duty% of max motor speed; speeds may differ due to differential drive to enable turning
    pwm_L.ChangeDutyCycle(dutyL)
    pwm_R.ChangeDutyCycle(dutyR)

    # Direction and Speed
    if direction == 0 or direction == 2:
        motorDirection(motor = 0, direction = 1)
        motorDirection(motor = 1, direction = 2)
         
    elif direction == 1 or direction == 3:
        pwm_L.ChangeDutyCycle(dutyL)
        pwm_R.ChangeDutyCycle(dutyR)

        motorDirection(motor = 0, direction = 2)
        motorDirection(motor = 1, direction = 1)
         
    start_time = time.time()

    while time.time() - start_time < dTime: # Drive the bot for that long
        pass

    # Bot should stop.
    pwm_L.stop()
    pwm_R.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    # Use to this test if the bot is driving properly.
    # Go Forward
    motorControl(dTime = 5, direction = 0)

    # Go Left
    motorControl(dTime = 5, direction = 1)

    # Go Right
    motorControl(dTime = 5, direction = 2)

    # Go Backward
    motorControl(dTime = 5, direction = 3)