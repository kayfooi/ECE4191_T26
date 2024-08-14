# from ... import ...

import gpiozero
import time

# notes:
# - could use pi for vision, send high/low signals for direction to arduino to control motors & encoder stuff

# encoder stuff

# define pinouts etc.

encoder = gpiozero.RotaryEncoder(a=21, b=20, max_steps = 100000)

# Step through duty cycle values, slowly increasing the speed and changing the direction of motion
encoder.steps = 0
for j in range(10):
    pwm.value = j/10
    dir1.value = not dir1.value
    dir2.value = not dir1.value
    print('Duty cycle:',pwm.value,'Direction:',dir1.value)
    time.sleep(5.0)
    print('Counter:',encoder.steps,'Speed:',(encoder.steps)/5.0,'steps per second\n')
    encoder.steps = 0

pwm.value =0

pwm.off()