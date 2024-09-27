import pigpio
from time import sleep, time
pi = pigpio.pi()

import unittest

SERVO_GPIO = 7
pi.set_mode(SERVO_GPIO, pigpio.OUTPUT)

def servo_set_angle(pin, angle, speed=-1):
    """
    Set angle of servo

    Parameters
    ---
    pin: int
        GPIO number of servo control pin
    angle: int
        Angle in degrees to set the servos position to
    speed: int
        Movement speed in degrees per second. If -1 no limit to speed
    """
    assert(0 <= angle <= 180), "Angle must be within 0 and 180 degrees"
    max_pulse = 2400
    min_pulse = 600
    get_pw = lambda a: (max_pulse - min_pulse) * a / 180 + min_pulse
    target_pulse_width = get_pw(angle)

    # Go straight to the angle
    if speed == -1:
        pi.set_servo_pulsewidth(pin, target_pulse_width)
        return
    
    # Increment servo at a set speed
    pulse_speed = get_pw(speed) - min_pulse
    update_delay = 1/50 # sec (Update the servo pulsewidth every update_delay seconds)
    increment = pulse_speed * update_delay # Increment pulse width by this much every step
    current_pw = pi.get_servo_pulsewidth(pin)
    direction = 1 if target_pulse_width > current_pw else -1
    print(current_pw, increment, target_pulse_width, pulse_speed, increment)
    count = 0
    while (current_pw + direction * increment - target_pulse_width) * direction <= 0:
        print(current_pw)
        current_pw += direction * increment
        assert(600 <= current_pw <= 2400), "Pulse Width must be between 600 and 2400"
        pi.set_servo_pulsewidth(pin, current_pw)
        sleep(update_delay)
        count += 1
    print(count)

    # Complete the rotation
    pi.set_servo_pulsewidth(pin, target_pulse_width)
    

    
class TestServo(unittest.TestCase):
    
    @unittest.skip("working")
    def test_servo_angle(self):
        servo_set_angle(SERVO_GPIO, 180)
        sleep(1)
        self.assertEqual(2400, pi.get_servo_pulsewidth(SERVO_GPIO))
        servo_set_angle(SERVO_GPIO, 90)
        sleep(1)
        self.assertEqual(1500, pi.get_servo_pulsewidth(SERVO_GPIO))
        servo_set_angle(SERVO_GPIO, 0)
        sleep(1)
        self.assertEqual(600,pi.get_servo_pulsewidth(SERVO_GPIO))
        servo_set_angle(SERVO_GPIO, 133)
        sleep(1)
        self.assertEqual(int((2400-600) * (133/180) + 600),pi.get_servo_pulsewidth(SERVO_GPIO))
        
        # Turn off
        pi.set_servo_pulsewidth(SERVO_GPIO, 0)
    
    def test_servo_speed(self):

        # Rotate from 180deg to 0deg in 2 seconds
        servo_set_angle(SERVO_GPIO, 180)
        sleep(1)
        s = time()
        servo_set_angle(SERVO_GPIO, 0, 90)
        e = time()
        
        print(e-s)
        self.assertLess(abs((e-s) - 2), 0.3, "Servo should have taken around 2 seconds")
        self.assertEqual(pi.get_servo_pulsewidth(SERVO_GPIO), 600)

        # Rotate from 0deg to 120deg in 4 seconds
        sleep(0.1)
        s = time()
        servo_set_angle(SERVO_GPIO, 120, 30)
        e = time()
        
        print(e-s)
        self.assertLess(abs((e-s) - 4), 0.3, "Servo should have taken around 4 seconds")
        self.assertEqual(pi.get_servo_pulsewidth(SERVO_GPIO), int((2400-600) * (120/180) + 600))

        # Rotate from 120deg to 90deg in 0.167 seconds
        sleep(0.1)
        s = time()
        servo_set_angle(SERVO_GPIO, 90, 180)
        e = time()
        
        print(e-s)
        self.assertLess(abs((e-s) - .167), 0.3, "Servo should have taken around 0.167 seconds")
        self.assertEqual(pi.get_servo_pulsewidth(SERVO_GPIO), 1500)

if __name__ == "__main__":
    unittest.main()