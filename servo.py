import pigpio
from time import sleep, time
import unittest

class Servo:
    MAX_PULSE = 2400 # usec
    MIN_PULSE = 600
    
    def __init__(self, pi, pin, home_angle):
        self.pi = pi
        self.pin = pin
        self.set_angle(home_angle)
        # pi.set_mode(pin, pigpio.OUTPUT)
        

    def set_angle(self, angle, speed=-1):
        """
        Set angle of servo

        Parameters
        ---
        angle: int
            Angle in degrees to set the servos position to
        speed: int
            Movement speed in degrees per second. If -1 no limit to speed
        """
        assert(0 <= angle <= 180), "Angle must be within 0 and 180 degrees"
        
        get_pw = lambda a: (self.MAX_PULSE - self.MIN_PULSE) * a / 180 + self.MIN_PULSE
        target_pulse_width = get_pw(angle)

        # Go straight to the angle
        if speed == -1:
            self.pi.set_servo_pulsewidth(self.pin, target_pulse_width)
            return
        
        # self.pi.set_mode(self.pin, 1)
        # Increment servo at a set speed
        pulse_speed = get_pw(speed) - self.MIN_PULSE
        update_delay = 1/50 # sec (Update the servo pulsewidth every update_delay seconds)
        increment = pulse_speed * update_delay # Increment pulse width by this much every step
        current_pw = self.pi.get_servo_pulsewidth(self.pin)
        direction = 1 if target_pulse_width > current_pw else -1
        # print(current_pw, increment, target_pulse_width, pulse_speed, increment)
        count = 0
        while (current_pw + direction * increment - target_pulse_width) * direction <= 0:
            # print(current_pw)
            current_pw += direction * increment
            # assert(600 <= current_pw <= 2400), "Pulse Width must be between 600 and 2400"
            if 600 <= current_pw <= 2400:
                self.pi.set_servo_pulsewidth(self.pin, current_pw)
            else:
                break
            sleep(update_delay)
            count += 1
        # print(count)

        # Complete the rotation
        self.pi.set_servo_pulsewidth(self.pin, target_pulse_width)

    def get_angle(self):
        return 180 * (self.pi.get_servo_pulsewidth(self.pin) - self.MIN_PULSE) / (self.MAX_PULSE - self.MIN_PULSE)
    
    def stop(self):
        """
        Stop servo
        """
        self.pi.set_servo_pulsewidth(self.pin, 0)

class TestServo(unittest.TestCase):
    def setUp(self):
        PADDLE_SERVO_GPIO = 14
        TIPPING_SERVO_GPIO = 7
        pi = pigpio.pi()
         #self.servo = Servo(pi, PADDLE_SERVO_GPIO, 180)
        self.servo = Servo(pi, TIPPING_SERVO_GPIO, 60)

    @unittest.skip("working")
    def test_servo_angle(self):
        # set to 180deg
        self.servo.set_angle(180)
        sleep(1)
        self.assertAlmostEqual(self.servo.get_angle(), 180)
        
        # set to 0 deg
        self.servo.set_angle(0)
        sleep(1)
        self.assertAlmostEqual(self.servo.get_angle(), 0)

        # set to 101deg
        self.servo.set_angle(101)
        sleep(1)
        self.assertAlmostEqual(self.servo.get_angle(), 101)
        
        # set to 90deg (home position)
        self.servo.set_angle(90)
        sleep(1)
        self.assertAlmostEqual(self.servo.get_angle(), 90)
        self.servo.stop()
    
    @unittest.skip("working")
    def test_servo_speed(self):
        # Rotate from 180deg to 0deg in 2 seconds
        self.servo.set_angle(180)
        sleep(1)
        s = time()
        self.servo.set_angle(0, 90) # 90 deg/sec
        e = time()
        
        self.assertLess(abs((e-s) - 2), 0.3, "Servo should have taken around 2 seconds")
        self.assertEqual(self.servo.get_angle(), 0)

        # Rotate from 0deg to 120deg in 4 seconds
        sleep(0.1)
        s = time()
        self.servo.set_angle(120, 30)
        e = time()
        
        self.assertLess(abs((e-s) - 4), 0.3, "Servo should have taken around 4 seconds")
        self.assertEqual(self.servo.get_angle(), 120)

        # Rotate from 120deg to 90deg in 0.167 seconds
        sleep(0.1)
        s = time()
        self.servo.set_angle(90, 180)
        e = time()
        
        self.assertLess(abs((e-s) - .167), 0.3, "Servo should have taken around 0.167 seconds")
        self.assertEqual(self.servo.get_angle(), 90)
    
    def test_angles(self):
        """
        Test random angles to tune parameters
        """
        # Paddle Mechanism
        self.servo.set_angle(150, 50)
        sleep(1)
        self.servo.set_angle(60, 50)
        sleep(1)
        self.servo.set_angle(150, 50)
        sleep(1)
        self.servo.stop()
        # Dumping Mechanisms
        # self.servo.set_angle(85, 15)
        # sleep(1)
        # self.servo.set_angle(177, 15)

if __name__ == "__main__":
    unittest.main()