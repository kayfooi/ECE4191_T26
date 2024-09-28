#!/usr/bin/env python
import pigpio
import time
import unittest

# PINOUT (aligns with RPi wiring spreadsheet GPIO Numbers on drive)
LEFT_ENC_A = 2
LEFT_EN = 13
LEFT_IN1 = 19
LEFT_IN2 = 26

RIGHT_ENC_A = 3
RIGHT_EN = 12
RIGHT_IN1 = 16
RIGHT_IN2 = 20

IR__BALLDETECT_IN = 5 # GPIO Number for IR Sensor (S pin)

class DCMotor:
    def __init__(self, pi:int, enc:int, en:int, in1:int, in2:int, for_dir = [0, 1]):
        """
        Handles driving motors and counting encoders
        
        Parameters
        ---
        pi : pigpio pi object
            reference to the raspberry pi
        enc : int
            encoder GPIO pin number
        en : int
            enable GPIO pin number (PWM)
        in1 : int
            IN1 GPIO pin number (direction control)
        in2 : int
            IN2 GPIO pin number (direction control)
        dir : list(int)
            Values of IN1 and IN2 GPIO to go forward (different for right and left motors)
        
        Returns
        ---
        distance : float
            Distance in meters travelled by the motor with wheel attached
        """
        self.pi = pi
        self.enc = enc
        self.en = en
        self.in1 = in1
        self.in2 = in2
        self.odo = 0 # odometer
        self.for_dir = for_dir

        self.direction = 1
        self.set_direction(1)

        # initialise encoder pin
        pi.set_mode(enc, pigpio.INPUT)
        pi.set_pull_up_down(enc, pigpio.PUD_UP)
        pi.callback(enc, pigpio.RISING_EDGE, self._pulse)

        # Direction
        pi.set_mode(in1, pigpio.OUTPUT)
        pi.set_mode(in2, pigpio.OUTPUT)
        pi.write(in1, 0)
        pi.write(in2, 1)

        # Driving motors with PWM @ 20 kHz
        pi.hardware_PWM(en, 20000, 0)
        
        # IR Sensor in ball collection mechanism
        pi.set_mode(IR__BALLDETECT_IN, pigpio.INPUT)
    
    def _pulse(self, gpio, level, tick):
        """
        Handle encoder sample
        """
        self.odo += self.direction
    
    def stop(self):
        """
        Stop motor
        """
        self.pi.set_PWM_dutycycle(self.en, 0)
    
    def set_direction(self, direction):
        """
        Parameters
        ---
        direction : int
            1 for forward, -1 for reverse
        """
        assert direction in [-1, 1], "Direction must be -1 (backwards) or 1 (forwards)"
        self.direction = direction
        if direction == 1:
            self.pi.write(self.in1, self.for_dir[0])
            self.pi.write(self.in2, self.for_dir[1])
        elif direction == -1:
            self.pi.write(self.in1, self.for_dir[1])
            self.pi.write(self.in2, self.for_dir[0])

    def drive(self, duty=0.3):
        """
        Parameters
        ---
        duty : float
            Duty cycle for PWM that drives motor. Must be in range 0.0 - 1.0
            Found that if duty < 0.25 the motor does not move
        """
        assert 0.0 <= duty <= 1.0, f'Duty cycle must be within range 0.0 - 1.0, was given {duty:.3f}'
        duty = round(duty * 255)
        self.pi.set_PWM_dutycycle(self.en, duty)

class DiffDrive:

    def __init__(self, pi):
        # initialise motors
        self.pi = pi
        self.motor_left = DCMotor(pi, LEFT_ENC_A, LEFT_EN, LEFT_IN1, LEFT_IN2, [0, 1])
        self.motor_right = DCMotor(pi, RIGHT_ENC_A, RIGHT_EN, RIGHT_IN1, RIGHT_IN2, [1, 0])

    def PID_speed_control(self, enc_l_delta, enc_r_delta, speed = 300, stop_for_ball = False, kp = 0.0003, kd = 0.000005, ki = 0.000001):
        """
        Control motors with simple PID contols for velocity
        
        Parameters
        ---
        enc_l_delta : int
            Achieve this change in encoder count on the left motor (can be negative)
        enc_r_delta : int
            Achieve this change in encoder count on the right motor (can be negative)
        speed : int
            Desired speed in encoder counts per second
        kp, kd, ki : float
            PID parameters
        
        Returns
        ---
        l_delta : int
            Resulting left encoder delta
        r_delta : int
            Resulting right encoder delta
        """

        if enc_l_delta == 0 and enc_r_delta == 0:
            return 0, 0
        
        enc_l_init = self.motor_left.odo
        enc_r_init = self.motor_right.odo

        enc_l_final = enc_l_init + enc_l_delta

        # sample/timeout parameters
        timeout = abs(enc_l_delta) / speed + 1.5 # seconds
        sample_time = 30e-3 # seconds
        max_count = round(timeout/sample_time)
        sample_count = 0
        goal_enc_per_sample = round(sample_time * speed) # desired encoder counts per sample

        # Drive motors
        dutyL = 0.25 # initial duty cycle (modified by PID control to achieve desired speed)
        dutyR = 0.25
        eL_prev_error = 0
        eR_prev_error = 0
        eL_total = 0
        eR_total = 0

        # Set direction
        self.motor_left.set_direction(-1 if enc_l_delta < 0 else 1)
        self.motor_right.set_direction(-1 if enc_r_delta < 0 else 1)

        # print("Goal encoders per sample: ", goal_enc_per_sample)

        sgn = self.motor_left.odo > enc_l_final
        zero_flag = 0

        # Wait for left encoder count to reach desired value or timeout
        while sgn == (self.motor_left.odo > enc_l_final):
            
            if stop_for_ball and self.pi.read(IR__BALLDETECT_IN) == 0:
                print("Ball Detected! Stopping")
                break

            start_sample_l = self.motor_left.odo
            start_sample_r = self.motor_right.odo

            self.motor_left.drive(dutyL)
            self.motor_right.drive(dutyR)

            time.sleep(sample_time)
            
            enc_l_speed = abs(self.motor_left.odo - start_sample_l)
            enc_r_speed = abs(self.motor_right.odo - start_sample_r)

            if enc_l_speed == 0 and enc_r_speed == 0:
                # encoders may be hung
                # TODO: reset the encoder pins
                zero_flag += 1
                # print(f"Duty Cycle: {dutyL:.2f} {dutyR:.2f}")
                # print("Encoders may be hung or specified speed may be too low.")

            eL = goal_enc_per_sample - enc_l_speed
            eR = goal_enc_per_sample - enc_r_speed

            # Apply PID adjustment to control speed (duty cycle)
            dutyL += (eL * kp) + (eL_prev_error * kd) + (eL_total * ki)
            dutyR += (eR * kp) + (eR_prev_error * kd) + (eR_total * ki)

            dutyL = max(min(dutyL, 1.0), 0.0)
            dutyR = max(min(dutyR, 1.0), 0.0)

            eL_prev_error = eL
            eR_prev_error = eR

            eL_total += eL
            eR_total += eR

            sample_count += 1
            if sample_count == max_count:
                print("Timeout reached. Terminating drive")
                break
            if zero_flag > 50:
                print("Encoders not counting. Terminating drive")
                break

            # Debugging statements
            # print("Odo: ", self.motor_left.odo,"Goal: ", enc_l_final)
            # print("Duty Cycles (L, R):", dutyL, dutyR)
            # print("Counts per sample (L, R): ", enc_l_speed, enc_r_speed)


        # turn off PWM
        # Reverse motor directions
        self.motor_left.set_direction(1 if enc_l_delta < 0 else -1)
        self.motor_right.set_direction(1 if enc_r_delta < 0 else -1)
        stop_time = 0.5 # max stopping time
        stop_count = stop_time / sample_count
        sample_count = 0
        while enc_l_speed > 0 and enc_r_speed > 0:
            if sample_count >= stop_count:
                print("Stopped after time")
                break
            # Stopping force
            start_sample_l = self.motor_left.odo
            start_sample_r = self.motor_right.odo

            self.motor_left.drive(dutyL)
            self.motor_right.drive(dutyR)

            time.sleep(sample_time)

            enc_l_speed = abs(self.motor_left.odo - start_sample_l)
            enc_r_speed = abs(self.motor_right.odo - start_sample_r)

            dutyL = 0.8 # kp * enc_l_speed * 5
            dutyR = 0.8 # kp * enc_r_speed * 5
            sample_count += 1

        self.motor_left.stop()
        self.motor_right.stop()

        time.sleep(0.5) # wait for motors to stop

        return (
            self.motor_left.odo - enc_l_init, # resulting left encoder delta
            self.motor_right.odo - enc_r_init # resulting right encoder delta
            )

    def translate(self, disp, speed, stop_for_ball=False):
        """
        Move the robot in a straight line.
        
        Parameters
        ---
        disp : float
            Displacement in meters (can be negative)
        speed : float
            Desired speed in m/s
        
        Returns
        ---
        l_delta : int
            Resulting left wheel movement
        r_delta : int
            Resulting right wheel movement
        """
        m_to_enc = 3380 # multiplier to convert meters to encoder count
        enc_dist = disp * m_to_enc

        offset = speed ** 2 * 1300
        if disp < 0: offset *= -1

        if abs(enc_dist) - abs(offset) > 200:
            enc_goal = enc_dist - offset
        else:
            enc_goal = enc_dist
        
        # print(enc_dist, enc_goal, offset)

        ld, rd = self.PID_speed_control(enc_goal, enc_goal, speed * m_to_enc, stop_for_ball)

        return (ld / m_to_enc, rd / m_to_enc)
    
    def rotate(self, angle, speed):
        """
        Rotate the robot on the spot
        
        Parameters
        ---
        angle : float
            Anti-clockwise rotation in degrees (can be negative for clockwise rotation)
        speed : float
            Desired speed in deg/s

        Returns
        ---
        l_delta : int
            Resulting left wheel movement
        r_delta : int
            Resulting right wheel movement
        """
        if angle == 0: return (0, 0)
        deg_to_enc = 8.2 # multiplier to convert degrees to encoder count
        offset = speed * 1 if angle > 30 else 10
        if angle > 0:
            ang_enc = angle * deg_to_enc - offset
        else:
            ang_enc = angle * deg_to_enc + offset
        ld, rd =self.PID_speed_control(ang_enc, -ang_enc, speed * deg_to_enc)
        
        return (ld / deg_to_enc, rd / deg_to_enc)


class TestDiffDrive(unittest.TestCase):
    ACTIVE_TESTS = [
        # "left_motor",
        # "right_motor",
        # "rotation",
        # "translation",
        "ball_detection"
    ]

    def setUp(self):
        pi = pigpio.pi()
        self.dd = DiffDrive(pi)
    
    def motor_routine(self, motor:DCMotor):
        initcount = motor.odo

        # drive forward
        motor.set_direction(1)
        motor.drive(0.3)
        time.sleep(0.5)
        motor.stop()
        time.sleep(0.1)
        midcount = motor.odo

        # drive backwards
        motor.direction = -1
        motor.drive(0.3)
        time.sleep(0.5)
        motor.stop()
        time.sleep(0.1)
        finalcount = motor.odo

        return initcount, midcount, finalcount

    @unittest.skipIf("left_motor" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_left_motor(self):
        """
        Test Left Motor is driving and counting
        """
        print("left motor test")
        i, m, f = self.motor_routine(self.dd.motor_left)
        self.assertGreater(m, i, "Encoder count after driving forward has not increased.")
        self.assertGreater(m, f, "Encoder count after driving backwards has not decreased.")
    
    @unittest.skipIf("right_motor" not in ACTIVE_TESTS, "right_motor test skipped")
    def test_right_motor(self):
        """
        Test Left Motor is driving and counting
        """
        print("right motor test")
        i, m, f = self.motor_routine(self.dd.motor_right)
        self.assertGreater(m, i, "Encoder count after driving forward has not increased.")
        self.assertGreater(m, f, "Encoder count after driving backwards has not decreased.")
    
    @unittest.skipIf("ball_detection" not in ACTIVE_TESTS, "ball_detection test skipped")
    def test_ball_detection(self):
        """
        Make sure the motor stops for ball
        """
        print("Ball detection test: Trigger IR Sensor stop the bot!")
        left, right = self.dd.translate(1.0, 0.08, True)

        # Robot should have stopped before requested distance
        self.assertLess(left, 1.0, "Robot should have stopped before requested distance")
        self.assertLess(right, 1.0, "Robot should have stopped before requested distance")
    
    @unittest.skipIf("rotation" not in ACTIVE_TESTS, "rotation test skipped")
    def test_rotation(self):
        ANGLE = 90
        SPEED = 30

        print(f"rotate {ANGLE}deg anticlockwise")
        left1, right1 = self.dd.rotate(ANGLE, SPEED)
        print(f'Motor left drove: {left1:.2f} deg')
        print(f'Motor right drove: {right1:.2f} deg')

        time.sleep(0.5)

        print(f"rotate {ANGLE}deg clockwise")
        left2, right2 = self.dd.rotate(-ANGLE, SPEED)
        print(f'Motor left drove: {left2:.2f} deg')
        print(f'Motor right drove: {right2:.2f} deg')

        self.assertLess(abs(left1 - ANGLE), 10, "Left motor inaccurate anticlockwise")
        self.assertLess(abs(right1 + ANGLE), 10, "Right motor inaccurate anticlockwise")
        self.assertLess(abs(left2 + ANGLE), 10, "Left motor inaccurate clockwise")
        self.assertLess(abs(right2 - ANGLE), 10, "Right motor inaccurate clockwise")
    
    @unittest.skipIf("translation" not in ACTIVE_TESTS, "translation test skipped")
    def test_translation(self):
        DISTANCE = 0.2
        SPEED = 0.1

        print(f"Driving forward {DISTANCE} m")
        left, right = self.dd.translate(DISTANCE, SPEED)
        print(f'Motor left drove: {left:.5f} m')
        print(f'Motor right drove: {right:.5f} m')
        self.assertLess(abs(left - DISTANCE), 0.1, "Left motor inaccurate forward")
        self.assertLess(abs(right - DISTANCE), 0.1, "Right motor inaccurate forward")
        time.sleep(0.5)

        # drive motors for 0.5m reverse
        print(f"Driving backwards {DISTANCE} m")
        left, right = self.dd.translate(-DISTANCE, SPEED)
        print(f'Motor left drove: {left:.5f} m')
        print(f'Motor right drove: {right:.5f} m')
        self.assertLess(abs(left + DISTANCE), 0.1, "Left motor inaccurate reverse")
        self.assertLess(abs(right + DISTANCE), 0.1, "Right motor inaccurate reverse")
        time.sleep(0.5)


if __name__ == "__main__":
    time.sleep(1)
    unittest.main()
