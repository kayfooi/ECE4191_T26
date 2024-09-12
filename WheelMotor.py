#!/usr/bin/env python
import pigpio
import time

# PINOUT (aligns with RPi wiring spreadsheet on drive)
LEFT_ENC_A = 2
LEFT_EN = 13
LEFT_IN1 = 19
LEFT_IN2 = 26

RIGHT_ENC_A = 3
RIGHT_EN = 12
RIGHT_IN1 = 16
RIGHT_IN2 = 20

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

        # initialise encoder pin
        pi.set_mode(enc, pigpio.INPUT)
        pi.set_pull_up_down(enc, pigpio.PUD_UP)
        pi.callback(enc, pigpio.RISING_EDGE, self._pulse)

        # Direction
        pi.set_mode(in1, pigpio.OUTPUT)
        pi.set_mode(in2, pigpio.OUTPUT)
        pi.write(in1, 0)
        pi.write(in2, 1)

        self.direction = 1
        self.set_direction(1)

        # Driving motors with PWM @ 20 kHz
        pi.hardware_PWM(en, 20000, 0)
    
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
            pi.write(self.in1, self.for_dir[0])
            pi.write(self.in2, self.for_dir[1])
        elif direction == -1:
            pi.write(self.in1, self.for_dir[1])
            pi.write(self.in2, self.for_dir[0])

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
        self.motor_left = DCMotor(pi, LEFT_ENC_A, LEFT_EN, LEFT_IN1, LEFT_IN2, [0, 1])
        self.motor_right = DCMotor(pi, RIGHT_ENC_A, RIGHT_EN, RIGHT_IN1, RIGHT_IN2, [1, 0])

    def test_drive(self):
        self.motor_left.stop()
        self.motor_right.stop()
        time.sleep(0.1)

        eL, eR = self.motor_left.odo, self.motor_right.odo

        self.motor_left.set_direction(1)
        self.motor_right.set_direction(1)

        self.motor_left.drive(0.3)
        self.motor_right.drive(0.3)

        time.sleep(0.5)
        print(f"Drove forward for 0.5 seconds: {self.motor_left.odo - eL}, {self.motor_right.odo - eR}")
        
        self.motor_left.stop()
        self.motor_right.stop()
        time.sleep(0.1)
        
        self.motor_left.set_direction(-1)
        self.motor_right.set_direction(-1)

        self.motor_left.drive(0.3)
        self.motor_right.drive(0.3)

        time.sleep(0.5)
        print(f"Drove backwards for 0.5 seconds: {self.motor_left.odo - eL}, {self.motor_right.odo - eR}")

        self.motor_left.stop()
        self.motor_right.stop()
        time.sleep(0.1)

    def PID_speed_control(self, enc_l_delta, enc_r_delta, speed = 300, kp = 0.0003, kd = 0.000005, ki = 0.000001):
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
        sample_time = 10e-3 # seconds
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

        print("Goal encoders per sample: ", goal_enc_per_sample)

        sgn = self.motor_left.odo > enc_l_final
        zero_flag = 0

        # Wait for left encoder count to reach desired value or timeout
        while sgn == (self.motor_left.odo > enc_l_final):
            
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
        self.motor_left.stop()
        self.motor_right.stop()

        time.sleep(0.5) # wait for motors to stop

        return (
            self.motor_left.odo - enc_l_init, # resulting left encoder delta
            self.motor_right.odo - enc_r_init # resulting right encoder delta
            )

    def translate(self, disp, speed):
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

        offset = speed ** 2 * 1500
        if disp < 0: offset *= -1

        if abs(enc_dist) - abs(offset) > 200:
            enc_goal = enc_dist - offset
        else:
            enc_goal = enc_dist
        
        # print(enc_dist, enc_goal, offset)

        ld, rd = self.PID_speed_control(enc_goal, enc_goal, speed * m_to_enc)

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
        deg_to_enc = 8.5 # multiplier to convert degrees to encoder count
        ang_enc = angle * deg_to_enc - 70
        ld, rd =self.PID_speed_control(ang_enc, -ang_enc, speed * deg_to_enc)
        
        return (ld / deg_to_enc, rd / deg_to_enc)

if __name__ == "__main__":
    pi = pigpio.pi()
    diff_drive = DiffDrive(pi)

    # diff_drive.test_drive()

    time.sleep(1.5)

    # drive motors for forward
    left, right = diff_drive.translate(0.5, 0.4)
    print(f'Motor left drove: {left:.5f} m')
    print(f'Motor right drove: {right:.5f} m')

    time.sleep(0.5)

    # drive motors for 0.5m reverse
    left, right = diff_drive.translate(-0.5, 0.2)
    print(f'Motor left drove: {left:.5f} m')
    print(f'Motor right drove: {right:.5f} m')

    time.sleep(0.5)

    # # rotate 90deg anticlockwise
    left, right = diff_drive.rotate(180, 30)
    print(f'Motor left drove: {left:.5f} deg')
    print(f'Motor right drove: {right:.5f} deg')

    time.sleep(0.5)

    # rotate 90deg clockwise
    left, right = diff_drive.rotate(-180, 30)
    print(f'Motor left drove: {left:.5f} deg')
    print(f'Motor right drove: {right:.4f} deg')
