import numpy as np
import unittest
import cv2
import time
from camera import Camera
import io
from datetime import datetime

try:
    import pigpio
    from WheelMotor import DiffDrive
    on_pi = True
except ImportError:
    print("pigpiod library not found. Continuing tests with no pi")
    on_pi = False

if on_pi:
    from rPi_sensor import laser
    tof = laser.PiicoDev_VL53L1X()
    from rPi_sensor.servo import Servo
else:
    tof = None

# GPIO Numbers
TIPPING_SERVO_GPIO = 7
PADDLE_SERVO_GPIO = 14
# Other GPIO is stored in WheelMotor.py

class Robot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = np.float64(init_th)
        if on_pi:
            self.pi = pigpio.pi()
            self.dd = DiffDrive(self.pi)
            self.tip_servo = Servo(self.pi, TIPPING_SERVO_GPIO) 
            self.paddle_servo = Servo(self.pi, PADDLE_SERVO_GPIO)
        else:
            self.pi = None 
            self.dd = None 
            self.tip_servo = None 
            self.paddle_servo = None
        
        self.camera = Camera(False)
    
    def is_on_pi(self):
        return on_pi
    
    # -------- CONTROL FUNCTIONS --------
    def rotate(self, angle, speed=20):
        """
        Rotate anti-clockwise  
        
        Parameters
        ----
        angle: Amount of rotation in degrees, can be negative. Should be in range (-180, +180)
        speed: rotational speed in degrees per second
        """
        # Send rotation instruction
        if on_pi:
            rotation_left, rotation_right, stop_code = self.dd.rotate(angle, speed)
            self.th += (rotation_left + rotation_right) / 2
        else:
            noise = 2 # magnitude of randomness (simulation)
            self.th += angle + np.random.random() * noise - noise/2
        

    
    def translate(self, displacement, speed=0.3):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        speed: speed in meters per second
        """
        # Send translation instruction
        if on_pi:
            disp_left, disp_right, stop_code = self.dd.translate(displacement, speed)
            avg_disp = (disp_left + disp_right )/ 2
            
            # Big Error
            if abs(avg_disp - displacement) > 0.1:
                print(f"Translation error large: {avg_disp - displacement:.3f} m")
            if abs(avg_disp - disp_left) > 0.05 or abs(avg_disp - disp_right) > 0.05:
                print(f"Large variance between left ({disp_left:.3f} m) and right ({disp_right:.3f} m) motor displacement")

            # Update position coordinates
            th_rad = np.radians(self.th)
            self.pos += np.array([
                avg_disp * np.cos(th_rad),
                avg_disp * np.sin(th_rad)
            ])
        else:
            th_rad = np.radians(self.th)
            noise = 0.05
            avg_disp = displacement + noise * np.random.random() - noise/2
            self.pos += np.array([
                avg_disp * np.cos(th_rad),
                avg_disp * np.sin(th_rad)
            ])

    
    def travelTo(self, p, rspeed=20.0, tspeed=0.3, complete=1.0):
        """
        Travel to given point, p, through rotation and translation

        Parameters
        ---
        p: ArrayLike
            2D world coordinate target location
        rspeed: float
            Rotational speed (degrees/sec)
        tspeed: float
            Straight line speed (m/sec)
        complete: float
            Total displacement to complete, ranging from (0.0 to 1.0)
        """
        rotation = self.calculateRotationDelta(p)
        disp = self.calculateDistance(p) * complete
        self.rotate(rotation)
        self.translate(disp)
    
    def collect_ball(self):
        """
        Actuates paddle mechanism to collect ball
        """
        rest_angle = 0 # TODO: check this before mounting
        collect_angle = 150 # TODO: check this before mounting
        collect_speed = 50 # degrees per second
        self.paddle_servo.set_angle(collect_angle, collect_speed)
        time.sleep(2) # allow the ball to roll off
        self.paddle_servo.set_angle(rest_angle)
        time.sleep(1)
        self.paddle_servo.stop()

        # No feedback, may need sensors to somehow detect if ball is collected
        # Hopefully it works reliably enough that we don't need this
    
    def dump_balls(self):
        """
        Actuates tipping mechanism to dump balls
        """
        # Dump
        dump_angle = 90 # TODO: check this before mounting
        dump_speed = 10 # degrees per second

        self.tip_servo.set_angle(dump_angle, dump_speed)
        time.sleep(5) # allow balls to exit
        # TODO: may have to add shaking mechanism if balls don't exit reliably

        # Return to original position
        rest_angle = 80 # TODO: check this before mounting
        return_speed = 20

        self.tip_servo.set_angle(rest_angle)
        time.sleep(0.5)
        self.tip_servo.stop()
    
    # -------- HELPER FUNCTIONS -----------
    
    def _getRotationMatrix(self):
        """
        Get 2D rotation matrix from bot orientation
        """
        th_rad = np.radians(self.th - 90)
        return np.array([
            [np.cos(th_rad), -np.sin(th_rad)],
            [np.sin(th_rad), np.cos(th_rad)]
        ])

    def calculateRotationDelta(self, p):
        """
        Parameters
        ----
        p: np.array (x, y) coordinates

        Return
        ----
        Rotation needed by bot in order to face p in degrees
        """
        delta = p - self.pos
        r = (np.degrees(np.arctan2(delta[1], delta[0])) - self.th)
        if r < -180:
            return r + 360
        if r > 180:
            return r - 360
        else:
            return r
    
    def calculateDistance(self, p):
        """
        Parameters
        ----
        p: np.array (x, y) coordinates

        Return
        ----
        Distance between bot and p
        """
        delta = p - self.pos
        return np.sqrt(np.sum(delta ** 2))
    
    def detectBalls(self, img=None):
        """
        Detects ball locations

        Paramters
        ----
        img: ArrayLike
            Optional test image
        
        Return
        ---
        balls_locs ArrayLike
            Array of detected ball locations in 2D world coordinates
        """

        # TODO: when calibrating camera again, ensure world coordinates are relative to bot facing 0 deg
        relative_pos = self.camera.detectBalls(img)
        ball_locs = relative_pos @ self._getRotationMatrix().T + self.pos
        return ball_locs

    # -------- VISUALISING FUNCTIONS --------
    def plot_bot(self, ax):
        """
        Show bot on ax
        """
        arrow_size = 0.2
        heading = np.radians(self.th)
        ax.arrow(self.pos[0], self.pos[1], 
                 arrow_size*np.cos(heading), 
                 arrow_size*np.sin(heading), 
                 color='k', width=arrow_size/3.5, label=None)
    
    def __str__(self):
        """
        Print representation of robot
        """
        time_now = datetime.now().strftime("%H:%M:%S")
        pos = f'({self.pos[0]:.3f}, {self.pos[1]:.3f})'
        return f'Robot @ {time_now}: {pos} facing {self.th:.1f} deg'




class TestBot(unittest.TestCase):
    """
    Test Robot specific functions. See `camera.py` for camera testing
    """
    ACTIVE_TESTS = [
        # "left_motor",
        # "right_motor",
        # "rotation",
        # "translation",
        # "ball_detection"
        # "collect_ball"
        "dump_balls"
    ]
    def setUp(self):
        self.bot = Robot()

    @unittest.skipIf("rotation_translation" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_translation(self):
        init_pos = self.bot.pos.copy()

        rotation = 45
        self.bot.rotate(rotation)
        distance = np.sqrt(0.3**2 + 0.3**2)
        self.bot.translate(distance)
        
        np.testing.assert_allclose(self.bot.pos, np.array([0.3, 0.3]), atol=5e-2)
        self.assertLess(abs(self.bot.calculateDistance(init_pos)-distance), 0.1)
        self.assertLess(abs(abs(self.bot.calculateRotationDelta(init_pos))-180), 0.1)
        print(self.bot)
    
    @unittest.skipIf("travel_to" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_travel_to(self):
        target = np.array([3, 4])
        self.bot.travelTo(target)
        np.testing.assert_allclose(self.bot.pos, target, atol=1e-1)

    @unittest.skipIf("rotation_delta" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_delta(self):
        a = np.vstack(np.radians(np.arange(0, 361, 45)))
        points = np.hstack((np.cos(a), np.sin(a)))
        expected = [0, 45, 90, 135, 180, -135, -90, -45, 0]

        for (i, angle) in enumerate(expected):
            with self.subTest(f"Test rotation delta to point: {np.round(points[i], 3)})"):
                self.assertAlmostEqual(self.bot.calculateRotationDelta(points[i]), angle)
    
    @unittest.skipIf("rotation_matrix" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_matrix(self):
        point = np.sqrt(np.array([2, 2]))
        self.bot.th = 45
        # Rotates points from camera coordinates to world coordinates
        R = self.bot._getRotationMatrix()
        np.testing.assert_allclose(R @ point, np.array([2., 0.]), atol=1e-7)
    
    @unittest.skipIf("ball_detection" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_ball_detection(self):
        # Open image(s) and pass to function
        img = cv2.imread('CV/test_imgs/test_images/testing0001.jpg')
        res = self.bot.detectBalls(img)
        print(res)
        # np.testing.assert_allclose(res, np.array([
        #     [0.071, 0.91]
        # ]), atol=0.005)

    def test_laser(self):

        for i in range(0, 15):
            dist = tof.read() # distance in mm
            print( i + ": " + str(dist) + "mm")
            time.sleep(0.1)   
    
    @unittest.skipIf(not on_pi, "Pi not connected")
    def test_dump_balls(self):
        print("Dumping balls...")
        self.bot.dump_balls()
    
    @unittest.skipIf("collect_ball" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_collect_ball(self):
        print("Collecting ball...")
        self.bot.collect_ball()

if __name__ == '__main__':
    unittest.main()