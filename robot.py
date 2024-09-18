import numpy as np
import unittest
import cv2
import time
from camera import Camera
import io



try:
    import pigpio
    from WheelMotor import DiffDrive
    on_pi = True
except ImportError:
    print("pigpiod library not found. Continuing tests with no pi")
    on_pi = False

if on_pi:
    from rPi_sensor import laser
    tof = laser.PiicoDev_VL53L1X
else:
    tof = None

class Robot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = init_th
        if on_pi:
            self.pi = pigpio.pi()
            self.dd = DiffDrive(self.pi)
        else:
            self.pi = None # 
            self.dd = None # 
        
        self.camera = Camera(False)
        
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
            rotation_left, rotation_right = self.dd.rotate(angle, speed)
            self.th += (rotation_left + rotation_right) / 2
        else:
            self.th += angle + np.random.random() * 2 - 1
        

    
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
            disp_left, disp_right = self.dd.translate(displacement, speed)
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
            avg_disp = displacement + 0.05 * np.random.random() - 0.025
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
    
    def _getRotationMatrix(self):
        """
        Get 2D rotation matrix from bot orientation
        """
        th_rad = np.radians(self.th)
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
        relative_pos = self.camera.detectBalls(img)
        ball_locs = relative_pos @ self._getRotationMatrix().T + self.pos
        return ball_locs



class TestBot(unittest.TestCase):
    """
    Test Robot specific functions. See `camera.py` for camera testing
    """
    def setUp(self):
        self.bot = Robot()

    def test_rotation_translation(self):
        init_pos = self.bot.pos.copy()

        rotation = 45
        self.bot.rotate(rotation)
        distance = np.sqrt(0.3**2 + 0.3**2)
        self.bot.translate(distance)
        
        np.testing.assert_allclose(self.bot.pos, np.array([0.3, 0.3]), atol=5e-2)
        self.assertLess(abs(self.bot.calculateDistance(init_pos)-distance), 0.1)
        self.assertLess(abs(abs(self.bot.calculateRotationDelta(init_pos))-180), 0.1)
    
    def test_travel_to(self):
        target = np.array([3, 4])
        self.bot.travelTo(target)
        np.testing.assert_allclose(self.bot.pos, target, atol=1e-1)

    def test_rotation_delta(self):
        a = np.vstack(np.radians(np.arange(0, 361, 45)))
        points = np.hstack((np.cos(a), np.sin(a)))
        expected = [0, 45, 90, 135, 180, -135, -90, -45, 0]

        for (i, angle) in enumerate(expected):
            with self.subTest(f"Test rotation delta to point: {np.round(points[i], 3)})"):
                self.assertAlmostEqual(self.bot.calculateRotationDelta(points[i]), angle)
    
    def test_rotation_matrix(self):
        point = np.sqrt(np.array([2, 2]))
        self.bot.th = 45
        # Rotates points from camera coordinates to world coordinates
        R = self.bot._getRotationMatrix()
        np.testing.assert_allclose(R @ point, np.array([0., 2.]), atol=1e-7)
    
    def test_ball_detection(self):
        # Open image(s) and pass to function
        img = cv2.imread('CV/test_imgs/test_images/testing0001.jpg')
        res = self.bot.detectBalls(img)
        np.testing.assert_allclose(res, np.array([
            [0.071, 0.91]
        ]), atol=0.005)

if __name__ == '__main__':
    unittest.main()