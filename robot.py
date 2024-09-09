import numpy as np
import unittest
import cv2
import time
from camera import Camera
from WheelMotor import DiffDrive
import pigpio

class Robot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = init_th
        self.pi = None # pigpio.pi()
        self.dd = None # DiffDrive(self.pi)
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
        rotation_left, rotation_right = self.dd.rotate(angle, speed)
    
        # Stop the robot if big error for now
        # assert abs(rotation - angle) < 5.0, "Rotation error too large. Aborting!"

        # Update State
        self.th += (rotation_left + rotation_right) / 2

    
    def translate(self, displacement, speed=0.3):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        speed: speed in meters per second
        """
        # Send translation instruction
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
        ball_locs = self._getRotationMatrix() @ relative_pos + self.pos
        return ball_locs



class TestBot(unittest.TestCase):
    """
    Test Robot specific functions. See `camera.py` for camera testing
    """
    def setUp(self):
        self.bot = Robot()

    # def test_rotation_translation(self):
    #     self.pos = self.init_pos.copy()
    #     self.bot.th = 0
    #     rotation = 45
    #     self.bot.rotate(rotation)
    #     distance = np.sqrt(2)
    #     self.bot.translate(distance)
    #     np.testing.assert_allclose(self.bot.pos, np.array([1., 1.]))

    #     self.assertAlmostEqual(self.bot.calculateDistance(self.init_pos), distance)
    #     self.assertAlmostEqual(abs(self.bot.calculateRotationDelta(self.init_pos)), 180)

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
        # self.bot.
        

if __name__ == '__main__':
    unittest.main()