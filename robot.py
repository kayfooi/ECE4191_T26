import numpy as np
import CV.camera as camera
import unittest
import cv2

class DiffDriveRobot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = init_th
        # Homography that transforms image coordinates to world coordinates
        self._H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    
    
    def rotate(self, angle, velocity=10.0):
        """
        Rotate clockwise  
        
        Parameters
        ----
        angle: Amount of rotation in degrees, can be negative. Should be in range (-180, +180)
        veclocity: rotational velocity in degrees per second
        """
        # TODO: implement rotation by interfacing with auduino
        
        # simulate perfect operation
        self.th += angle

    
    def translate(self, displacement, velocity=1.0):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        velocity: velocity in meters per second
        """
        # TODO: implement moving forward/backward in a straight line
        
        # simulate perfect operation
        th_rad = np.radians(self.th)
        self.pos += np.array([
            displacement * np.cos(th_rad),
            displacement * np.sin(th_rad)
        ])

    
    def _getRotationMatrix(self):
        """
        Get 2D rotation matrix from bot orientation
        """
        th_rad = np.radians(self.th)
        return np.array([
            [np.cos(th_rad), -np.sin(th_rad)],
            [np.sin(th_rad), np.cos(th_rad)]
        ])

    def detect_ball(self, img=None):
        """
        Captures Image and detects tennis ball locations in world coordinates

        Parameters:
        img: np array representing image

        Return
        ----
        position: world coordinates (x, y) of detected ball, can be None
        """
        if img is None:
            # capture from camera
            img = camera.capture()
            

        ball_loc = np.array(camera.detect_ball(img))

        if ball_loc is not None:
            # apply homography
            relative_pos = self._H @ np.append(ball_loc, [1])
            # translate into world coordinates
            return self._getRotationMatrix() @ relative_pos[:2] + self.pos
        else:
            return True # None (change once detection is added)
    
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
    



class TestBot(unittest.TestCase):
    def setUp(self):
        self.init_pos = np.array([0., 0.])
        self.init_th = 0
        self.bot = DiffDriveRobot(self.init_pos.copy(), 0)

    def test_translation(self):
        self.pos = self.init_pos.copy()
        self.bot.translate(1.5)
        np.testing.assert_allclose(self.bot.pos, np.array([1.5, 0.]))
    
    def test_rotation(self):
        self.bot.th = 0
        rotation = 90
        self.bot.rotate(rotation)
        self.assertEqual(self.bot.th, rotation)
    
    def test_rotation_translation(self):
        self.pos = self.init_pos.copy()
        self.bot.th = 0
        rotation = 45
        self.bot.rotate(rotation)
        distance = np.sqrt(2)
        self.bot.translate(distance)
        np.testing.assert_allclose(self.bot.pos, np.array([1., 1.]))

        self.assertAlmostEqual(self.bot.calculateDistance(self.init_pos), distance)
        self.assertAlmostEqual(abs(self.bot.calculateRotationDelta(self.init_pos)), 180)

    def test_ball_detection(self):
        # TODO: load in test images, assert output is sensible from bot.detect_ball
        ...
    
    def test_rotation_matrix(self):
        point = np.sqrt(np.array([2, 2]))
        self.bot.th = 45
        # Rotates points from camera coordinates to world coordinates
        R = self.bot._getRotationMatrix()
        np.testing.assert_allclose(R @ point, np.array([0., 2.]), atol=1e-7)
        # self.bot.
        

if __name__ == '__main__':
    unittest.main()