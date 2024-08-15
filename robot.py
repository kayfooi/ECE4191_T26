import numpy as np
import CV.camera as camera

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
        # TODO: implement rotation
        ...

    
    def translate(self, displacement, velocity=1.0):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        velocity: velocity in meters per second
        """
        # TODO: implement moving forward/backward in a straight line
        ...
    
    def _getRotationMatrix(self):
        """
        Get 2D rotation matrix from bot orientation
        """
        th_rad = np.radians(self.th)
        return np.array([
            [np.cos(th_rad), -np.sin(th_rad)],
            [np.sin(th_rad), np.cos(th_rad)]
        ])

    def detect_ball(self):
        """
        Captures Image and detects tennis ball locations in world coordinates

        Return
        ----
        position: world coordinates (x, y) of detected ball, can be None
        """
        img = camera.capture()
        ball_loc = camera.detect_ball(img)

        if ball_loc:
            # apply homography
            relative_pos = self._H @ ball_loc
            # translate into world coordinates
            return self._getRotationMatrix() @ relative_pos + self.pos
        else:
            return None
    
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
        return np.degrees(np.atan(delta[1]/delta[0])) - self.th
    
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
    
        