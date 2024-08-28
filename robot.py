import numpy as np
import unittest
import cv2
import serial
import time
from ultralytics import YOLO
import glob
import sys


class DiffDriveRobot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = init_th
        self.model = YOLO("CV/YOLOv2.pt")

        # Connection to Arduino board
        ports = []
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/ttyACM*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')
        # print(ports)
        # Wake camera up
        self.cap = cv2.VideoCapture(-1, cv2.CAP_V4L)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        time.sleep(0.5)
        # self.cap = None

        try:
            self.ser = serial.Serial(ports[0], 9600, timeout=1.0)
            # important sleep to allow serial communication and camera to initialise
            time.sleep(1.5)
        except serial.SerialException as e:
            print("Could not connect to arduino.")
            self.ser = None
            print(e)
        
        # Homography that transforms image coordinates to world coordinates
        # As of 28th Aug 9AM
        self._H = np.array([
            [-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
            [-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
            [-0.0029850356852013345, -0.04116105685090471, 1.0],
        ])
    
    def __del__(self):
        # body of destructor
        self.cap.release()
        
    def capture(self):
        ret, frame = self.cap.read()
        return frame

    def _arduino_instruction(self, instruction):
        """
        Parameters
        ---
        instruction : str
            Needs to be of the form 'R_90.0' to rotate 90.0 degrees clockwise for example, or 'T_1.29' to drive forward 1.29 m for example.
            Values can be negative to go in the other direction.
        """
        
        # Send instruction to Arduino
        print(f"Sending instruction: {instruction}")
        self.ser.write(instruction.encode("utf-8"))

        # time.sleep(2)
        # Wait for instruction to finish (buggy)
        max_time = 10.0 # timeout after this amount of seconds
        sleep_time = 0.05 # wait this amount of seconds between checks
        checks = 0
        while checks < max_time / sleep_time:
            try:
                in_waiting = self.ser.in_waiting
            except OSError:
                in_waiting = 0
            if in_waiting > 0:
                line = self.ser.readline().decode('utf-8').rstrip()
                try:
                    part = line.split("_")
                    value = float(part[1]) # amount the robot actually did
                    print(f"Successfully complete {line} after {sleep_time*checks:.2f} sec")
                    return value
                except(ValueError, IndexError) as e:
                    print(e)
                    print(f"Received: '{line}' from Arduino, expecting '{instruction} complete'")
            time.sleep(sleep_time)
            checks += 1
        print("Timed out")
        return None

    def coordTran(self, x, y):
        instruction = f"P {x} {y}"
        self._arduino_instruction(instruction)


    def rotate(self, angle, velocity=10.0):
        """
        Rotate the robot on the spot. (anti-clockwise direction is positive)  
        
        Parameters
        ----
        angle: Amount of rotation in degrees, can be negative. Should be in range (-180, +180)
        veclocity: rotational velocity in degrees per second
        """

        # TODO: implement velocity
        # Send rotation instruction
        if angle < 0: 
            sign =  'p' 
        else:
            sign = 'm'
        instruction = f"R_{abs(angle):.3f}_{sign}"
        rotation = self._arduino_instruction(instruction)
        
        if rotation is not None:
            # Stop the robot if big error for now
            assert abs(rotation - angle) < 5.0, "Rotation error too large. Aborting!"

            # Update State on PI
            self.th += rotation

    
    def translate(self, displacement, velocity=1.0):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        velocity: velocity in meters per second
        """
        # TODO: implement velocity
        
        # Send translation instruction
        if displacement < 0: 
            sign =  'm' 
        else:
            sign = 'p'
        instruction = f"T_{abs(displacement*1000):.3f}_{sign}"
        print(instruction)
        actual_displacement = self._arduino_instruction(instruction)
        
        # Stop the robot if big error for now
        assert abs(actual_displacement - displacement) < 0.1, "Translation error too large. Aborting!"

        # Update state on PI
        th_rad = np.radians(self.th)
        self.pos += np.array([
            actual_displacement * np.cos(th_rad),
            actual_displacement * np.sin(th_rad)
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
    
    def apply_YOLO_model(self, img):
        # Predict with the model
        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (1280, 720))
        # scale = np.array(image.shape[:2]) / np.array((640,  640))

        results = self.model(image, conf=0.50, verbose=False)  # predict on an image

        points = []
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for b in boxes:
                d = b.xywh[0]
                # points.append([d[])
                points.append([d[0], d[1] + d[3]/2])
            # result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk

        # TODO: choose the best points detected
        points = np.array(points)
        print(points)
        if len(points) > 0:
            best_point = np.argmin(np.abs(points[:, 0] - image.shape[0]))
            return (points).astype(int)
        else:
            return None

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
            img = self.capture()
        

        ball_loc = self.apply_YOLO_model(img)
        print(ball_loc)
        if ball_loc is not None:
            # apply homography
            relative_pos = self.image_to_world(ball_loc)
            # translate into world coordinates
            return self._getRotationMatrix() @ relative_pos[0][:2] + self.pos
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
        r = (np.degrees(np.arctan2(delta[1], delta[0])) - self.th - 90)
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
    
    # Convert image coords to world coords with homography H
    def image_to_world(self, image_coords):
        H = [
            [-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
            [-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
            [-0.0029850356852013345, -0.04116105685090471, 1.0],
            ]
        # Convert to homogeneous coordinates
        homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
        
        # Apply the homography
        world_coords = np.dot(H, homogeneous_coords.T).T
        
        # Convert back from homogeneous coordinates
        world_coords = world_coords[:, :2] / world_coords[:, 2:]
        
        return world_coords
    



class TestBot(unittest.TestCase):
    def test_setup(self):
        print("testing")
        self.init_pos = np.array([0., 0.])
        self.init_th = 0
        self.bot = DiffDriveRobot(self.init_pos.copy(), 0)
        

        ball = self.bot.detect_ball()

        print(ball)

        # self.bot.th = 0
        # rotation = -40
        # self.bot.rotate(rotation)
        # self.assertEqual(self.bot.th, rotation)
        # self.bot.translate(1.5)
        # np.testing.assert_allclose(self.bot.pos, np.array([1.5, 0.]))
        # self.pos = self.init_pos.copy()
        # self.bot.th = 0
        # rotation = 45
        # self.bot.rotate(rotation)
        # distance = 0.6
        # self.bot.translate(distance)
        # np.testing.assert_allclose(self.bot.pos, np.array([1., 1.]))

        # self.assertAlmostEqual(self.bot.calculateDistance(self.init_pos), distance)
        # self.assertAlmostEqual(abs(self.bot.calculateRotationDelta(self.init_pos)), 180)
        

    # def test_ball_detection(self):
    #     # TODO: load in test images, assert output is sensible from bot.detect_ball
    #     ...
    
    # def test_rotation_matrix(self):
    #     point = np.sqrt(np.array([2, 2]))
    #     self.bot.th = 45
    #     # Rotates points from camera coordinates to world coordinates
    #     R = self.bot._getRotationMatrix()
    #     np.testing.assert_allclose(R @ point, np.array([0., 2.]), atol=1e-7)
    #     # self.bot.
        

if __name__ == '__main__':
    unittest.main()

