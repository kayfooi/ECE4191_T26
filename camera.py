import numpy as np
import cv2
import time
import unittest
from ultralytics import YOLO

class Camera:
    """
    Handles all things related to computer vision

    Parameters
    ---
    open_cam: bool
        Connect to webcam
    """
    def __init__(self, open_cam=True):
        # Initialise USB Camera
        if open_cam:
            self.cap = cv2.VideoCapture(-1, cv2.CAP_V4L)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            # TODO: set autoexposure settings
            time.sleep(0.5)
        else:
            self.cap = None

        # Load ball detection model Model
        self.model = YOLO("CV/YOLOv2.pt")

        # Homography that transforms image coordinates to world coordinates
        self._H = np.array([
            [-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
            [-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
            [-0.0029850356852013345, -0.04116105685090471, 1.0],
        ])
    
    def __del__(self):
        self.cap.release()

    def capture(self):
        """
        Capture frame from camera
        """
        ret, img = self.cap.read()
        if ret:
             return img
        else:
             print("Image not captured")
             return None
    
    def apply_YOLO_model(self, img):
        # Predict with the model
        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
            # result.save(filename="result.jpg")  # save to disk

        # TODO: Filter out invlalid detections based on box size
        points = np.array(points)
        return points.astype(int)

    def detectBalls(self, img=None):
        """
        Captures Image and detects tennis ball locations in world coordinates relative to camera

        Parameters
        ---
        img: *optional ArrayLike
            Test image

        Return
        ----
        position: [(float, float)]
            world coordinates (x, y) of detected balls
        """
        if img is None:
            # capture from camera
            img = self.capture()
        
        ball_locs = self.apply_YOLO_model(img)

        # translate into world coordinates
        return self.image_to_world(ball_locs)
        

    def image_to_world(self, image_coords):
        """
        Converts image coordinates to world coordinates (relative to camera)
        """
        # Convert to homogeneous coordinates
        homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
        
        # Apply the homography
        world_coords = np.dot(self._H, homogeneous_coords.T).T
        
        # Convert back from homogeneous coordinates
        world_coords = world_coords[:, :2] / world_coords[:, 2:]
        
        return world_coords

    def detect_box(self, img=None):
        """
        TODO: Detect cardboard box location
        
        Parameters
        ---
        img: *optional ArrayLike
            Test image

        Returns
        ---
        box_loc: ArrayLike | None
            Something like: world coordinate of box corner in our quadrant?
        """
        if img is None:
            # capture from camera
            img = self.capture()
        
        return None
    
    def detect_lines(self, img=None):
        """
        TODO: Detect tennis court lines

        Parameters
        ---
        img: *optional ArrayLike
            Test image
        
        Return
        ---
        lines: ArrayLike
            Some representation of detected lines, maybe in world coordinates or just image coordinates idk
        """
        if img is None:
            # capture from camera
            img = self.capture()
        
        return None
    



class TestBot(unittest.TestCase):
    """
    Test Camera specific functions
    """
    def setUp(self):
        self.cam = Camera(False)

    def test_image_to_world(self):
        img_c = np.array([100, 100])
        self.cam.image_to_world(img_c)

    def test_YOLO_model(self):
        # Open image(s) and pass to model
        ...

    def test_ball_detection(self):
        # Open image(s) and pass to function
        ...
    
    def test_line_detection(self):
        # Open images and pass to function
        ...

    def test_box_detection(self):
        # Open images and pass to function
        ...
        

if __name__ == '__main__':
    unittest.main()