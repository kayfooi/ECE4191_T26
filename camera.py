import numpy as np
import cv2
import time
import unittest
import ncnn # faster/lighter than ultralytics and torch

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
            self.cap = cv2.VideoCapture(-1, cv2.CAP_V4L) # for the pi
            # self.cap = cv2.VideoCapture(0) # this may work if you are on a laptop
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Default buffer size is 4, changes to brightness might not be observed until 4 frames are read
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # NOTE: I don't think our camera has exposure settings
            # set the brightness instead (in range -255 to +255)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -50)
            time.sleep(0.1)
        else:
            self.cap = None

        # Load ball detection model
        self.model = ncnn.Net()
        self.model.load_param("./CV/YOLO_ball_box_detection_ncnn_model/model.ncnn.param")
        self.model.load_model("./CV/YOLO_ball_box_detection_ncnn_model/model.ncnn.bin")

        # Homography that transforms image coordinates to world coordinates
        self._H = np.array([
            [-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
            [-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
            [-0.0029850356852013345, -0.04116105685090471, 1.0],
        ])
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def capture(self):
        """
        Capture frame from camera
        """
        if self.cap is not None:
            ret, img = self.cap.read()
            if ret:
                return img
        else:
             print("Image not captured")
             return None
    
    def apply_YOLO_model(self, image, visualise=False):
        """
        Apply YOLO model to get locations of balls
        """
        # Preprocessing
        height, width, _ = image.shape
        img_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = image.copy()
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (640, 640)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        stride = 32
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        im = np.stack([img])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        in0 = np.ascontiguousarray(im/255).astype(np.float32)  # contiguous

        # Prepare input
        mat_in = ncnn.Mat(in0[0])

        # Run inference
        extractor = self.model.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        out = np.array(mat_out)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []

        # plt.hist(out[6, :])
        # plt.show()
        # max = np.argmax(out[6, :])
        # print(out[:, max])

        for i in range(mat_out.w):
            detection = out[:, i]
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                # Object detected
                xywh = detection[:4] / 640
                y = detection[1]
                y = (y - top) / (640 - top - bottom)
                
                center_x = int(xywh[0] * width)
                center_y = int(y * height)
                w = int(xywh[2] * width)
                h = int(xywh[3] * width)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold=0.5,nms_threshold=0.1,top_k=5)
        classes = ['box', 'legs', 'tennis-ball']
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        if visualise:
            vis_image = image.copy()

        ball_coords = []
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                
                if label == 'tennis-ball':
                    ball_coords.append([x + w/2, y + h])
                
                if visualise:
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_image, label, (x, y + 30), font, 2, color, 3)
        
        if visualise:
            out_file = 'YOLO_result.jpg'
            cv2.imwrite(out_file, vis_image)
            print(f"Labelled image saved to {out_file}")
        ball_coords = np.array(ball_coords)
        return ball_coords.astype(int)

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
    



class TestCamera(unittest.TestCase):
    """
    Test Camera specific functions
    """
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    @unittest.skip("skipped")
    def test_image_to_world(self):
        self.cam = Camera(False) # no camera
        img_c = np.array([[100, 100]])
        self.cam.image_to_world(img_c)

    # @unittest.skip("skipped")
    def test_YOLO_model(self):
        # Open image(s) and pass to model
        self.cam = Camera(False) # no camera
        img = cv2.imread('CV/test_imgs/test_images/testing0001.jpg')
        res = self.cam.apply_YOLO_model(img, True)
        print("Ball detected at:", res)

    def test_capture(self):
        self.cam = Camera(True)
        self.startTime = time.time()
        img = self.cam.capture()
        if img is not None:
            cv2.imwrite("test_results/capture_result.jpg", img)
        self.assertTrue(img is not None, "Camera did not capture anything")
        
    


    def test_ball_detection(self):
        # Open image(s) and pass to function
        ...
    
    def test_line_detection(self):
        # Open images and pass to function
        ...

    def test_box_detection(self):
        # Open images and pass to function
        ...

def _capture_loop():
    cam = Camera(True)
    i = 0
    
    while True:
        s = time.time()
        img = cam.capture()
        if img is not None:
            cv2.imwrite(f"./test_results/result{i}.jpg", img)
        e = time.time()
        print(f"Frame {i}: {(e-s)*1e3:.2f} msec")
        inp = input("x to escape, any other key to capture: ")
        try:
            adj = int(inp)
            cam.cap.set(cv2.CAP_PROP_BRIGHTNESS, adj)
        except ValueError:
            if inp == "x":
                break
        i += 1

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCamera)
    unittest.TextTestRunner(verbosity=0).run(suite)