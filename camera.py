import time
s = time.time()
import numpy as np
import cv2
import unittest
import ncnn # faster/lighter than ultralytics and torch
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
e = time.time()

print(f"Camera modules import: {e-s:.3f} sec")

# Comparing the two edges of a white line gradients should be similar
GRADIENT_SIMILARITY_THRESH = 0.03
INTERCEPT_DIFF_THRESH = 55
RESULT_OUT_PREFIX = f'test_results/{int(time.time())}' # save results here for debugging
LINE_CONF_THRESHOLD = 900 # lower if we want more sensitivity
IMG_WIDTH = 640
IMG_HEIGHT = 480

os.mkdir(RESULT_OUT_PREFIX) 
# Main camera interface
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Default buffer size is 4, changes to brightness might not be observed until 4 frames are read
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # NOTE: I don't think our camera has exposure settings
            # set the brightness instead (in range -255 to +255)
            # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -50)
            time.sleep(0.1)
        else:
            self.cap = None

        # Load ball detection model
        self.model = ncnn.Net()

        # Balls and boxes
        self.model.load_param("./CV/YOLO_ball_box_detection_ncnn_model/model.ncnn.param")
        self.model.load_model("./CV/YOLO_ball_box_detection_ncnn_model/model.ncnn.bin")

        # Homography that transforms image coordinates to world coordinates
        # self._H = np.array([
        #     [-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
        #     [-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
        #     [-0.0029850356852013345, -0.04116105685090471, 1.0],
        # ])
        # For HP Webcam (8th October)
        self._H = np.array([
            [ 2.51805091e-04, -1.83088698e-02,  2.37992253e+01],
            [-4.44788004e-02,  6.59539865e-04,  1.40098008e+01],
            [-1.40510443e-03,  1.14570475e-01,  1.00000000e+00]
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
            if confidence > 0.5:
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold=0.01,nms_threshold=0.01,top_k=10)
        classes = ['box', 'tennis-ball']
        # classes = ['tennis-ball']
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        if visualise:
            vis_image = image.copy()

        results = {}
        for c in classes:
            results[c] = []

        # print(boxes)
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                
                results[label].append(np.array([x + w/2, y + h]).astype(int))
                
                if visualise:
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_image, f"{label} {confidences[i]:.2f}", (x, y + 30), font, 1.25, color, 2)
        
        if visualise:
            return results, vis_image
        
        # coords = np.array(coords)
        return results

    def detectBalls(self, img=None, visualise=False):
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
        
        results = self.apply_YOLO_model(img)
        ball_locs = np.array(results['tennis-ball'])

        # Detect if there is a line befor the ball
        # This may be better to apply to a closest ball or if there is uncertainty a ball lies within our quadrant
        valid_ball_locs = []
        if visualise:
            result_img = img.copy()
        for target in ball_locs:
            line_pair, confidence = detect_white_line(img, target, 12, visualise=False)
            if visualise:
                result_img = visualize_results(result_img, target, confidence, line_pair)
            if confidence < LINE_CONF_THRESHOLD: # confidence of a line between ball and bot
                valid_ball_locs.append(target) # only valid if no line is there
        
        valid_ball_locs = np.array(valid_ball_locs)

        # translate into world coordinates
        if visualise:
            return self.image_to_world(valid_ball_locs), result_img
        else:
            return self.image_to_world(valid_ball_locs)
        

    def image_to_world(self, image_coords):
        """
        Converts image coordinates to world coordinates (relative to camera)
        """
        if len(image_coords) == 0:
            return []

        # Convert to homogeneous coordinates
        homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
        
        # Apply the homography
        world_coords = np.dot(self._H, homogeneous_coords.T).T
        
        # Convert back from homogeneous coordinates
        world_coords = world_coords[:, :2] / world_coords[:, 2:]
        
        return world_coords

    def world_to_image(self, world_coords):
        image_coords = np.dot(np.linalg.inv(self._H),world_coords.T).T
        image_coords = image_coords[:, :2] / image_coords[:, 2:]
        return image_coords

    def detect_box(self, img=None, visualise=False):
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

        results = self.apply_YOLO_model(img)
        box_loc = np.array(results["box"])

        if visualise:
            result_img = img.copy()
            for p in box_loc:
                cv2.drawMarker(result_img, 
                    tuple(p), 
                    (0, 0, 255), cv2.MARKER_CROSS, 10, 3)
            return self.image_to_world(box_loc), result_img

        return self.image_to_world(box_loc)
    
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


# ALL line detection stuff (used by Camera)
def speedy_ransac(points, num_iterations=100, threshold=10.0, min_inliers=3, estimate = None):
    """
    Implement a speedy RANSAC algorithm to find the line of best fit.
    
    Args:
    points (np.array): Nx2 array of (x, y) coordinates
    num_iterations (int): Number of iterations for RANSAC
    threshold (float): Maximum distance for a point to be considered an inlier
    min_inliers (int): Minimum number of inliers required for a good fit
    
    Returns:
    tuple: (slope, intercept) of the best fit line, or None if no good fit found
    """
    best_inliers = []
    best_slope = best_intercept = None
    min_interations = 10

    for i in range(num_iterations):
        # Randomly select two points
        sample = points[np.random.choice(points.shape[0], 2, replace=True)]
        
        # Calculate slope and intercept
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        
        if x1 == x2:
            continue  # Skip vertical lines
        
        slope = (y2 - y1) / (x2 - x1)

        if estimate is not None and abs(estimate - slope) > GRADIENT_SIMILARITY_THRESH:
            continue # exit this iteration

        intercept = y1 - slope * x1
        
        # Calculate distances of all points to the line
        distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        
        # Find inliers
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_slope = slope
            best_intercept = intercept
        
        if len(best_inliers) > min_inliers and i >= min_interations:
            break
    
    if len(best_inliers) > min_inliers:
        # Refine the fit using all inliers
        x = points[best_inliers, 0]
        y = points[best_inliers, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        best_slope, best_intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return best_slope, best_intercept, best_inliers
    else:
        return (None, None, best_inliers)


def detect_white_line(image, target_point, num_paths=10, visualise=False):
    """
    Detect if there is a white line between the target point and the bottom mid-point of the image
    """
    height, width = image.shape[:2]
    mid_bottom = (width // 2, height - 1)
    
    def extract_path(start, end, offset):
        """
        Extract a single pixel path to the point
        """
        x1, y1 = start
        x2, y2 = end
        path_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        x = np.linspace(x1, x2, path_length).astype(int)
        y = np.linspace(y1, y2, path_length).astype(int)
        
        # Apply horizontal offset
        x = np.clip(x + offset, 0, width - 1)
        
        return image[y, x], np.array(list(zip(x, y)))
    
    def process_path(path):
        """
        Find potential white segment in path
        """
        # Apply Sobel Y filter
        sobel_y = cv2.Sobel(path, cv2.CV_64F, 0, 1, ksize=3)
        
        def detect_peaks(channel, take_max = 5):
            """
            detect peaks in gradient (i.e. edges)
            """
            positive_peaks, _ = find_peaks(channel, distance=10)
            negative_peaks, _ = find_peaks(-channel, distance=10)
            
            # return top results
            return (
                positive_peaks[np.argsort(channel[positive_peaks])[- 1 : -1 - take_max: -1]], 
                negative_peaks[np.argsort(-channel[negative_peaks])[- 1 : -1 - take_max : -1]]
            )
        
        # Find peaks for R and G channels
        r_pos_peaks, r_neg_peaks = detect_peaks(sobel_y[:, 0])
        g_pos_peaks, g_neg_peaks = detect_peaks(sobel_y[:, 1])

        def blue_white_score(posp, negp):
            """
            Looks at the blueness of surrounding pixels and whiteness of potential white segment
            """
            # plot_blue_white_score(path, posp, negp)
            # Compare line sample to before and after
            sample_size = 10
            padding = 5
            pre_sample = path[max(0, posp-padding-sample_size):posp-padding]
            line_sample = path[posp:negp]
            post_sample = path[negp+padding:min(negp+padding+sample_size, len(path)-5)]
            
            def get_blue_score(s):
                mean_rg = np.mean(s[:, [1,2]])
                mean_b = np.mean(s[:, 0])
                return ((mean_b / mean_rg) - 1) * 300, mean_rg

            pre_blue, pre_rg = get_blue_score(pre_sample)
            post_blue, post_rg = get_blue_score(post_sample)

            # Get white score
            # rgb values should be similar
            consistency = 1/(0.5+np.mean(np.var(line_sample, axis=1))) * 200

            white_rg = np.mean(line_sample[:, [1,2]])

            # both should be positive
            pre_change = white_rg - pre_rg
            post_change = white_rg - post_rg

            # Heavily penalise small jumps
            if pre_change < 10:
                pre_change = -100
            if post_change < 10:
                post_change = -100
            
            total_score = pre_blue + post_blue + consistency + pre_change + post_change
            # print(f"Total Blue-White Score: {total_score:.2f}")
            return total_score


        # Calculate confidence scores
        def calculate_best(pos_peaks, neg_peaks, channel):
            """
            Scoring each peak
            """
            best_score = 0
            best_pair = None
            SPACE_BETWEEN_POS_NEG = 40 # roughly max line width
            DIFF_POS_NEG = 0 # pos_peaks[0] * 0.05 # allows slight differents between positive and negative

            for posp in pos_peaks:
                CLOSE_TO_CAM = 10 + posp # favour points closer to camera
                prominence = channel[posp]
                for negp in neg_peaks:
                    # positive edge is always leading
                    if negp > posp:
                        width = negp-posp
                        if width < SPACE_BETWEEN_POS_NEG * 1.1:
                            conf = prominence**2 / (CLOSE_TO_CAM + DIFF_POS_NEG + (width/SPACE_BETWEEN_POS_NEG) ** 8 + abs(prominence + channel[negp]))
                            conf += blue_white_score(posp, negp)
                            if conf > best_score:
                                best_pair = [posp, negp]
                                best_score = conf
            
            return np.array(best_pair), best_score
        
        r_pair, r_confidence = calculate_best(r_pos_peaks, r_neg_peaks, sobel_y[:, 0])
        g_pair, g_confidence = calculate_best(g_pos_peaks, g_neg_peaks, sobel_y[:, 1])
        
        # Combine r and g scores
        conf = 0
        pair = np.array([0, 0])

        # Both red and green channels have identified a segment
        if r_confidence != 0 and g_confidence != 0:
            distance = np.linalg.norm(r_pair - g_pair)

            # Average the position of the peaks if they are close
            if distance < 10:
                pair = (r_pair + g_pair) / 2 # average coordinates
                conf = (1 - distance/50) * (r_confidence + g_confidence)
            # Take red channel if they are not close
            else:
                pair = r_pair
                conf = r_confidence
        elif r_confidence != 0:
            pair = r_pair
            conf = r_confidence
        elif g_confidence != 0:
            pair = g_pair
            conf = g_confidence

        if visualise:
            return conf, sobel_y, (r_pos_peaks, r_neg_peaks, g_pos_peaks, g_neg_peaks), pair.astype(int)
        else:
            return pair.astype(int), conf

    # Extract and process multiple paths
    offsets = np.linspace(-width // 5, width // 5, num_paths)
    chosen_peaks = []
    confidences = []
    leading_points = []
    trailing_points = []

    if visualise:
        all_path_points = []
        all_sobel_data = []
        all_peaks = []

    for offset in offsets:
        path, path_points = extract_path(mid_bottom, target_point, int(offset))

        if visualise:
            confidence, sobel_data, peaks, pair = process_path(path)
            
            all_path_points.append(path_points)
            all_sobel_data.append(sobel_data)
            all_peaks.append(peaks)
            
        else:
            pair, confidence = process_path(path)
        
        leading_points.append(path_points[pair[0]])
        trailing_points.append(path_points[pair[1]])
        chosen_peaks.append(pair)
        confidences.append(confidence)
    
    # Find slope and intercept of leading and trailing edges
    lead_slope, lead_int, lead_inliers = speedy_ransac(np.array(leading_points), threshold=10, min_inliers=num_paths//3)
    trail_slope, trail_int, trail_inliers = speedy_ransac(np.array(trailing_points), threshold=10, min_inliers=num_paths//3, estimate=lead_slope)
    
    confidences = np.array(confidences)
    lead_confidence = np.sum(confidences[lead_inliers])
    trail_confidence = np.sum(confidences[trail_inliers])
    combined_confidence = (lead_confidence + trail_confidence) / num_paths

    lines = [
        [lead_slope, lead_int],
        [trail_slope, trail_int]
    ]

    if lead_slope is not None and trail_slope is not None:
        # Check the two edges make a sensible line
        grad_diff = abs(lead_slope - trail_slope)
        int_diff = lead_int - trail_int
        # may depend on resolution (GRADIENT_SIMILARITY_THRESH works well for 640x480px)
        if not (grad_diff < GRADIENT_SIMILARITY_THRESH and -INTERCEPT_DIFF_THRESH//2 < int_diff < INTERCEPT_DIFF_THRESH):
            lines = None # invalid detection
    else:
        # No lines found
        lines = None
    if visualise:
        # Visualize results
        result_image = visualize_results(image, target_point, combined_confidence, lines, leading_points, trailing_points, all_path_points)
        path_index = num_paths // 2
        peak_fig = plot_peaks(all_sobel_data[path_index], all_peaks[path_index], path_index, chosen_peaks[path_index])
        return lines, combined_confidence, result_image, peak_fig
    else:
        return lines, combined_confidence

def plot_peaks(sobel_data, peaks, path_index, chosen_peaks):
    r_pos_peaks, r_neg_peaks, g_pos_peaks, g_neg_peaks = peaks
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot R channel
    ax1.plot(sobel_data[:, 2], label='R channel')
    ax1.plot(r_pos_peaks, sobel_data[r_pos_peaks, 2], "ro", label='Positive peaks')
    ax1.plot(r_neg_peaks, sobel_data[r_neg_peaks, 2], "go", label='Negative peaks')
    ax1.plot(chosen_peaks, sobel_data[chosen_peaks, 2], "kx", label='Chosen Peaks')
    ax1.set_title(f'R Channel - Path {path_index}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot G channel
    ax2.plot(sobel_data[:, 1], label='G channel')
    ax2.plot(g_pos_peaks, sobel_data[g_pos_peaks, 1], "ro", label='Positive peaks')
    ax2.plot(g_neg_peaks, sobel_data[g_neg_peaks, 1], "go", label='Negative peaks')
    ax2.plot(chosen_peaks, sobel_data[chosen_peaks, 1], "kx", label='Chosen Peaks')
    ax2.set_title(f'G Channel - Path {path_index}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# The visualize_results function and example usage remain the same as in the previous version
def visualize_results(image, target_point, confidence, lines, leading_edge = [], trailing_edge = [], all_path_points=[]):
    height, width = image.shape[:2]
    mid_bottom = (width // 2, height - 1)
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw paths
    for path_points in all_path_points:
        for x, y in path_points:
            cv2.circle(vis_image, (x, y), 1, (0, 255, 255), -1)
    
    # Draw target point
    cv2.circle(vis_image, target_point, 5, (0, 0, 255), -1)
    
    # Draw mid-bottom point
    cv2.circle(vis_image, mid_bottom, 5, (255, 0, 0), -1)
    
    # Draw leading and trailing edges
    for p in leading_edge:
        cv2.drawMarker(vis_image, 
                tuple(p), 
                (0, 255, 0), cv2.MARKER_CROSS, 5)
    for p in trailing_edge:
        cv2.drawMarker(vis_image, 
                tuple(p), 
                (0, 0, 255), cv2.MARKER_CROSS, 5)
    
    if lines is not None:
        for l in lines:
            x1, x2 = 0, width
            y1, y2 = int(l[1]), int(l[0] * width + l[1])
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add text with results
    # cv2.putText(vis_image, f"Leading edge: {leading_edge:.2f}", (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(vis_image, f"Trailing edge: {trailing_edge:.2f}", (10, 60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    colour = (0, 0, 255) if confidence > LINE_CONF_THRESHOLD else (0,255,0)
    cv2.putText(vis_image, f"Line confidence: {confidence:.2f}", 
                (np.clip(target_point[0], 10, width-30), np.clip(target_point[1], 10, height-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    
    return vis_image

class TestCamera(unittest.TestCase):
    """
    Test Camera specific functions
    """
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    # @unittest.skip("skipped")
    def test_image_to_world(self):
        self.cam = Camera(False) # no camera
        img_c = np.array([[IMG_WIDTH//2, IMG_HEIGHT//2]])
        coord = self.cam.image_to_world(img_c)
        print(coord)
    
    @unittest.skip("skipped")
    def test_world_to_image(self): 
        self.cam = Camera(False)   
        for x in np.arange(-2, 2, 0.5):
            for y in np.arange(1.0, 5, 0.5):
                world_c = np.array([[x, y, 1]])
                img_c = self.cam.world_to_image(world_c)
                world_check = self.cam.image_to_world(img_c)
                np.testing.assert_allclose(world_c[:, :2], world_check, atol=0.01)
                print(world_c, "(world) -> (img)", img_c)
                print(img_c, "(img) -> (world)", world_check, '\n')

    @unittest.skip("skipped")
    def test_YOLO_model(self):
        # Open image(s) and pass to model
        self.cam = Camera(False) # no camera

        for n in range(1,6):
            image_path = f'CV/test_imgs/box/{n:04g}.jpg'
            img = cv2.imread(image_path)
            res, res_img = self.cam.apply_YOLO_model(img, visualise=True)
            cv2.imwrite(f'{RESULT_OUT_PREFIX}/YOLO_result{n}.jpg', res_img)
            print(res)

    @unittest.skip("skipped")
    def test_capture(self):
        self.cam = Camera(True)
        self.startTime = time.time()
        img = self.cam.capture()
        if img is not None:
            cv2.imwrite(f"{RESULT_OUT_PREFIX}/capture_result.jpg", img)
        self.assertTrue(img is not None, "Camera did not capture anything")

    @unittest.skip("skipped")
    def test_ball_detection(self):
        # Open image(s) and pass to function
        self.cam = Camera(False)
        for n in range(0,190,10):
            image_path = f'CV/test_imgs/test_images/testing{n:04g}.jpg'
            # image_path = f'./test_imgs/blender/oneball/normal{n:04g}.jpg'
            image = cv2.imread(image_path)
            locs, result_img = self.cam.detectBalls(image, visualise=True)
            cv2.imwrite(f"{RESULT_OUT_PREFIX}/ball_detect_result_{n}.jpg", result_img)
    
    @unittest.skip("skipped")
    def test_line_detection(self):
        # pass images to line detection
        for n in range(0,60,20):
            image_path = f'CV/test_imgs/test_images/testing{n:04g}.jpg'
            # image_path = f'./test_imgs/blender/oneball/normal{n:04g}.jpg'
            image = cv2.imread(image_path)
            target_points = [(640, 200), (100, 200), (800, 150)]  # Example target points
            for target_point in target_points: #target_points:
                s = time.time()
                lines, confidence, vis_image, peak_fig = detect_white_line(image, target_point, 12, True)
                tp_str = f'{target_point[0]}_{target_point[1]}'
                cv2.imwrite(f'{RESULT_OUT_PREFIX}/line_detect_{n}_{tp_str}.jpg', vis_image)
                peak_fig.savefig(f'{RESULT_OUT_PREFIX}/line_detect_{n}_{tp_str}_peaks.jpg')
                e = time.time()

                s = time.time()
                lines, confidence = detect_white_line(image, target_point, 12)
                e = time.time()
    
    @unittest.skip("skipped")
    def test_box_detection(self):
        # Open images and pass to function
        self.cam = Camera(False)
        for n in range(1,6):
            image_path = f'CV/test_imgs/box/{n:04g}.jpg'
            # image_path = f'./test_imgs/blender/oneball/normal{n:04g}.jpg'
            image = cv2.imread(image_path)
            locs, result_img = self.cam.detect_box(image, visualise=True)
            cv2.imwrite(f"{RESULT_OUT_PREFIX}/box_detect_result_{n}.jpg", result_img)
    
    


def _capture_loop(detect=False, stream=False, straight_line=False):
    """
    Continuously captures images and saves to test_results folder

    Parameters
    ---
    detect: bool
    Run YOLO model and save bounding boxes on output
    """
    cam = Camera(True)
    i = 0
    
    while True:
        s = time.time()
        img = cam.capture()

        if img is not None:
            
            if detect:
                results, result_img = cam.apply_YOLO_model(img, visualise=True)
            else:
                result_img = img.copy()

            # Stright vertical line and horizontal line
            if straight_line:
                cv2.line(result_img, (IMG_WIDTH//2, 0), (IMG_WIDTH//2, IMG_HEIGHT), (0, 0, 255), 1)
                cv2.line(result_img, (0, IMG_HEIGHT//2), (IMG_WIDTH, IMG_HEIGHT//2), (0, 0, 255), 1)
        else:
            print("Capture failed.")
            break

        e = time.time()

        out_file = f"./test_results/result{i}.jpg"
        # Use this if off the pi
        if stream:
            cv2.imshow('Calibration, [press q to quit, s to save]', result_img)
            k = cv2.waitKey(50) 
        else:
            cv2.imwrite(out_file, result_img)
            print(f"Captured in {e-s} sec. Saved in {out_file}.")
            k = input("ENTER to continue, q to quit: ")
        
        if k == ord('q') or k == 'q':
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            cv2.imwrite(out_file, result_img)
            print(f"Captured in {e-s} sec. Saved in {out_file}")
            i += 1


def _overlay_calibration(stream=True):
    """
    Parameters
    ---
    stream : bool
        Stream video from webcam if true. Save one image if false
    """
    cam = Camera(True)
    xlines = np.arange(0.5, 5.0, 0.5)
    ylines = np.arange(-3.0, 3.0, 0.5)

    lines = []
    
    
    for x in xlines:
        img_cs = cam.world_to_image(np.array([
            [x, ylines[0], 1],
            [x, ylines[-1], 1],
            [x, ylines[len(ylines)//2], 1]
        ])).astype(int)
        lines.append((img_cs, str(x)))
    
    for y in ylines:
        img_cs = cam.world_to_image(np.array([
            [xlines[0], y, 1],
            [xlines[-1], y, 1],
            [xlines[len(xlines)//2], y, 1]
        ])).astype(int)
        lines.append((img_cs, str(y)))
    

    def draw_line(calibration_img, c, label):
        col = (0, 0, 255)
        cv2.line(calibration_img, tuple(c[0]), tuple(c[1]), col, 1)
        cv2.putText(calibration_img, label, tuple(c[2] + [15, 15]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))

    while True:
        calibration_img = cam.capture()
        for l in lines:
            draw_line(calibration_img, l[0], l[1])

        # Use this if off the pi
        if stream:
            cv2.imshow('Calibration, [press q to quit]', calibration_img)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        
        # Capture one image and save if on the pi
        else:
            cv2.imwrite('test_results/calibration.jpg', calibration_img)
            print("Overlaid grid onto capture. Saved in test_results/calibration.jpg")
            break


if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCamera)
    # unittest.TextTestRunner(verbosity=0).run(suite)
    _capture_loop(detect=True, stream=True, straight_line=False)
    # _overlay_calibration(stream=True)