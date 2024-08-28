### Based off code provided in ECE4191 Github file called Camera_image_retrieval.ipynb
# Author: Edric Lay, 28/07
# Last edited: Edric Lay, 30/07
import cv2
import numpy as np
from ultralytics import YOLO
import time
from camera_calibrate import H, image_to_world

class Camera:
    def __init__(self, display=False, log=True):
        # Wake camera up
        self.cap = cv2.VideoCapture(0)
        time.sleep(1)
    
    def capture(self):
        ret, frame = self.cap.read()
        return frame
    
    def detect_ball(self, img):
        return YOLOv2(img)


# Load a model
model1 = YOLO("YOLOv1.pt")  # load a custom model
model2 = YOLO("YOLOv2.pt")

def YOLOv1(image):
    # Predict with the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # scale = np.array(image.shape[:2]) / np.array((640,  640))

    results = model1(image, conf=0.20, verbose=False)  # predict on an image

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
    return (np.array(points)).astype(int)

def YOLOv2(image):
    # Predict with the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # scale = np.array(image.shape[:2]) / np.array((640,  640))

    results = model2(image, conf=0.22, verbose=False)  # predict on an image

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
    return (np.array(points)).astype(int)

def capture():
    # Read from camera feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS, 40)

    # Display
    ret = False
    while (not ret):
        ret, frame = cap.read()

    # Close
    cap.release()

    return frame


def detect_ball_circularity_no_colour(image):
    """
    image > grayscale > blurred > Canny edge detection > countours > check circularity & area
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for tennis ball (yellow-green)
    lower_yellow = np.array([230, 230, 230])
    upper_yellow = np.array([255, 255, 255])
    
    # Create a mask for the tennis ball color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store detected ball coordinates
    ball_coordinates = []
    
    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)
        
        # Filter out small noisy contours
        if area > 100:  # Adjust this threshold as needed
            # Find the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Check if the contour is approximately circular
            circularity = 4 * np.pi * area / (2 * np.pi * radius) ** 2
            if 0.8 < circularity < 1.2:  # Adjust this range as needed
                ball_coordinates.append(center)
                
                # Draw circle on the original image (for visualization)
                cv2.circle(image, center, radius, (0, 255, 0), 2)
    
    # Optionally, save or display the result
    cv2.imwrite('detected_balls.jpg', image)
    
    return ball_coordinates

    return np.array(tennis_balls)

def detect_ball_circularity_no_blue(img):
    """
    image > blurred > take out blue > Canny edge detection > countours > check circularity & area
    """
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the range of blue hue
    # lower_blue = np.array([50, 5, 5])
    # upper_blue = np.array([180, 255, 255])
    lower = np.array([0, 0, 0])
    upper = np.array([254, 254, 254])

    # Create a mask for blue hue
    mask = cv2.inRange(hsv, lower, upper)
    
    # Remove noise with morphological operations
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)


    # Apply the mask to the original image
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result[mask > 0] = 0

    # Find contours
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        # Calculate the circularity of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Apply circularity and size filtering
        if 0.7 < circularity < 1.2 and area > 10:
            # Compute the minimum enclosing circle and centroid
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Further filter small objects
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    results.append(center)

    # Save the result
    # cv2.imwrite('output_image.jpg', result)

    # Optional: Display the result
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array(results)

def peek(img):
    cv2.imshow("Peek", img)
    cv2.waitKey(0)

def find_circles_template_match(image, min_diameter=20, max_diameter=100, downsample_factor = 0.5):
    
    # Downsample the image
    downsampled = cv2.resize(image, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA)
    
        # Convert to HSV color space
    hsv = cv2.cvtColor(downsampled, cv2.COLOR_BGR2HSV)
    
    # Define blue color range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Invert the mask to keep non-blue pixels
    non_blue_mask = cv2.bitwise_not(blue_mask)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(downsampled, downsampled, mask=non_blue_mask)
    
    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    peek(gray)
    
    # Adjust min and max diameters for downsampled image
    min_diameter_down = int(min_diameter * downsample_factor)
    max_diameter_down = int(max_diameter * downsample_factor)
    
    # Create a list to store the detected circles
    detected_circles = []
    
    # Loop through different sizes
    for diameter in range(min_diameter_down, max_diameter_down + 1, 2):
        # Create a circular template
        border = 3
        twidth = diameter + border*2
        template = np.zeros((twidth, twidth), dtype=np.uint8)
        cv2.circle(template, (twidth//2, twidth//2), diameter//2, 255, 1)
        
        # Perform template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find positions where the matching score is above a threshold
        threshold = 0.3
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt[0] + diameter//2, pt[1] + diameter//2
            detected_circles.append((x, y, diameter))
    
    # Apply non-maximum suppression to remove overlapping detections
    # detected_circles = non_max_suppression(detected_circles)
    detected_circles = [[x, y+diameter//2] for (x, y, diameter) in detected_circles]
    return np.array(detected_circles) / downsample_factor

def non_max_suppression(circles, overlap_thresh=0.5):
    if len(circles) == 0:
        return []
    
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    keep = []
    
    while len(circles) > 0:
        circle = circles.pop(0)
        keep.append(circle)
        
        circles = [c for c in circles if iou(circle, c) < overlap_thresh]
    
    return keep

def iou(circle1, circle2):
    x1, y1, d1 = circle1
    x2, y2, d2 = circle2
    
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if distance > (d1 + d2) / 2:
        return 0
    
    area1 = np.pi * (d1/2)**2
    area2 = np.pi * (d2/2)**2
    
    intersection_area = circle_intersection_area(d1/2, d2/2, distance)
    
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area

def circle_intersection_area(r1, r2, d):
    if d <= abs(r2 - r1):
        return np.pi * min(r1, r2)**2
    if d >= r1 + r2:
        return 0
    
    r1_sq, r2_sq = r1**2, r2**2
    d_sq = d**2
    
    alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
    beta = np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
    
    area = (r1_sq * alpha + r2_sq * beta -
            0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) *
                          (d - r1 + r2) * (d + r1 + r2)))
    
    return area


import torch
import cv2
import numpy as np
from torchvision import transforms

def process_frame(frame, model):
    # Convert the frame to RGB (YOLOv5 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch tensor
    transform = transforms.ToTensor()
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Process the outputs (this will depend on your specific model)
    # For YOLOv5, the output is already in a convenient format
    return outputs

def draw_boxes(frame, detections):
    for result in detections:
        print(result)
        for box in result.boxes:
            det = box.xyxy # det is (x1, y1, x2, y2, conf, cls)
            # print(det)
            x1, y1, x2, y2 = map(int, det[0])
            world_c = image_to_world(np.array([[(x1 + x2)/2, y2]]), H)

            label = f"{result.names[int(box.cls[0])]} @ ({world_c[0, 0]:.2f}, {world_c[0,1]:.2f}) conf: {box.conf[0]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main():
    model = model2
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model1(frame, conf=0.50, verbose=False)
        frame_with_boxes = draw_boxes(frame, detections)

        cv2.imshow('Object Detection', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use to this test if the camera is working properly.
    # img = capture()
    # cv2.imshow('test', img)

    # for i in range(20):
    #     imgpath = f'test_imgs/test_images/real{i:04d}.jpg'
    #     img = cv2.imread(imgpath)
    #     YOLOv1(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    main()






