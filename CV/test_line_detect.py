import ncnn
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

model = ncnn.Net() #type:ignore
model.load_param("./YOLO_ball_detection_ncnn_model/model.ncnn.param")
model.load_model("./YOLO_ball_detection_ncnn_model/model.ncnn.bin")

import numpy as np

def calculate_adaptive_threshold(image):
    height, _, _ = image.shape
    avg_color = np.mean(image, axis=(0, 1))
    avg_gray = np.mean(avg_color)
    threshold = min(int(avg_gray) + 70, 253)
    return threshold

def isolate_white_pixels(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(image, image, mask=white_mask)
    return result

def plot_bgr_hsv_column(image, column):
    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")
    
    # Check if the column is within the image bounds
    if column < 0 or column >= image.shape[1]:
        raise ValueError("Column index out of bounds")
    
    # Extract the specified column
    bgr_column = image[:, column, :]
    
    # Convert BGR to HSV
    hsv_column = cv2.cvtColor(bgr_column.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Create x-axis values (pixel indices)
    pixel_indices = np.arange(bgr_column.shape[0])
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot BGR values
    ax1.plot(pixel_indices, bgr_column[:, 0], 'b-', label='Blue')
    ax1.plot(pixel_indices, bgr_column[:, 1], 'g-', label='Green')
    ax1.plot(pixel_indices, bgr_column[:, 2], 'r-', label='Red')
    
    ax1.set_title(f'BGR Values for Column {column}')
    ax1.set_xlabel('Pixel Index')
    ax1.set_ylabel('Color Intensity')
    ax1.legend()
    ax1.grid(True)
    
    # Plot HSV values
    ax2.plot(pixel_indices, hsv_column[:, 0], 'm-', label='Hue')
    ax2.plot(pixel_indices, hsv_column[:, 1], 'c-', label='Saturation')
    ax2.plot(pixel_indices, hsv_column[:, 2], 'y-', label='Value')
    
    ax2.set_title(f'HSV Values for Column {column}')
    ax2.set_xlabel('Pixel Index')
    ax2.set_ylabel('HSV Values')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# image = cv2.imread('your_image.jpg')
# plot_bgr_hsv_column(image, 100)  # Plot BGR and HSV values for column 100

def apply_YOLO_model(image):
    """
    Apply YOLO model to get locations of balls
    """
    # Preprocessing
    height, width, _ = image.shape
    img_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = image.copy()
    shape = img.shape[:2]  # current shape [height, width]
    size = 640
    new_shape = (size, size)

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
    mat_in = ncnn.Mat(in0) #type:ignore

    # Run inference
    extractor = model.create_extractor()
    extractor.input("in0", mat_in)
    ret, mat_out = extractor.extract("out0")
    out = np.array(mat_out)
    points = []

    # Process results (find the max)
    for i in range(mat_out.w):
        detection = out[:, i]
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.6:
            # Object detected
            # TODO: Filter out invalid detections based on box size
            xywh = detection[:4] / size # Centered coordinates
            y = detection[1]
            y = (y - top) / (size - top - bottom)
            
            x = int(xywh[0] * width)
            y = int(y * height + xywh[3] * width * 0.5)
            
            # Check if it already exists
            duplicate = False
            for p in points:
                if (p[0] - x) ** 2 + (p[1] - y)**2 < 5:
                    duplicate = True
                    break # don't add duplicate points
            
            if not duplicate:
                points.append([x, y])

    points = np.array(points)
    return points.astype(int)

def plot_rgb(signal):
    # plt.plot(signal[:, 0], 'b')
    plt.plot(signal[:, 0], 'g')
    plt.plot(signal[:, 1], 'r')

def is_line_before_ball(img, target):
    """
    Check if there's a line before the ball in the image.
    
    Args:
    img (numpy.ndarray): The input image.
    target (list): The [x, y] coordinates of the detected ball.
    
    Returns:
    bool: True if the ball is not centered (implying a line might be before it), False otherwise.
    """
    h, w, c = img.shape
    # Reduce image size to speed up processing
    scale_down_factor = 1
    h, w = h//scale_down_factor, w//scale_down_factor
    target = np.array(target).astype(int) // scale_down_factor
    img = cv2.resize(img, (w, h))
    xmid = w//2
    
    # Define the width of the path to analyze
    path_width = 50
    os = path_width // 2  # offset from path center
    path_height = h - target[1]

    # Create an array to store the pixels along the path to the ball
    path_to_ball = np.zeros((path_height + 1, path_width, c)).astype(np.uint8)
    
    # Calculate the gradient of the path
    gradient = (xmid - target[0]) / path_height

    # Extract the pixels along the path
    for y in range(path_height):
        x = int(target[0] + y * gradient)
        if x-os > 0 and x + os < w:
            path_to_ball[y] = img[y + target[1], x-os:x+os, :]
        else:
            return True  # Ball is not centered in the image

    # Apply Sobel edge detection to the path
    ksize = 5
    edges = cv2.Sobel(path_to_ball, cv2.CV_16S, 0, 1, ksize=ksize, scale=10)

    # Analyze multiple columns along the path
    xcoords = [os//2, os, os+os//2]
    for xcoord in xcoords:
        edge = edges[:, xcoord, :]

        # Find peaks in the red channel (most prominent in blue -> white edge)
        rpeaks, rprops = find_peaks(edge[:, 2], prominence=20000, width=3, wlen=20, distance=10)
        
        for i, peak in enumerate(rpeaks):
            # Ignore peaks at the very start or end of the path
            if peak < 5 or peak > path_height - 6:
                continue

            # Extract properties of the peak
            width = rprops["widths"][i]
            rb = rprops["right_bases"][i]
            lb = rprops["left_bases"][i]
            gwidth = rb - lb
            offset = int(gwidth/2)
            prominence = rprops["prominences"][i]

            # Extract the window around the peak
            peak_window = edge[lb:rb, 1:]
            
            # Search for a matching negative peak (white -> blue transition)
            min_err = gwidth * prominence ** 2 / 4
            og_min_err = min_err
            neg_peak_window = None
            min_dp = None
            errs = []
            minimising = False
            no_min = 0

            for dp in range(int(offset), min(30, path_height - rb)):
                comp_peak_window = edge[lb + dp:rb + dp, 1:]
                combined = (peak_window + comp_peak_window).astype(np.int64)
                
                # Calculate error between the two peaks
                err = np.sum(combined**2)
                errs.append(err)

                # Update if this is the best match so far
                if err < min_err:
                    neg_peak_window = comp_peak_window
                    min_err = err
                    min_dp = dp
                    minimising = True
                    no_min = 0
                else:
                    if minimising:
                        no_min += 1
                    if no_min > 10:
                        break

            # If a matching negative peak is found, mark it on the image
            if neg_peak_window is not None:
                # Mark the positive peak
                cv2.drawMarker(path_to_ball, (xcoord, peak), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                orig_y = peak + target[1]
                orig_x = target[0] - os + xcoord + int(peak * gradient)
                cv2.drawMarker(img, (orig_x, orig_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                
                # Mark the corresponding negative peak
                cv2.drawMarker(path_to_ball, (xcoord, peak+min_dp), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                orig_y = peak + min_dp + target[1]
                orig_x = target[0] - os + xcoord + int((peak + min_dp) * gradient)
                cv2.drawMarker(img, (orig_x, orig_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

    # Display the processed image
    cv2.imshow('img', img)
    cv2.waitKey()
      

for n in range(1, 100, 10):
    # n = 27
    image_path = f'./test_imgs/test_images/testing{n:04g}.jpg'
    # image_path = f'./test_line_{n}.jpg'
    print(n)
    img = cv2.imread(image_path)
    
    s = time.time()
    points = apply_YOLO_model(img)
    e = time.time()
    print(f'Model took: {(e-s) * 1e3 : .2f} ms')

    # is_line_before_ball(img, points[1])
    for p in points:
        is_line_before_ball(img, p)

    for (i, p) in enumerate(points):
        cv2.putText(img, f'tennis-ball-{i}', (p[0]-50, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
        cv2.drawMarker(img, tuple(p), (0, 0, 255), cv2.MARKER_CROSS, 20, 3, 0)
    
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(points)

cv2.destroyAllWindows()