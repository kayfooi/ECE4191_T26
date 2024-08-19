### Based off code provided in ECE4191 Github file called Camera_image_retrieval.ipynb
# Author: Edric Lay, 28/07
# Last edited: Edric Lay, 30/07
import cv2
import numpy as np

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, 3) #type:ignore
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tennis_balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if 5 < area < 700 and circularity > 0.4:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            tennis_balls.append([cx, cy])

    return np.array(tennis_balls)

if __name__ == "__main__":
    # Use to this test if the camera is working properly.
    img = capture()
    cv2.imshow('test', img)

