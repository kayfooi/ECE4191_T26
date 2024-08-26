import cv2
import numpy as np

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    
    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv

def detect_tennis_ball(image):
    # Load the image
    # image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None

    # Resize the image for consistency
    # resized = cv2.resize(image, (600, 400))

    # Preprocess the image
    hsv = preprocess_image(image)

    # Define HSV range for the tennis ball (adjustable)
    greenLower = (30, 40, 40)
    greenUpper = (85, 255, 255)

    # Create a mask for the tennis ball
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    # Remove noise with morphological operations
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        # Calculate the circularity of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Apply circularity and size filtering
        if 0.7 < circularity < 1.2 and area > 100:
            # Compute the minimum enclosing circle and centroid
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Further filter small objects
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    results.append(center)
                    # Draw the circle and centroid on the image
                    # cv2.circle(resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # cv2.circle(resized, center, 5, (0, 0, 255), -1)
                    # print(f"Detected ball at center: {center}")

    # Show the result
    # cv2.imshow("Detected Tennis Ball", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.array(results)

# Example usage with your image:
import imutils
from collections import deque

def detect_ball_circularity_colour(frame):
    greenLower = (30, 40, 40)
    greenUpper = (85, 255, 255)
    pts = deque(maxlen=64)
    
    # resize the frame, blur it, and convert it to the HSV color space
    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = []

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:  # avoid division by zero
            # only proceed if the radius meets a minimum size
            if radius > 5:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # update the points queue
                pts.appendleft(center)
                # output the center of the ball (x, y) coordinates
                # print(f"Center: {center}")


    
    return np.array(pts)


#detect_tennis_ball(r'Test\test0.png')
