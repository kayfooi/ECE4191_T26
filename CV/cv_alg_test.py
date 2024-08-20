import cv2
import numpy as np

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    
    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv

def detect_tennis_ball(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None

    # Resize the image for consistency
    resized = cv2.resize(image, (600, 400))

    # Preprocess the image
    hsv = preprocess_image(resized)

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

                    # Draw the circle and centroid on the image
                    cv2.circle(resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(resized, center, 5, (0, 0, 255), -1)
                    print(f"Detected ball at center: {center}")

    # Show the result
    cv2.imshow("Detected Tennis Ball", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage with your image:




#detect_tennis_ball(r'Test\test0.png')
