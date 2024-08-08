import cv2
import numpy as np
from matplotlib import pyplot as plt

# Initialize the camera (assuming it's the only USB camera connected)
cap = cv2.VideoCapture(1)  # Change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 40)

# Actual width of the marker (in cm)
actual_width = 2.0  # Adjust this to the actual width of your marker

# Focal length in pixels
focal_length = 620.0  # in pixels

ret = False
while not ret:
    ret, frame = cap.read()

# Convert the frame to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the range for the color of the whiteboard marker (adjust as needed)
# Assuming the marker is blue
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Apply the color threshold to create a mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Perform edge detection using Canny
edges = cv2.Canny(mask, 100, 200)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find the whiteboard marker
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if area > 1000:  # Adjust the area threshold as needed
        # Draw a bounding box around the detected marker
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate the distance to the marker
        perceived_width = w  # width of the bounding box
        distance = (actual_width * focal_length) / perceived_width
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the original frame with detected whiteboard marker and distance
plt.subplot(131)
plt.imshow(frame[:, :, [2, 1, 0]])
plt.title('Original Image with Detection')
plt.xticks([])
plt.yticks([])

# Display the mask
plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title('Color Mask')
plt.xticks([])
plt.yticks([])

# Display the edge-detected frame
plt.subplot(133)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()

cap.release()
