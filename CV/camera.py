### Based off code provided in ECE4191 Github file called Camera_image_retrieval.ipynb
# Author: Edric Lay, 28/07
# Last edited: Edric Lay, 30/07
import cv2

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

if __name__ == "__main__":
    # Use to this test if the camera is working properly.
    img = capture()
    cv2.imshow(img)

