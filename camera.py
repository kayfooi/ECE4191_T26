### Based off code provided in ECE4191 Github file called Camera_image_retrieval.ipynb
import cv2
import matplotlib.pyplot as plt

def cameraDisp():
    # Read from camera feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS, 40)

    # Display
    ret = False
    while (not ret):
        ret, frame = cap.read()
        cv2.imshow(frame)

    # Close
    cap.release()

if __name__ == "__main__":
    # Use to this test if the camera is working properly.
    cameraDisp()