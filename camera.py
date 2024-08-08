### Based off code provided in ECE4191 Github file called Camera_image_retrieval.ipynb
# Author: Edric Lay, 28/07
# Last edited: Edric Lay, 30/07
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

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


## image classification

# normalising images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

