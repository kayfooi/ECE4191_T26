"""
Given a tiled/grid floor with known geometry (see calibration.csv)
and an image of the floor from a fixed camera, calculates the homography 
that transforms image coordinates to 2D world coordinates.

Requires manual labelling of real-world points and corresponding image
points.

"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Convert image coords to world coords with homography H
def image_to_world(image_coords, H):
    # Convert to homogeneous coordinates
    homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
    
    # Apply the homography
    world_coords = np.dot(H, homogeneous_coords.T).T
    
    # Convert back from homogeneous coordinates
    world_coords = world_coords[:, :2] / world_coords[:, 2:]
    
    return world_coords

# figure for distortion corrections
def calibrate():

    # Load the data
    data = pd.read_csv('calibration.csv')

    # Prepare arrays for findHomography function
    src_points = data[['Image X (px)', 'Image Y (px)']].values.astype(np.float64)

    true_coords = (data[['World X (m)', 'World Y (m)']].values).astype(np.float64)





    # Calculate the homography matrix using RANSAC
    H, mask = cv2.findHomography(src_points, true_coords, cv2.RANSAC, 5.0)
    calc_coords = image_to_world(src_points, H)

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot calculated positions
    ax.scatter(calc_coords[:, 0], calc_coords[:, 1], color='red', s=40, label='Calculated', marker='x')

    # Plot true positions
    ax.scatter(true_coords[:, 0], true_coords[:, 1], color='blue', s=40, alpha=0.5, label='Wold', marker='x')

    ax.set_xlabel('X World Coordinate')
    ax.set_ylabel('Y World Coordinate')
    ax.set_title('Calculated vs True World Coordinates of Tennis Balls (RANSAC)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()

    # Print the homography matrix
    print("\nHomography Matrix:")

    print('[')
    for l in H:
        print(str(list(l))+',')
    print(']')

    plt.show()

# Last calculated 9AM Wed 28th Aug
H = [
[-0.014210389999953848, -0.0006487560233598932, 9.446387805048925],
[-0.002584902022933329, 0.003388864890354594, -17.385493275570447],
[-0.0029850356852013345, -0.04116105685090471, 1.0],
]

if __name__ == "__main__":
    calibrate()