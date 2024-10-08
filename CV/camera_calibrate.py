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

# figure for distortion corrections
fig1, ax1 = plt.subplots(figsize=(8, 6))
def correct_barrel_distortion(image_coords, alpha = 0.00, plot_correction = False):
    CAM_X_RES = 640
    CAM_Y_RES = 480
    DIST_CORRECTION = alpha # parameter to correct barrel distortion

    # Correct Barrel Distortion
    centre_offset = image_coords - np.array([CAM_X_RES/2, CAM_Y_RES/2])
    centre_distance = np.vstack(np.sqrt(np.sum(centre_offset**2, axis=1)))
    centre_dir = centre_offset/centre_distance # normalise
    new_image_coords = image_coords + centre_dir*centre_distance*DIST_CORRECTION
    
    if plot_correction:
        ax1.set_title('Image Barrel Distortion Correction')
        ax1.scatter(image_coords[:, 0], image_coords[:, 1], color='red', s=40, label='Raw Image Coordinates', marker='x')
        ax1.scatter(new_image_coords[:, 0], new_image_coords[:, 1], color='blue', s=40, label='Corrected Coordinates', marker='x')
        ax1.scatter(CAM_X_RES/2, CAM_Y_RES/2, color='k', marker='s', label='Image Centre')
        ax1.axis('equal')
        ax1.grid(True)
        ax1.legend()
    
    return new_image_coords

# Mean Square Error
def calculate_error(calc, true):
    SE = np.sum((calc - true)**2)
    MSE = SE / true.shape[0]
    return MSE

# Load the data
data = pd.read_csv('calibration.csv')

# Prepare arrays for findHomography function
src_points = data[['Image X (px)', 'Image Y (px)']].values.astype(np.float64)

true_coords = (data[['World X (m)', 'World Y (m)']].values).astype(np.float64)

# Convert image coords to world coords with homography H
def image_to_world(image_coords, H):
    # Convert to homogeneous coordinates
    homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
    
    # Apply the homography
    world_coords = np.dot(H, homogeneous_coords.T).T
    
    # Convert back from homogeneous coordinates
    world_coords = world_coords[:, :2] / world_coords[:, 2:]
    
    return world_coords

# Trying differrent distortion correction parrameters to see which one performs best
# Have found that this doesn't really fix anything
alphas = list(np.arange(-0.1, 0.1, 0.01))
min_error = math.inf
best_alpha = 0.00

errors = []
for alpha in alphas:
    # corrected = correct_barrel_distortion(src_points, alpha, False)
    # Calculate the homography matrix using RANSAC
    H, mask = cv2.findHomography(srcPoints=, true_coords, cv2.RANSAC, 5.0)
    calculated_world_coords = image_to_world(corrected, H)
    error = calculate_error(calculated_world_coords, true_coords)
    errors.append(error)
    if error < min_error:
        best_H = H.copy()
        best_calc_coords = calculated_world_coords.copy()
        min_error = error
        best_alpha = alpha

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title("Finding optimal distortion correction parameter")
ax2.scatter(alphas, errors)
print(f"Best alpha: {best_alpha}\nMean Square Error: {min_error}")
# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))

# Plot calculated positions
ax.scatter(best_calc_coords[:, 0], best_calc_coords[:, 1], color='red', s=40, label='Calculated', marker='x')

# Plot true positions
ax.scatter(true_coords[:, 0], true_coords[:, 1], color='blue', s=40, alpha=0.5, label='Wold', marker='x')

# Test points
tennis_ball_img_pos = np.array([[380, 150]]).astype(np.float64)
print("Tennis ball calculated position:")
tennis_ball_world_pos = image_to_world(correct_barrel_distortion(tennis_ball_img_pos, best_alpha), best_H)
print(tennis_ball_world_pos)

ax.scatter(tennis_ball_world_pos[:, 0], tennis_ball_world_pos[:, 1], s=50, color='#cced00', label='Tennis Ball')
ax.set_xlabel('X World Coordinate')
ax.set_ylabel('Y World Coordinate')
ax.set_title('Calculated vs True World Coordinates of Tennis Balls (RANSAC)')
ax.grid(True)
ax.axis('equal')
ax.legend()

# Print the homography matrix
print("\nHomography Matrix:")
print(best_H)

plt.show()