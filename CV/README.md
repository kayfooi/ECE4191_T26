## COMPUTER VISION NOTES

#### Calibration
- [camera_calibrate.py](camera_calibrate.py) calculates a homography matrix that transforms image coordinates into real-world coordinates (relative to a camera positioned at (0, 0, 0))
  - takes [calibration_img.png](calibration_img.png) and [calibration.csv](calibration.csv) as input
  - plots a lot of stuff
  - [calibration.csv](calibration.csv) is currently created manually (we should only need to do this a single time once the robot is constructed and the camera is in a fixed position relative to the ground)

#### Distortion
- Minor barrel distortion appears present
- No major effect on homography accuracy, especially for pixels in the centre of the image
- Attempted to use [checkerboard](./checkerboard/) images to [correct distortion with OpenCV](https://learnopencv.com/camera-calibration-using-opencv/) with minimal success

#### Object Identification
- Will probably use a combination of colour masking, edge detection, maybe k-means clustering?

#### Test case generation
- Blender used to generate many images to test the performance of various CV algorithms 
- Simulates barrel distortion from camera
- Various lighting / shadow conditions
- Various amouts of noise / motion blur

#### TODO
- [x] Geometry
- [ ] Tennis ball identification
- [ ] Test case generation


