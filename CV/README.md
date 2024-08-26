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

#### Testing
- Put all images in a test set in a single folder
- If you want to automatically test accuracy, create a cases.json file with the structure:
- [
    {
      "caseID": str,
      "cam_heading": int,
      "cam_location": [x,y,z],
      "balls": [{
          "ball_id": str,
          "world": [x,y,z],
          "image": [u,v],
          "in_bounds": bool
      }]
    }
  ]
- *^^ [image_labeller.py](test_imgs/image_labeller.py) can do this for you*
- Import ball detections algorithms in [test_cv.py](test_cv.py)
- Run test_cv (make sure you are in CV folder)

#### TODO
- [x] Geometry
- [ ] Tennis ball identification
- [x] Test case generation


