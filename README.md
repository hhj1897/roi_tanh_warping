# Face Warping Test
The script to test few different face warping method.

## Dependencies
* numpy
* opencv-python
* pytorch
* saic_vision.object_detector (or any face detector)

## How to Run
`python face_warping_test.py -v 0 -p 1 -r`

Command-line arguments:
```
-v VIDEO: Index of the webcam to use (start from 0)
-x WIDTH: Width of the warped frames (default=256)
-y HEIGHT: Height of the warped frames (default=256)
-p POLAR: Use tanh-polar warping (when set to 1) or 
          tanh-circular warping (when set to 2) instead of 
          normal tanh warping (when set to 0, which is the default value)
-r: To also show restored frames
```
