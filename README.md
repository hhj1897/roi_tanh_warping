# ROI Tanh Warping
Differentiable implementation of some ROI-tanh warping methods.

## Dependencies
* pytorch
* \[optional\] numpy
* \[optional\] opencv-python
* \[optional\] saic_vision.object_detector (or any face detector)

## How to Install
`pip install -e .`

## How to Test
`python face_warping_test.py -v 0 -p 1 -r -t`

Command-line arguments:
```
-v VIDEO: Index of the webcam to use (start from 0)
-x WIDTH: Width of the warped frames (default=256)
-y HEIGHT: Height of the warped frames (default=256)
-p POLAR: Use tanh-polar warping (when set to 1) or 
          tanh-circular warping (when set to 2) instead of 
          normal tanh warping (when set to 0, default)
-o OFFSET: Angular offset in degrees (default=0)
-r: To also show restored frames
-c: To also compare with OpenCV-based reference implementation
-s: To use square-shaped detection box
-n: To use nearest-neighbour interpolation during restoration
-k: Keep aspect ratio in tanh-polar or tanh-circular warping
```
