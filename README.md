# Purpose
This package offers classes and methods for pose estimation on images. This can be done either with deep learning
based facial landmark detection or by detecting ArUco markers.

# Installation
Get the latest published version: `pip install headpose` \
or install directly from GitHub: `pip install git+https://github.com/OleBialas/headpose.git`
To use landmark detection you additionally have to install
[pytorch and torchvision](https://pytorch.org/get-started/locally/)

# Example
```python
import cv2
from headpose.detect import PoseEstimator

est = PoseEstimator()  #load the model
# take an image using the webcam (alternatively, you could load an image)
cam = cv2.VideoCapture(0)
for i in range(cv2.CAP_PROP_FRAME_COUNT):
    cam.grab()
ret, image = cam.retrieve()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cam.release()

est.detect_landmarks(image, plot=True)  # plot the result of landmark detection
roll, pitch, yawn = est.pose_from_image(image)  # estimate the head pose




```


# Sources & Further Reading
This [blog post](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/) nicely explained the concepts 
and mathematics behind pose estimation and this [tutorial](https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4)
walks through the single steps of detecting facial landmarks with pytorch



