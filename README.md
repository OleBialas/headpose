# Purpose
The purpose of this package is to provide a simple API to estimate the head pose based on a single image of a face.\

# Installation
You can install the package via pip: `pip install headpose` \
All the dependencies should be installed automatically. Note that the trained models are only compatible with
tensorflow versions >2 and <2.4 (requires Python 3.5 - 3.8).

# Head Pose Estimation
```python
import cv2
from headpose import PoseEstimator

est = PoseEstimator()  #load the model
# take an image using the webcam (alternatively, you could load an image)
cam = cv2.VideoCapture(0)
for i in range(cv2.CAP_PROP_FRAME_COUNT):
    cam.grab()
ret, image = cam.retrieve()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cam.release()

roll, pitch, yawn = est.pose_from_image(image)  # estimate the head pose
est.plot_face_detection_marks(image)  # plot the image with the face detection marks




```


# Sources & Further Reading
The code is inspired by this 
[blog post](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a)
which does a good job explaining the steps which are performed to obtain the head pose. The pose estimation uses a
deep neural network - the pretrained models are taken from
[this github repo](https://github.com/vardanagarwal/Proctoring-AI).
If you want to understand the math behind pose estimation, check out this
[tutorial](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

