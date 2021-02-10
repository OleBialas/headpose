from headpose import DIR, PoseEstimator
import os
import cv2
import numpy
import random

images = os.listdir(DIR/"tests"/"test_images")
est = PoseEstimator()


def test_pose_from_image():
    """ Get the pose from the saved test images and compare the result to the one saved in the text files. """
    thresholds = [.99, .9, .8, .5]
    for thresh in thresholds:
        est.threshold = thresh
        test_image_angles = \
            numpy.loadtxt(DIR/"tests"/f"test_image_angles_threshold_{int(thresh*100)}.txt")
        for i, image in enumerate(images):
            image_data = cv2.imread(str(DIR/"tests"/"test_images"/image))
            roll, pitch, yaw = est.pose_from_image(image_data)
            assert roll == test_image_angles[i, 0]
            assert pitch == test_image_angles[i, 1]
            assert yaw == test_image_angles[i, 2]


def test_plotting():
    est.threshold = .5
    image = random.choice(images)
    image_data = cv2.imread(str(DIR / "tests" / "test_images" / image))
    est.plot_face_detection_marks(image_data, show=False)
