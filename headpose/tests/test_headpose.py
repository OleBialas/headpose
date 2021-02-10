from headpose import DIR, PoseEstimator
import os
import cv2
import random

images = os.listdir(DIR/"tests"/"test_images")
est = PoseEstimator()


def test_pose_from_image():
    thresholds = [.99, .9, .8, .5]
    for thresh in thresholds:
        est.threshold = thresh
        for i, image in enumerate(images):
            image_data = cv2.imread(str(DIR/"tests"/"test_images"/image))
            roll, pitch, yaw = est.pose_from_image(image_data)


def test_plotting():
    est.threshold = .5
    image = random.choice(images)
    image_data = cv2.imread(str(DIR / "tests" / "test_images" / image))
    est.plot_face_detection_marks(image_data, show=False)
