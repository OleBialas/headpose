from headpose.detect import PoseEstimator
from pathlib import Path
import cv2
est = PoseEstimator()

def test_landmark_detection():
    image_folder = Path(__file__).parent/"test_images"
    for image in image_folder.glob("pose*"):
        image = cv2.imread(str(image))
        landmarks = est.detect_landmarks(image)
        assert landmarks.shape == (68, 2)


def test_pose_from_image():
    image_folder = Path(__file__).parent/"test_images"
    for image_left, image_right in zip(image_folder.glob("*left*"), image_folder.glob("*right*")):
        image_left = cv2.imread(str(image_left))
        image_right = cv2.imread(str(image_right))
        assert est.pose_from_image(image_left)[0] < est.pose_from_image(image_right)[0]
    for image_up, image_down in zip(image_folder.glob("*up*"), image_folder.glob("*down*")):
        image_up = cv2.imread(str(image_up))
        image_down = cv2.imread(str(image_down))
        assert est.pose_from_image(image_up)[0] > est.pose_from_image(image_down)[0]
