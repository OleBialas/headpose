from pathlib import Path
import cv2
import numpy as np
import logging
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
root = Path(__file__).parent
face_cascade = cv2.CascadeClassifier(str(root / "haarcascade_frontalface_default.xml"))

model_points = np.array([[0.0, 0.0, 0.0],  # Tip of the nose
                        [0.0, -330.0, -65.0],  # Chin
                        [-225.0, 170.0, -135.0],  # Left corner of the left eye
                        [225.0, 170.0, -135.0],  # Right corner of the right eye
                        [-150.0, -150.0, -125.0],  # Left corner of the mouth
                        [150.0, -150.0, -125.0]])  # Right corner of the mouth


class PoseEstimator:
    def __init__(self, method):
        if method == "landmarks":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ResNet()
            # TODO: check if the network is present, if not download it from the repo
            self.model.load_state_dict(torch.load(root / "model_weights.zip", map_location=device))
        elif method == "aruco":
            pass
        else:
            raise ValueError("Possible methods are 'landmarks' or 'aruco'!")

    def detect_landmarks(self, image):
        """
        Pass an image through the trained neural network to detect facial landmarks.
        Arguments:
            image (array-like): An image containing exactly one face for which the landmarks are detected
        Returns:
            (array): a array with the x and y coordinates of 68 facial landmarks
        """
        if image.ndim == 3:  # convert color to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        if len(faces) != 1:
            raise ValueError("There must be exactly one face in the image!")
        all_landmarks = []
        for (x, y, w, h) in faces:
            image = image[y:y + h, x:x + w]
            image = TF.resize(Image.fromarray(image), size=(224, 224))
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])
        with torch.no_grad():
            landmarks = self.model(image.unsqueeze(0))
        landmarks = (landmarks.view(68, 2).detach().numpy() + 0.5) * np.array([w, h]) + np.array([x, y])
        return landmarks

    def pose_from_image(self, image):
        size = image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")

        faceboxes = self.extract_cnn_facebox(image)
        if len(faceboxes) > 1:
            logging.warning("There is more than one face in the image!")
            return None, None
        elif len(faceboxes) == 0:
            logging.warning("No face detected!")
            return None, None
        else:
            facebox = faceboxes[0]
            face_img = image[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (128, 128))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = self.detect_marks([face_img])
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            shape = marks.astype(np.uint)
            image_pts = np.float32([shape[17], shape[21], shape[22], shape[26],
                                    shape[36], shape[39], shape[42], shape[45],
                                    shape[31], shape[35], shape[48], shape[54],
                                    shape[57], shape[8]])
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vec, translation_vec) = \
                cv2.solvePnP(self.model_points, image_pts, camera_matrix,
                             dist_coeffs)

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
            angles[0, 0] = angles[0, 0] * -1

            return angles[1, 0], angles[0, 0], angles[2, 0]  # roll, pitch, yaw


class ResNet(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        # set input channel to one so the network accepts grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # number of output features --> x,y coordinates of the 68 landmarks
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x