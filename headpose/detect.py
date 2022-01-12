from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
try:
   import torch
   import torch.nn as nn
   import torchvision.transforms.functional as TF
   from torchvision import models
except ModuleNotFoundError:
    torch=False
root = Path(__file__).parent
face_cascade = cv2.CascadeClassifier(str(root / "haarcascade_frontalface_default.xml"))

model_points = np.array([[0.0, 0.0, 0.0],  # Tip of the nose [30]
                        [0.0, -330.0, -65.0],  # Chin [8]
                        [-225.0, 170.0, -135.0],  # Left corner of the left eye  [45]
                        [225.0, 170.0, -135.0],  # Right corner of the right eye [36]
                        [-150.0, -150.0, -125.0],  # Left corner of the mouth [54]
                        [150.0, -150.0, -125.0]])  # Right corner of the mouth [48]


class PoseEstimator:
    def __init__(self, method, weights=None):
        self.method = method
        if method == "landmarks":
            if torch is False:
                raise ImportError("Landmark estimation requires pytorch and torchvision!")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ResNet()
            if weights is None:  # use the pre-trained model from the repo
                weights = get_model_weights()
            self.model.load_state_dict(torch.load(weights, map_location=device))
            self.model.eval()
        elif method == "aruco":
            self.arucodict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        else:
            raise ValueError("Possible methods are 'landmarks' or 'aruco'!")

    def detect_landmarks(self, image, plot=False):
        """
        Pass an image through the trained neural network to detect facial landmarks.
        Arguments:
            image (array-like): An image containing exactly one face for which the landmarks are detected
        Returns:
            (numpy.ndarray | matplotlib.figure.Figure): a array with the x and y coordinates of 68 facial landmarks or
                the figure with the image and the detected landmarks.
        """
        if image.ndim == 3:  # convert color to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, 1.1, 5)
        if len(faces) != 1:
            raise ValueError("There must be exactly one face in the image!")
        for (x, y, w, h) in faces:
            image_crop = image[y:y + h, x:x + w]
            image_crop = TF.resize(Image.fromarray(image_crop), size=[224, 224])
            image_crop = TF.to_tensor(image_crop)
            image_crop = TF.normalize(image_crop, [0.5], [0.5])
        with torch.no_grad():
            landmarks = self.model(image_crop.unsqueeze(0))
        landmarks = (landmarks.view(68, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.scatter(landmarks[:, 0], landmarks[:, 1], s=5)
            return fig
        else:
            return landmarks

    def pose_from_image(self, image, return_matrices=False):
        # approximate camera coefficients # TODO: add option to calibrate
        size = image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        if self.method == "landmarks":
            landmarks = self.detect_landmarks(image)
            image_points = landmarks[[30, 8, 45, 36, 54, 48]]  # pick points corresponding to the model
            success, rotation_vec, translation_vec = \
                cv2.solvePnP(model_points, image_points, camera_matrix, distortion_coefficients)
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
            angles[0, 0] = angles[0, 0] * -1
        elif self.method == "aruco":
            corners, ids, rejected = cv2.aruco.detectMarkers(image, self.arucodict, parameters=self.params)
            if len(corners) != 1:
                raise ValueError("There must be exactly one marker in the image")  # TODO: support multiple markers
            rotation_vec, translation_vec, _objPoints = \
                cv2.aruco.estimatePoseSingleMarkers(corners, .05, camera_matrix, distortion_coefficients)
            rotation_mat = - cv2.Rodrigues(rotation_vec)[0]
            pose_mat = cv2.hconcat(rotation_mat, translation_vec.T)
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
            angles[1, 0] = np.degrees(np.arcsin(np.sin(np.radians(angles[1, 0]))))
            angles[0, 0] = angles[0, 0] * -1
        if return_matrices:
            return angles[1, 0], angles[0, 0], angles[2, 0], \
                   camera_matrix, distortion_coefficients, rotation_vec, translation_vec
        else:
            return angles[1, 0], angles[0, 0], angles[2, 0]  # azimuth, elevation, tilt


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


def get_model_weights():
    """
    Return the path to the folder containing the weights of the pre-trained model or download them if necessary.
    Returns:
        (str): absolute path to the folder containing the face landmark dataset.
    """
    import requests
    model_weights = root / "model_weights.zip"
    if not model_weights.exists():
        print("downloading model weights, this may take a while ...")
        url='https://raw.githubusercontent.com/OleBialas/headpose/main/headpose/model_weights.zip'
        r = requests.get("https://raw.githubusercontent.com/OleBialas/headpose/main/headpose/model_weights.zip")
        with open(model_weights,'wb') as output_file:
            output_file.write(r.content)
    return str(model_weights)
