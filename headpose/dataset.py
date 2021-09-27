import cv2
import random
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import imutils
from math import *
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset


class Transforms:
    """ apply a series of transformations to the images to avoid overfitting. """
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks


class FaceLandmarksDataset(Dataset):

    def __init__(self, file, transform=None):
        self.root_dir = Path(file).parent
        tree = ET.parse(file)
        root = tree.getroot()
        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform

        for filename in root[2]:
            self.image_filenames.append(self.root_dir/filename.attrib['file'])

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(str(self.image_filenames[index]), 0)
        landmarks = self.landmarks[index]
        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5  # zero center the landmarks

        return image, landmarks

    def plot_sample(self):
        """
        Plot a random sample from the images in the dataset with it's landmarks
        """
        index = np.random.randint(0, len(self.image_filenames), 1)[0]
        image = Image.open(self.image_filenames[index])
        landmarks = self.landmarks[index]
        crop = self.crops[index]
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker=".")
        width, height = int(crop["width"]), int(crop["height"])
        x, y = int(crop["left"]), int(crop["top"])
        plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))
        plt.axis("off")
        plt.show()


def get_dlib_faces(path=None):
    """
    Return the path to the folder containing dlib's facial landmark dataset and download it if necessary.
    Arguments:
        path (None | str): path to look for the dataset or download it to. If None this defauls to 'home/dlib_data'
    Returns:
        (str): absolute path to the folder containg the face landmark dataset.
    """
    import urllib.request
    import tarfile
    if path is None:
        path = Path.home()/"dlib_data"
    elif not Path(path).exists():
        raise ValueError("The specified path does not exist!")
    if not path.exists():
        path.mkdir()
    if not (path/"ibug_300W_large_face_landmark_dataset").exists():
        print(f"downloading the data to {path}. This may take a while ...")
        tar = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
        stream = urllib.request.urlopen(tar)
        tar = tarfile.open(fileobj=stream, mode="r|gz")
        tar.extractall(path=path)
    return str(path/"ibug_300W_large_face_landmark_dataset")