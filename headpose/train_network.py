from pathlib import Path
import argparse
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from dataset import FaceLandmarksDataset, Transforms, get_dlib_faces
from detect import ResNet
root = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-n", type=int, default=1, help="Number of training and validation epochs.")
parser.add_argument("--valsize", "-v", type=float, default=0.1, help="percentage of the data used for validation.")
parser.add_argument("--dataset", "-d", help="Path to a .xml file containing image filenames and landmark coordinates.")
parser.add_argument("--weights", "-w", help="Path to .zip file with network weights for initialization.")
parser.add_argument("--learnrate", "-l", type=float, default=0.0001, help="Networks learning rate.")
parser.add_argument("--batchsizetrain", "-bt", type=int, default=64, help="Batch size for training")
parser.add_argument("--batchsizeval", "-bv", type=int, default=8, help="Batch size for validation")
parser.add_argument("--outfolder", "-o", help="Folder to which weights and loss record are written.")
parser.add_argument("--plot", default=False, type=bool, help="If True, plot detected landmarks after every epoch.")
args = parser.parse_args()

network = ResNet()  # initialize the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if cuda is available

if args.dataset is not None:  # use the specified
    dataset = FaceLandmarksDataset(file=args.dataset, transform=Transforms())
else:  # use the dlib dataset
    dataset = FaceLandmarksDataset(file=get_dlib_faces()+"/labels_ibug_300W_train.xml", transform=Transforms())
if args.weights is not None:  # load weights
    network.load_state_dict(torch.load(args.weights, map_location=device))
if args.outfolder is not None:
    out_folder = Path(args.outfolder)
else:
    out_folder = Path(__file__).absolute().parent

if args.plot is True:  # use the first image in the database to visualize the training process
    face_cascade = cv2.CascadeClassifier(str(root / "haarcascade_frontalface_default.xml"))
    image = cv2.cvtColor(cv2.imread(str(dataset.image_filenames[0])), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    if len(faces) != 1:
        raise ValueError("There must be exactly one face in the image!")
    for (x, y, w, h) in faces:
        image_crop = image[y:y + h, x:x + w]
        image_crop = TF.resize(Image.fromarray(image_crop), size=[224, 224])
        image_crop = TF.to_tensor(image_crop)
        image_crop = TF.normalize(image_crop, [0.5], [0.5])

network.to(device)
network.train()  # set network to "training mode"

len_valid_set = int(args.valsize * len(dataset))
len_train_set = len(dataset) - len_valid_set
print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))
train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsizetrain, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchsizeval, shuffle=True, num_workers=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=args.learnrate)

start_time = time.time()
loss_record = np.zeros((2, args.epochs))
for epoch in range(args.epochs):

    loss_train, loss_valid, running_loss = 0, 0, 0

    for step in range(1, len(train_loader) + 1):
        images, landmarks = next(iter(train_loader))
        images = images.to(device)
        landmarks = landmarks.view(landmarks.size(0), -1).to(device)

        predictions = network(images)

        optimizer.zero_grad()  # clear all the gradients before calculating them
        loss_train_step = criterion(predictions, landmarks)  # find the loss for the current step
        loss_train_step.backward()  # calculate the gradients
        optimizer.step()  # update the parameters

        # calculate loss and print running loss
        loss_train += loss_train_step.item()
        running_loss = loss_train / step
        print(f"training steps {step} of {len(train_loader)}. Loss: {running_loss:.5f}")

    network.eval()  # set the network to evaluation mode
    with torch.no_grad():

        for step in range(1, len(valid_loader) + 1):
            images, landmarks = next(iter(valid_loader))

            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)

            predictions = network(images)

            loss_valid_step = criterion(predictions, landmarks)  # find the loss for the current step

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid / step
            print(f"step {step} of {len(train_loader)}. Loss: {running_loss:.5f}")

    # divide by batch number to get the loss for the whole epoch
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    loss_record[0, epoch], loss_record[1, epoch] = loss_train, loss_valid
    print(f'Epoch: {epoch+1}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')

    if args.plot is True:
        fig, ax = plt.subplots()
        predictions = network(image_crop.unsqueeze(0).to(device))
        predictions = (predictions.view(68, 2).detach().cpu().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        ax.imshow(image, cmap="gray")
        ax.scatter(predictions[:, 0], predictions[:, 1], s=6)
        fig.savefig(out_folder/f"image{epoch}.jpg")

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time() - start_time))
torch.save(network.state_dict(), out_folder/"model_weights.zip")
np.save(str(out_folder/"loss_record.npy"), loss_record)
print(f"Saved the trained model to {out_folder}")

