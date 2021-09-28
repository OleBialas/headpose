from pathlib import Path
import time
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import Dataset
from headpose.dataset import FaceLandmarksDataset, Transforms, get_dlib_faces


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


def train(network, dataset, num_epochs, val_size=.1, batch_train=64, batch_val=8, learning_rate=0.0001, save=True):
    if not isinstance(network, nn.Module) and hasattr(network, "forward"):
        raise ValueError("Network must be a torch module with a `forward method")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.train()  # set network to "training mode"

    len_valid_set = int(val_size * len(dataset))
    len_train_set = len(dataset) - len_valid_set
    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))
    train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
    # shuffle and batch the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_val, shuffle=True, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

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
        print(f'Epoch: {epoch}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time() - start_time))

    if save:
        write_path = Path(__file__).parent/"model_weights.zip"
        torch.save(network.state_dict(), write_path)
        print(f"Saved the trained model to {write_path}")


if __name__ == "__main__":
    xml_file = get_dlib_faces()+"/labels_ibug_300W_train.xml"
    dataset = FaceLandmarksDataset(file=xml_file, transform=Transforms())
    network = ResNet()
    num_epochs = 1
    train(network, dataset, num_epochs, val_size=.1, batch_train=64, batch_val=8, learning_rate=0.0001)