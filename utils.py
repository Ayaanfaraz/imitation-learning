"""
Handles dataset organization for dataloader.
Additionally used as a utility to check model accuracy.
"""

import csv
import os
import torch

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarlaAliDataset(Dataset):
    """
    Classification data from CarlaAliDataset.
    Represented as tuples of 3 x 150 x 200 images and their vectors of data/labels
    """
    def __init__(self, dataset_path):

        self.csv_tuples = []
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Extract data from csv.
        labels_path = os.path.join(dataset_path, "labels.csv")
        with open(labels_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                self.csv_tuples.append((row[0], row[1], row[2], row[3], row[4])) 
                # 0 is rgb fname, 1 is sem name, 2 is data/labels vector

        # Cut out the csv headers from extracted data.
        self.csv_tuples = self.csv_tuples[1:]


    def __len__(self):
        """
        Your code here
        returns length of dataset.
        """
        return len(self.csv_tuples)


    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        # All pairs of image and label are added to csv_tuples string list.
        # Grab the data vector from 2nd index, get only labels and change to floats
        data = [float(self.csv_tuples[idx][2]),float(self.csv_tuples[idx][3]), float(self.csv_tuples[idx][4])]

        border = (0, 150, 0, 0) # cut 0 from left, 30 from top, right, bottom

        # Rgb image as a tensor
        rgb_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][0]))
        rgb_image = ImageOps.crop(rgb_image, border)
        rgb_tensor = self.transform(rgb_image)

        # Sem image as an input tensor
        sem_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][0]))
        sem_image = ImageOps.crop(sem_image, border)
        sem_tensor = self.transform(sem_image)

        return rgb_tensor, sem_tensor, torch.tensor(data)


def load_data(dataset_path, num_workers=0, batch_size=128):
    """
    Driver function to create dataset and return constructed dataloader.
    """
    dataset = CarlaAliDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(yhat, y): # yhat is not labels it is prediction
    """
    Returns accuracy between true labels and predictions.
    """
    # outputs_idx = outputs.max(1)[1].type_as(labels)
    # return outputs_idx.eq(labels).float().mean()

    def equals(y_tensor, yhat_tensor):
        if round((y_tensor.item()),2) == round(yhat_tensor.item(),2):
            return 1
        else:
            return 0

    cor_steer = cor_throttle = cor_brake = 0
    for j in range(len(y)):
        cor_steer += equals(y[j,0],yhat[j,0])
        cor_throttle += equals(y[j,1],yhat[j,1])
        cor_brake += equals(y[j,2],yhat[j,2])

    accuracy_steer = round(cor_steer/128, 3)
    accuracy_throttle = round(cor_throttle/128, 3)
    accuracy_brake = round(cor_brake/128, 3)
    accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 3)
    return accuracy_avg
