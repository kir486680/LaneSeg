from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(1):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0)
        #print("Data size",data.shape)
        #data = data.squeeze(0)
        #plt.imshow(data.permute(1, 2, 0)  , interpolation='none')
        #plt.show()
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample


