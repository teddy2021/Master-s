import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os
import binvox


class ModelDataset(Dataset):
    def __init__(self, annotations_file, img_dir,
    transform=None, target_transform=None):
        with open(annotations_file) as f:
            directories = f.readline().strip().split(',')
        datapoints = []
        for datafile in directories:
            datapoints.append(
                pd.read_csv(datafile, header=0)
            )
        self.img_labels=pd.concat(datapoints, ignore_index=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        images = [name \
                  for name in self.img_labels.iloc[idx][1:-2]]

        img_paths = [ os.path.join( self.img_labels.iloc[idx, 0], image)\
                     for image in images
            ]

        images = [read_image(img_paths[i]).float() for i in range(len(img_paths) - 1)]
        label = torch.from_numpy(
            binvox.Binvox.read(
                self.img_labels.iloc[idx, -1],
                'dense'
            ).numpy()
        )
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.stack([image for image in images]), label

class SegDataset(Dataset):
    def __init__(self, annotations_file, img_dir,
        transform=None, target_transform=None):
        with open(annotations_file) as f:
            directories = f.readline().strip().split(',')
        datapoints = []
        for datafile in directories:
            datapoints.append(
                pd.read_csv(datafile, header=0)
            )
        self.img_labels=pd.concat(datapoints, ignore_index=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0])
        label = read_image(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label
