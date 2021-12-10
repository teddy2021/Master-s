from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import pandas as pd
import os


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir,
        transform=None, target_transform=None):
        self.img_labels = pd.read_csv(
        annotations_file, names=['base_dir',
                                    'image0', 'image1',
                                    'image2', 'image3',
                                    'image4', 'image5',
                                    'image6', 'image7',
                                    'image8', 'image9',
                                    'image10', 'image11',
                                    'image12', 'image13',
                                    'model_file'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_paths = [os.path.join(
            self.img_labels.iloc[idx,0] + "/" + self.img_dir,
            self.img_labels.iloc[idx, i]) for i in range(1,14,1)]
        images = [read_image(img_paths[i]) for i in range(len(img_paths) - 1)]
        label = self.img_labels.iloc[idx, 15]
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            label = self.target_transform(label)
        return [image.float() for image in images] , label
