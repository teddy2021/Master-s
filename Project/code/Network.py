import UNet.Unet as Unet
from CustomData import CustomDataset as CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opti
import math

class Network():
    def __init__(self, path, image_count, image_size=512, batch=64):
        self.image_count = image_count
        self.data = DataLoader(CustomDataset(path, "screenshots"), shuffle=True, batch_size=batch)
        self.Unets = [Unet.Unet(
            [2**i for i in range(int(math.log2(image_size))) if 2**i <= image_size],
           [2**i for i in range(int(math.log2(image_size)), 4, -1)]) for x in range(image_count)]
        self.loss = nn.MSELoss()
        self.optimizer = [opti.Adam(unet.parameters()) for unet in self.Unets]

    def forward(self):
        for x,_ in self.data:
            for y in x:



net = Network("./images.csv", 14)
net.forward()

