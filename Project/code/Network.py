import UNet.Unet as Unet
from CustomData import ModelDataset as MData
from CustomData import SegDataset as SData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opti
import torch
import math

class ModelNetwork():
    def __init__(self, path, node_count=14, image_size=512, batch=64):
        self.data = DataLoader(MData(path, "screenshots", ), shuffle=True, batch_size=batch)
        channels = []
        for x in range(2, int(math.log2(image_size))):
            if(2 ** x <= image_size):
                channels.append(2**x)
        in_chs = [ i for i in channels ]
        channels.reverse()
        out_chs = channels[:-1]

        self.unet = Unet.Unet(in_chs, out_chs, retain_size = False)
        self.loss = nn.MSELoss()
        self.optimizer = opti.Adam(self.unet.parameters())

    def forward(self):
        error = []
        for x, tx in self.data:
            for y in range(len(tx)-1):
                datapoint = x[y]
                label = tx[y]
                res = self.unet(datapoint)
                print(datapoint.shape)
                print(res.shape)
                print(label.shape, '\n')
                error.append(self.loss(res, label))
        return torch.mean(error)

class SegNetwork():

    def __init__(self, path, node_count=14, image_size=512, batch=64):
        self.data = DataLoader(SData(path, "screenshots", ), shuffle=True, batch_size=batch)
        channels = []
        for x in range(2, int(math.log2(image_size))):
            if(2 ** x <= image_size):
                channels.append(2**x)
        in_chs = [ i for i in channels ]
        channels.reverse()
        out_chs = channels[:-1]

        self.unet = Unet.Unet(in_chs, out_chs, retain_size = True)
        self.loss = nn.MSELoss()
        self.optimizer = opti.Adam(self.unet.parameters())

    def forward(self):
        loss = []
        res = []
        for x, lx in self.data:
            print(lx.shape)
            result = self.unet(x)
            loss.append(self.loss(result, lx))
            res.append(result)
        return res, torch.mean(loss)
