import torch.nn as nn
import UNet.Convoluter as Convoluter

class Down(nn.Module):

    def __init__(self, channels=[1, 2]):
        super().__init__()
        self.down_steps = nn.ModuleList([
                Convoluter.Convoluter(
                    channels[i], channels[i+1]
            ) for i in range(len(channels) - 1)
            ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        inpt = x
        for step in self.down_steps:
            res = step(inpt)
            inpt = self.pool(res)
            features.append(res)
        return features
