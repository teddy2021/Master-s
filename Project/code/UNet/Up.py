import torch.nn as nn
import torch
import torchvision
import UNet.Convoluter as Convoluter


class Up(nn.Module):

    def __init__(self, channels=[2, 1]):
        super().__init__()
        self.channels = channels
        self.up_convolutions = nn.ModuleList([
            nn.ConvTranspose2d(
                channels[i], channels[i+1]
            , 2, 2) for i in range(len(channels) - 1)
        ])
        self.up_steps = nn.ModuleList([
            Convoluter.Convoluter(
                channels[i], channels[i+1]
            ) for i in range(len(channels) - 1)
        ])

    def crop(self, left_in, x):
        _, _, H, W = x.shape
        return torchvision\
            .transforms\
            .CenterCrop(
                [H, W])(left_in)

    def forward(self, x, down_out):
        up_in = x
        for i in range(len(self.channels) - 1):
            left_in = down_out[i]
            res_up = self.up_convolutions[i](up_in)
            cropped = self.crop(left_in, res_up)
            inpt = torch.cat([res_up, cropped], dim=1)
            up_in = self.up_steps[i](inpt)
        return up_in
