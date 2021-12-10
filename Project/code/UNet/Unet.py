import UNet.Up as Up
import UNet.Down as Down
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self, down_channels = [1, 2], up_channels = [2, 1], retain_size=True):
        super().__init__()
        self.down_arc = Down.Down(down_channels)
        self.up_arc = Up.Up(up_channels)
        self.retain_size = retain_size
        self.result = nn.Conv2d(up_channels[-1], 4, 1)

    def forward(self, x, out_size=(512,512)):
        down_features = self.down_arc(x)
        res = self.up_arc(down_features[::-1][0], down_features[::-1][1:])
        res = self.result(res)
        if self.retain_size:
            res = nn.functional.interpolate(res, out_size)
        return res


