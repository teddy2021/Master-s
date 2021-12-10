import torch.nn as nn

class Convoluter(nn.Module):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.convolution1 = nn.Conv2d(inputs, outputs, 3)
        self.activation = nn.ReLU()
        self.convolution2 = nn.Conv2d(outputs, outputs, 3)

    def forward(self, x):
        return \
    self.activation(
            self.convolution2(
                self.activation(
                    self.convolution1(x)
                )
            )
        )
