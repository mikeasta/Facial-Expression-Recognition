import torch
from torch import nn, tensor

class FacialExpressionRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=10, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10, 
                out_channels=10, 
                kernel_size=3
            ),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            # Other layers
        )

    def forward(self, x: tensor.Tensor) -> torch.Tensor:
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        res = self.linear(x2)
        return res