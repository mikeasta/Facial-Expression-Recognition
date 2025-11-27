import torch
from typing import Tuple, List
from torch import nn

class FacialExpressionRecognitionModel(nn.Module):
    def __init__(
            self, 
            image_height: int, 
            image_width: int,
            in_channels: int,
            hidden_channels: int,
            hidden_features: int,
            out_features: int,
        ):

        self.image_height = image_height
        self.image_width = image_width

        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
             nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2
        
        in_features = int(self.image_width * self.image_height * hidden_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features,
                out_features=hidden_features
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_features,
                out_features=out_features
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        res = self.classifier(x2)
        return res