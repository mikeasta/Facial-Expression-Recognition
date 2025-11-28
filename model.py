import torch
from typing import Tuple, List
from torch import nn

class FacialExpressionRecognitionModel(nn.Module):
    def __init__(
            self, 
            image_height: int, 
            image_width: int,
            in_channels: int,
            num_classes: int,
        ):

        self.image_height = image_height
        self.image_width = image_width

        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=32, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.image_height /= 2
        self.image_width /= 2
        
        in_features = int(self.image_width * self.image_height * 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features,
                out_features=128
            ),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)
        res = self.classifier(x4)
        return res