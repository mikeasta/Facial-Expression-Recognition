import os
import torch
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from engine import test_step, load_model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import FacialExpressionRecognitionModel

# Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare image transforms
test_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor()
])

# Get data folder
current_dir_path = Path.cwd()
test_data_path = current_dir_path/ "data" / "test"

# Load test data
test_dataset = ImageFolder(
    root=test_data_path, 
    transform=test_transforms
)

# Prepare iterable
test_dataloader = DataLoader(dataset=test_dataset)

# Initialize model
model = FacialExpressionRecognitionModel(
    image_height=96,
    image_width=96,
    in_channels=3,
    num_classes=7
).to(device)

# Get model weights folder
model_name = "fer_model_epoch_10.pth"
model_path = current_dir_path / "models" / model_name

# Load weights
model = load_model(model=model, model_path=model_path)

# Test model on one epoch
loss_fn = torch.nn.CrossEntropyLoss()
test_loss, test_acc = test_step(model, test_dataloader, loss_fn)
print(f"{test_loss=}, {test_acc=}")