import os
import torch
from pathlib import Path
from typing import Tuple
from torchvision import transforms
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
model_name = "fer_model_epoch_0.pth"
model_path = current_dir_path / "models" / model_name

# Load weights
model.load_state_dict(torch.load(f=model_path))

# Single epoch step
def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module
) -> Tuple[int, int]:
    """
    Returns test loss and acc
    """
    # Model inference
    model.eval()
    test_loss, test_acc = 0, 0,
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device).to(memory_format=torch.channels_last), y.to(device)
            y_pred_logits = model(X)
            test_loss += loss_fn(y_pred_logits, y)

            y_pred_labels = torch.argmax(y_pred_logits)
            test_acc += (y_pred_labels == y).sum().item() / len(y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return (test_loss, test_acc)

loss_fn = torch.nn.CrossEntropyLoss()

test_loss, test_acc = test_step(model, test_dataloader, loss_fn)

print(f"{test_loss=}, {test_acc=}")