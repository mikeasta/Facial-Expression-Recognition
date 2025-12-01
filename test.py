import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import FacialExpressionRecognitionModel
from engine import load_model, test_batch_and_plot_results

# Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare image transforms
test_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.Grayscale(num_output_channels=1),
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
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=16
)

# Initialize model
model = FacialExpressionRecognitionModel(
    image_height=96,
    image_width=96,
    in_channels=1,
    num_classes=7
)#.to(device)

# Get model weights folder
model_name = "fer_model_epoch_9.pth"
model_path = current_dir_path / "models" / model_name

# Load weights
load_model(model=model, model_path=model_path)

# Test model on one epoch
loss_fn = torch.nn.CrossEntropyLoss()
classes = test_dataloader.dataset.classes

test_batch_and_plot_results(
    model=model,
    batch=next(iter(test_dataloader)),
    loss_fn=loss_fn,
    device=device,
    classes=classes
)