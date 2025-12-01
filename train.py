import os
import torch
from datetime import datetime as dt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from model import FacialExpressionRecognitionModel
from typing import Tuple
from timeit import default_timer as timer 
from tqdm import tqdm
from torchsummary import summary
from engine import train_step, test_step, save_model

# Device-agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image transforms
train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])



# Import datasets
current_dir_path = Path(os.getcwd())
train_path = os.path.join(current_dir_path, "data\\train")
test_path = os.path.join(current_dir_path, "data\\test")

train_data = datasets.ImageFolder(root=train_path, transform=train_transforms, target_transform=None)
test_data = datasets.ImageFolder(root=test_path, transform=test_transforms)

# Create dataloaders
num_workers = 0
batch_size  = 32

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False
)

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int = 5
) -> torch.nn.Module:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Save path
    models_path = current_dir_path / "models"

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.3f} | "
            f"train_acc: {train_acc:.3f} | "
            f"test_loss: {test_loss:.3f} | "
            f"test_acc: {test_acc:.3f}"
        )

        # Each 10 epochs - save
        if (epoch + 1) % 10 == 0:
            save_model(
                model=model, 
                directory=models_path, 
                epoch=epoch
            )

        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    
    return results

# Train model
torch.manual_seed(42)
model = FacialExpressionRecognitionModel(
    image_height=96,
    image_width=96,
    in_channels=3,
    num_classes=7
)

# Load model
# model_path = current_dir_path / "models" / "fer_model_epoch_50.pth"
# model.load_state_dict(torch.load(f=model_path))
# model.to(device=device).to(memory_format=torch.channels_last)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    lr=1e-3, 
    params=model.parameters(), 
    momentum=0.9, 
    weight_decay=1e-4
)
epochs = 100

print(summary(model, (3, 96, 96)))
start_time = timer()
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs
)
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")