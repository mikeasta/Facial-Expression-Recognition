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

current_dir_path = Path(os.getcwd())

# Import datasets
train_path = os.path.join(current_dir_path, "data\\train")
test_path = os.path.join(current_dir_path, "data\\test")

train_data = datasets.ImageFolder(root=train_path, transform=train_transforms, target_transform=None)
test_data = datasets.ImageFolder(root=test_path, transform=test_transforms)

# Create dataloaders
NUM_WORKERS = 0
BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)

# Training and testing functions
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> Tuple[int, int]:
    """
    Returns train loss and train accuracy
    """
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).to(memory_format=torch.channels_last), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate class
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (y_pred_class==y).sum().item()/len(y_pred)
        # print(f"{batch=}, {loss.item()=}, {acc=}")
        train_acc += acc
    
    # Adjuct metric to get average values
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return (train_loss, train_acc)

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module
) -> Tuple[int, int]:
    """
    Returns test loss and test accuracy
    """
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device).to(memory_format=torch.channels_last), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return (test_loss, test_acc)

def save_model(
        model: torch.nn.Module,
        directory: Path | str,
        filename: Path | str = None,
        epoch: int | str = "none"
) -> None:
    if not filename:
        filename = f"fer_model_epoch_{epoch}.pth"
    
    save_path = os.path.join(directory, filename)
    torch.save(obj=model.state_dict(), f=save_path)


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
            optimizer=optimizer
        )

        test_loss, test_acc = test_step(
            model=model, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn,
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
            save_model(model=model, directory=models_path, epoch=epoch)

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