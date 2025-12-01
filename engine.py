import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Training and testing functions
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str
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
        loss_fn: torch.nn.Module,
        device: str
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
    """
    Saves model to specific directory with specific name.
    If there is no filename in, sets default value
    """
    if not filename:
        filename = f"fer_model_epoch_{epoch}.pth"
    
    save_path = os.path.join(directory, filename)
    torch.save(obj=model.state_dict(), f=save_path)

def load_model(
        model: torch.nn.Module,
        model_path: Path | str
) -> None:
    """
    Loads model weights into model
    """
    model.load_state_dict(torch.load(f=model_path))

def create_dataloaders(
        data_path: Path | str,
        train_transforms: transforms,
        test_transforms: transforms,
        batch_size: int = 32,
        num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads data and creates dataloaders
    """
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_data = datasets.ImageFolder(root=train_path, transform=train_transforms, target_transform=None)
    test_data = datasets.ImageFolder(root=test_path, transform=test_transforms)

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

    return (train_dataloader, test_dataloader)

def test_batch_and_plot_results(
        model: torch.nn.Module,
        batch: torch.Tensor,
        loss_fn: torch.nn.Module,
        device: str,
        classes: List[str]
):
    """
    Tests model using single batch and draws prediction result
    """
    model.eval()
    X, y = batch
    with torch.inference_mode():
        X, y = X.to(device=device).to(memory_format=torch.channels_last), y.to(device=device)
        test_pred_logits = model(X)
        test_loss = loss_fn(test_pred_logits, y)
        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_loss /= len(y)

        # Draw predictions
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 9))
        for i in range(len(y)):
            ay = i // 4
            ax = i % 4
            axs[ay, ax].imshow(X.squeeze()[i], cmap="gray")
            axs[ay, ax].set_title(f"Predicted: {classes[test_pred_labels[i]]} | True: {classes[y[i]]}")
        plt.tight_layout()
        plt.show()

