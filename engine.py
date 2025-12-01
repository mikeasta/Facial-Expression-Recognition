import os
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader

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
) -> torch.nn.Module:
    """
    Loads model weights into model
    """
    return model.load_state_dict(torch.load(f=model_path))