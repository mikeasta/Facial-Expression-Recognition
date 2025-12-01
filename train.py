import os
import torch
import mlflow
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from timeit import default_timer as timer 
from torchvision import transforms
from model import FacialExpressionRecognitionModel
from engine import train_step, test_step, save_model, create_dataloaders

# Device-agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# MLFlow Experiment setup
mlflow.set_experiment("Facial Expression Reconition Custom Model")
mlflow.enable_system_metrics_logging()

# Image transforms
train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomCrop(size=(96, 96)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Create loaders
current_dir_path = Path(os.getcwd())
data_path = os.path.join(current_dir_path, "data")

num_workers = 0
batch_size  = 32

train_dataloader, test_dataloader = create_dataloaders(
    data_path=data_path,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
)

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int = 5
):
    """
    Base train pipeline
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch size": 32,
            "learning rate": 1e-3
        })

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
                    epoch=epoch+1
                )

            # Log metrics
            mlflow.log_metric(key="train loss", value=train_loss, step=epoch)
            mlflow.log_metric(key="train acc", value=train_acc, step=epoch)
            mlflow.log_metric(key="test loss", value=test_loss, step=epoch)
            mlflow.log_metric(key="test acc", value=test_acc, step=epoch)
        
        mlflow.pytorch.log_model(pytorch_model=model, name="model")
        mlflow.set_tags({
            "model type": "cnn",
            "dataset": "fer dataset",
            "framework": "pytorch"
        })

# Custom model
torch.manual_seed(42)
model = FacialExpressionRecognitionModel(
    image_height=96,
    image_width=96,
    in_channels=1,
    num_classes=7
)

epochs = 150
alpha=1e-3
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    lr=alpha, 
    params=model.parameters(), 
    momentum=0.9, 
    weight_decay=1e-4
)

print(
    summary(
        model, 
        input_size=(32, 1, 96, 96),
        row_settings=["var_names"],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20
    )
)

start_time = timer()
train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs
)
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")