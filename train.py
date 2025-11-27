import torch
from model import FacialExpressionRecognitionModel

device = "cuda" if torch.cuda.is_available() else "cpu"