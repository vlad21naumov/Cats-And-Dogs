import torch

from constants import N_CHANNELS, NUM_CLASSES, SIZE_H, SIZE_W
from conv_model import ConvClassifier
from model import SimpleClassifier


def get_model(label: str) -> torch.nn.Module:
    if label == "linear":
        return SimpleClassifier(N_CHANNELS, SIZE_H, SIZE_W, NUM_CLASSES)
    if label == "conv":
        return ConvClassifier(NUM_CLASSES)
    else:
        raise ValueError(f"There is no such model with label {label}")
