from typing import Any, Dict

import torch

from conv_model import ConvClassifier
from model import SimpleClassifier


def get_model(model_conf: Dict[str, Any]) -> torch.nn.Module:
    """Select model to run experiments"""
    label = model_conf["label"]
    if label == "linear":
        return SimpleClassifier(
            model_conf["num_channels"],
            model_conf["image_height"],
            model_conf["image_width"],
            model_conf["num_classes"],
        )
    if label == "conv":
        return ConvClassifier(model_conf["num_classes"])
    else:
        raise ValueError(f"There is no such model with label {label}")
