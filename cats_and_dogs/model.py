import torch


class SimpleClassifier(torch.nn.Module):
    """Linear model for classification of images"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 96 * 96, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
