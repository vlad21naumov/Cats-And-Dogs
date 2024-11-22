import torch


class SimpleClassifier(torch.nn.Module):
    def __init__(self, n_channels, size_h, size_w, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_channels * size_h * size_w, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
