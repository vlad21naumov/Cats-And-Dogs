import os

import fire
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import (
    BATCH_SIZE,
    DATA_PATH,
    IMAGE_MEAN,
    IMAGE_STD,
    LR,
    MODELS_PATH,
    N_EPOCHS,
    NUM_WORKERS,
    SIZE_H,
    SIZE_W,
)
from model_selector import get_model
from trainer import ImageClassifier


def main(model_label):
    transformer = transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "train_11k"), transform=transformer
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "val"), transform=transformer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = get_model(model_label)
    module = ImageClassifier(model, lr=LR)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=MODELS_PATH,
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    fire.Fire(main)
