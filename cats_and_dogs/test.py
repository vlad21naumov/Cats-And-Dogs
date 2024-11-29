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
    N_CHANNELS,
    NUM_CLASSES,
    NUM_WORKERS,
    SIZE_H,
    SIZE_W,
)
from model import SimpleClassifier
from trainer import ImageClassifier


def main(test_dir: str, checkpoint_name: str) -> None:
    transformer = transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        f"{DATA_PATH}/{test_dir}", transform=transformer
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = SimpleClassifier(N_CHANNELS, SIZE_H, SIZE_W, NUM_CLASSES)
    module = ImageClassifier.load_from_checkpoint(
        f"{MODELS_PATH}/{checkpoint_name}", model=model, lr=LR
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    results = trainer.test(module, dataloaders=test_loader)
    print(results)


if __name__ == "__main__":
    fire.Fire(main)
