import os

# import fire
import hydra
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from logger_selector import get_logger
from model_selector import get_model
from trainer import ImageClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    transformer = transforms.Compose(
        [
            transforms.Resize(
                (config["model"]["image_height"], config["model"]["image_width"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                config["model"]["image_mean"], config["model"]["image_std"]
            ),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data_loading"]["train_data_path"]), transform=transformer
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data_loading"]["val_data_path"]), transform=transformer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = get_model(config["model"])
    module = ImageClassifier(model, lr=config["training"]["lr"])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["model"]["model_local_path"],
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = get_logger(config["logging"])

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,  # to resolve warnings
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
