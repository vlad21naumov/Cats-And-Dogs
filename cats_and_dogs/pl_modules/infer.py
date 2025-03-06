import numpy as np
import pytorch_lightning as pl
from model import ImageClassifier

from data import MyDataModule


def main():
    dm = MyDataModule(
        train_path="../../data/train_11k",
        val_path="../../data/val",
        test_path="../../data/test_labeled",
        batch_size=64,
        num_workers=6,
    )

    model = ImageClassifier.load_from_checkpoint(
        "../../checkpoints/conv-classifier/epoch=09-val_loss=0.4654.ckpt"
    )
    trainer = pl.Trainer(accelerator="cpu")

    accs = trainer.predict(model, datamodule=dm)
    print(f"Test accuracy: {np.mean(accs):.2f}")


if __name__ == "__main__":
    main()
