import os

import pytorch_lightning as pl
from classifiers import ConvClassifier
from model import ImageClassifier

from data import MyDataModule


def main():
    pl.seed_everything(42)
    # torch.set_float32_matmul_precision("medium")
    dm = MyDataModule(
        train_path="../../data/train_11k",
        val_path="../../data/train_11k",
        batch_size=32,
        num_workers=6,
    )
    model = ImageClassifier(ConvClassifier(num_classes=2), 1e-3)

    loggers = [
        # pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        # pl.loggers.MLFlowLogger(
        #     experiment_name=cfg.artifacts.experiment_name,
        #     tracking_uri="file:./.logs/my-mlflow-logs",
        # ),
        # pl.loggers.TensorBoardLogger(
        #     "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
        # ),
        pl.loggers.WandbLogger(project="mlops-logging-demo", name="conv-classifier"),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join("../../checkpoints", "conv-classifier"),
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=5,
            every_n_train_steps=None,
            every_n_epochs=1,
        )
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        # devices=0,
        precision=32,
        max_epochs=10,
        # max_steps - alternative
        accumulate_grad_batches=1,
        val_check_interval=1.0,
        overfit_batches=0,
        num_sanity_val_steps=4,
        deterministic=False,
        benchmark=False,
        gradient_clip_val=2.0,
        profiler=None,
        log_every_n_steps=1,
        detect_anomaly=False,
        enable_checkpointing=True,
        logger=loggers,
        callbacks=callbacks,
    )

    # Batch size tuner
    # tuner = pl.tuner.Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
