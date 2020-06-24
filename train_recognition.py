import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser

import pytorch_lightning as pl

from plates.models import PlatesRecognition


def main(args):
    # Создаем модель
    model = PlatesRecognition(args)

    # Тренировщик
    trainer = pl.Trainer.from_argparse_args(args)

    # Хаки для того, чтобы заставить работать ранний останов и сохранение моделей

    # Ранний останов в случае ухудшения dice
    trainer.configure_early_stopping(pl.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True
    ))

    # Сохранение модели в определенную структуру
    trainer.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='checkpoints/recongnition_{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=-1,
        period=0,
        verbose=True
    )
    trainer.configure_checkpoint_callback()

    # Обучим модель
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PlatesRecognition.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)