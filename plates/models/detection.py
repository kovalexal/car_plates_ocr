import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN

from plates.datasets import DetectionDataset, detection_collate
from plates.transforms import Compose, ToTensor, RandomHorisontalFlip
from plates.utils import dice_coeff


class PlatesDetector(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Параметры, специфичные для модели"""

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--data_path', type=str, help='path to data (train, val)')
        parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')

        parser.add_argument('--train_json', type=str, help='path to json with train data')
        parser.add_argument('--train_batch_size', type=int, default=6, help='train batch size')

        parser.add_argument('--val_json', type=str, help='path to json with val data')
        parser.add_argument('--val_batch_size', type=int, default=4, help='val batch size')
        parser.add_argument('--val_bbox_score_threshold', type=float, default=0.8, help='validation threshold for score')
        parser.add_argument('--val_mask_proba_threshold', type=float, default=0.05, help='validation threshold for mask')

        parser.add_argument('--learning_rate', type=float, default=3e-4, help='initial learning rate')

        return parser


    def __init__(self, hparams):
        super().__init__()

        # Гиперпараметры модели в pytorch_lightning
        self.hparams = hparams

        # Mask-RCNN
        self.mask_rcnn = maskrcnn_resnet50_fpn(
            pretrained_backbone=True,
            pretrained=True
        )

        # Обновляем выход для предсказания номерного знака
        num_classes = 2
        in_features = self.mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)

        # Разрешаем обновлять только определенные параметры
        for parameter in self.mask_rcnn.parameters():
            parameter.requires_grad = False
        for parameter in self.mask_rcnn.backbone.fpn.parameters():
            parameter.requires_grad = True
        for parameter in self.mask_rcnn.rpn.parameters():
            parameter.requires_grad = True
        for parameter in self.mask_rcnn.roi_heads.parameters():
            parameter.requires_grad = True


    def forward(self, images, targets=None):
        return self.mask_rcnn(images, targets)


    ####################################################################
    ############################## Helpers #############################
    ####################################################################


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return {
            'optimizer': optimizer
        }


    ####################################################################
    ############################# Training #############################
    ####################################################################


    def train_dataloader(self):
        transforms = Compose([
            RandomHorisontalFlip(p=0.2),
            ToTensor(),
        ])

        train_dataset = DetectionDataset(
            self.hparams.data_path,
            self.hparams.train_json,
            transforms
        )

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=detection_collate,
            drop_last=True,
            shuffle=True
        )


    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [image.to(self._device) for image in images]
        targets = [{k: v.to(self._device) for k, v in target.items()} for target in targets]

        # Mask-RCNN возвращает словарь с loss'ами
        loss_dict = self.forward(images, targets)
        # Наша ошибка - сумма всех лоссов
        loss = sum(loss for loss in loss_dict.values())

        # Логируем в tensorboard
        self.logger.log_metrics({'train_loss': loss.item()}, step=self.global_step)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # Разморозим все слои после первой эпохи
        for parameter in self.mask_rcnn.parameters():
            parameter.requires_grad = True
        return {}

    ####################################################################
    ############################ Validation ############################
    ####################################################################


    def val_dataloader(self):
        transforms = Compose([
            ToTensor(),
        ])

        val_dataset = DetectionDataset(
            self.hparams.data_path,
            self.hparams.val_json,
            transforms
        )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=detection_collate,
            drop_last=False,
            shuffle=False
        )


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [image.to(self._device) for image in images]
        predictions = self.forward(images)

        dice_coeffs = []
        for img_idx in range(len(predictions)):
            real_mask = targets[img_idx]['masks'].sum(dim=0)

            idxs = (predictions[img_idx]['labels'] == 1) & (predictions[img_idx]['scores'] > self.hparams.val_bbox_score_threshold)
            predicted_masks = predictions[img_idx]['masks'][idxs, 0, :, :]
            if predicted_masks.shape[0] == 0:
                predicted_mask = torch.zeros_like(real_mask, dtype=torch.float32)
            else:
                predicted_mask = (predicted_masks.max(dim=0)[0] > self.hparams.val_mask_proba_threshold).float()
            dice_coeffs.append(dice_coeff(predicted_mask, real_mask))

        val_dice = torch.mean(torch.stack(dice_coeffs))

        self.logger.log_metrics({'val_dice': val_dice.item()}, step=self.global_step)

        return {
            'val_dice': val_dice,
        }


    def validation_epoch_end(self, outputs):
        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()

        # Логируем в tensorboard
        self.logger.log_metrics({'val_dice': avg_dice.item()}, step=self.global_step)

        return {
            'val_dice': avg_dice,
            'progress_bar': {
                'val_dice': avg_dice.item()
            }
        }