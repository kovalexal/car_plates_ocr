import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms import Compose, ToTensor
import pytorch_lightning as pl

from plates.datasets.recognition import abc
from plates.datasets import GeneratedRecognitionDataset, ExtractedRecognitionDataset, recognition_collate
from plates.transforms import Resize


class FeatureExtractor(nn.Module):
    def __init__(self, input_size=(520, 115), backbone='resnet18', output_len=20):
        super().__init__()

        w, h = input_size
        resnet = getattr(models, backbone)(pretrained=True)

        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AvgPool2d(kernel_size=(math.ceil(h / 32), 1))
        self.proj = nn.Conv2d(math.ceil(w / 32), output_len, kernel_size=1)

        if backbone == 'resnet18':
            self.num_output_features = self.cnn[-1][-1].bn2.num_features
        elif (backbone == 'resnet50') or (backbone == 'resnext50_32x4d'):
            self.num_output_features = self.cnn[-1][-1].bn3.num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W')
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super().__init__()

        self.num_classes = num_classes
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(
            in_features=fc_in,
            out_features=num_classes
        )

    def __init_hidden(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)

    def __prepare_features(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C = 1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        return x.squeeze(1).permute(2, 0, 1)

    def forward(self, x):
        x = self.__prepare_features(x)

        batch_size = x.size(1)
        h_0 = self.__init_hidden(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)

        x = self.fc(x)
        return x


class CRNN(nn.Module):
    def __init__(self, backbone='resnet18', alphabet=abc, cnn_input_size=(520, 115), cnn_output_len=20, rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=False):
        super().__init__()

        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size,
            backbone=backbone,
            output_len=cnn_output_len
        )
        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=len(alphabet) + 1,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional
        )

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs



class PlatesRecognition(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Параметры, специфичные для модели"""

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--backbone', type=str, default='resnet18', help='path to generated data')

        parser.add_argument('--generated_data_path', type=str, help='path to generated data')
        parser.add_argument('--extracted_data_path', type=str, help='path to data extracted from train')

        parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
        parser.add_argument('--batch_size', type=int, default=6, help='train batch size')

        parser.add_argument('--cnn_output_len', type=int, default=20, help='cnn sequence length')
        parser.add_argument('--rnn_hidden_size', type=int, default=128, help='rnn hidden size')
        parser.add_argument('--rnn_num_layers', type=int, default=2, help='number of rnn layers')
        parser.add_argument('--rnn_dropout', type=float, default=0.3, help='dropout of rnn')
        parser.add_argument('--rnn_bidirectional', type=bool, default=False, help='is rnn bidirectional or not')

        parser.add_argument('--learning_rate', type=float, default=3e-4, help='initial learning rate')

        return parser

    def __init__(self, hparams):
        super().__init__()

        # Гиперпараметры модели в pytorch_lightning
        self.hparams = hparams

        # Задаем нашу модель
        self.crnn = CRNN(
            backbone=self.hparams.backbone,
            cnn_input_size=(520, 115),
            cnn_output_len=self.hparams.cnn_output_len,
            rnn_hidden_size=self.hparams.rnn_hidden_size,
            rnn_num_layers=self.hparams.rnn_num_layers,
            rnn_dropout=self.hparams.rnn_dropout,
            rnn_bidirectional=self.hparams.rnn_bidirectional
        )

        # Замораживаем все слои предобученного resnet
        for parameter in self.crnn.features_extractor.cnn.parameters():
            parameter.requires_grad = False


    def forward(self, images):
        return self.crnn(images)


    ####################################################################
    ############################## Helpers #############################
    ####################################################################


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return {
            'optimizer': optimizer
        }

    def prepare_data(self):
        transforms = Compose([
            Resize((520, 115)),
            # ToTensor(),
        ])

        generated_dataset = GeneratedRecognitionDataset(
            self.hparams.generated_data_path,
            transforms
        )

        extracted_model_dataset = ExtractedRecognitionDataset(
            self.hparams.extracted_data_path,
            mask='*.ebox.*',
            transforms=transforms
        )

        extracted_dataset = ExtractedRecognitionDataset(
            self.hparams.extracted_data_path,
            mask='*.box.*',
            transforms=transforms
        )

        dataset = ConcatDataset([
            generated_dataset,
            extracted_dataset,
            extracted_model_dataset
        ])

        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size

        self.__train_dataset, self.__val_dataset = random_split(dataset, [train_size, val_size])


    ####################################################################
    ############################# Training #############################
    ####################################################################


    def train_dataloader(self):
        return DataLoader(
            self.__train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=recognition_collate,
            drop_last=True,
            shuffle=True
        )


    def training_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        seqs_gt = batch['seqs']
        seqs_lens_gt = batch['seq_lens']

        seqs_pred = self.crnn(images).cpu()
        log_probs = F.log_softmax(seqs_pred, dim=2)
        seqs_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=seqs_gt,
            input_lengths=seqs_lens_pred,
            target_lengths=seqs_lens_gt
        )

        # Логируем в tensorboard
        self.logger.log_metrics({'train_loss': loss.item()}, step=self.global_step)

        return {'loss': loss}


    def training_epoch_end(self, outputs):
        # Разморозим все слои после первой эпохи
        for parameter in self.crnn.parameters():
            parameter.requires_grad = True
        return {}


    ####################################################################
    ############################ Validation ############################
    ####################################################################


    def val_dataloader(self):
        return DataLoader(
            self.__val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=recognition_collate,
            drop_last=False,
            shuffle=False
        )


    def validation_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        seqs_gt = batch['seqs']
        seqs_lens_gt = batch['seq_lens']

        seqs_pred = self.crnn(images).cpu()
        log_probs = F.log_softmax(seqs_pred, dim=2)
        seqs_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=seqs_gt,
            input_lengths=seqs_lens_pred,
            target_lengths=seqs_lens_gt
        )

        self.logger.log_metrics({'val_loss': loss.item()}, step=self.global_step)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Логируем в tensorboard
        self.logger.log_metrics({'val_loss': avg_loss.item()}, step=self.global_step)

        return {
            'val_loss': avg_loss,
            'progress_bar': {
                'val_loss': avg_loss.item()
            }
        }