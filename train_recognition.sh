#!/usr/bin/env sh

python train_recognition.py \
  --gpus 1 \
  --max_epochs 10 \
  --val_check_interval 0.25 \
  --num_workers 0 \
  --batch_size 64 \
  --generated_data_path /home/kovalexal/Spaces/dev/car_plates_ocr/data/generated_60k_clean \
  --extracted_data_path /home/kovalexal/Spaces/dev/car_plates_ocr/data/train \
  --backbone resnext50_32x4d \
  --cnn_output_len 32 \
  --rnn_bidirectional true
#  --train_percent_check 0.2 \
#  --val_percent_check 0.2