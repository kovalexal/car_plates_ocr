#!/usr/bin/env sh

python train_detection.py \
  --gpus 1 \
  --max_epochs 10 \
  --val_check_interval 0.25 \
  --data_path /home/kovalexal/Spaces/dev/car_plates_ocr/data \
  --train_json /home/kovalexal/Spaces/dev/car_plates_ocr/data/train_sampled.json \
  --train_batch_size 4 \
  --val_json /home/kovalexal/Spaces/dev/car_plates_ocr/data/val_sampled.json \
  --val_batch_size 2 \
  --val_bbox_score_threshold 0.95 \
  --val_mask_proba_threshold 0.05 \
  --num_workers 16
#  --resume_from_checkpoint ./checkpoints/detector_epoch=01-val_dice=0.77.ckpt
#  --train_percent_check 0.02 \
#  --val_percent_check 0.2