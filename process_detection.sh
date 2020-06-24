#!/usr/bin/env sh

#python process_detection.py \
#  --detection_model checkpoints/detector_epoch=00-val_dice=0.88.ckpt \
#  --device cuda \
#  --data_path /home/kovalexal/Spaces/dev/car_plates_ocr/data \
#  --json_path /home/kovalexal/Spaces/dev/car_plates_ocr/data/train.json

python process_detection.py \
  --detection_model checkpoints/detector_epoch=00-val_dice=0.88.ckpt \
  --device cuda \
  --data_path /home/kovalexal/Spaces/dev/car_plates_ocr/data/test