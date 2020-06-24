import os
import sys
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from tqdm import tqdm

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from plates.models import PlatesDetector
from plates.datasets import DetectionDataset
from plates.transforms import Compose, ToTensor


def order_points(pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def fill_holes(img):
    im_th = img.copy()


    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    if img[0,0] != 0:
        print("WARNING: Filling something you shouldn't")
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def main(args):
    device = torch.device(args.device)
    model = PlatesDetector.load_from_checkpoint(args.detection_model).to(args.device)
    model.eval()
    transforms = Compose([
        ToTensor()
    ])
    dataset = DetectionDataset(args.data_path, args.json_path, transforms=transforms)

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        # if i != 1339:
        #     continue

        image, target = dataset[i]
        info = dataset.info[i]
        file = info['file']

        cv2_image = cv2.imread(file).astype(np.float32)
        # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Вырезаем реальные данные, если они есть
        if 'nums' in info:
            for j, plate in enumerate(info['nums']):
                box = np.array(plate['box'])
                text = plate['text']
                image_plate = four_point_transform(cv2_image, box)

                # Формируем название выходного файла
                output_file = os.path.splitext(file)
                output_file = '{}.box.{}.{}{}'.format(output_file[0], j, text, output_file[1])

                # Записываем результат
                cv2.imwrite(output_file, image_plate)

        # Прогоняем через модель
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]

        # Выделяем номерные знаки
        for k in range(len(output['scores'])):
            # Пропускаем все кроме номера
            if output['labels'][k] != 1:
                continue

            # Пропускаем все номера по порогу
            if output['scores'][k] < args.score_threshold:
                continue

            # Накладываем маску
            mask = output['masks'][k].detach().cpu().numpy()
            mask = (mask[0] > args.mask_threshold).astype(np.uint8)
            mask = fill_holes(mask)

            # Находим минимальный обрамляющий четырехугольник
            contour = cv2.findContours(mask.copy(), 1, 1)[0][0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Вырезаем номерной знак
            image_plate = four_point_transform(cv2_image, box)

            # plt.imshow(image_plate / 255.)
            # plt.show()

            # Если у нас была информация (тренировочная выборка), то необходимо найти данный номерной знак в имеющейся информации
            if ('nums' in info):
                box = order_points(box)

                # Находим крайние точки
                x0, y0 = np.min(box[:, 0]), np.min(box[:, 1])
                x1, y1 = np.max(box[:, 0]), np.max(box[:, 1])
                bbox = [x0, y0, x1, y1]

                # Находим максимальный iou с одним из plate
                max_iou = -float('inf')
                text = ''
                for plate in info['nums']:
                    # Находим крайние точки
                    plate_box = order_points(np.array(plate['box']))
                    x0, y0 = np.min(plate_box[:, 0]), np.min(plate_box[:, 1])
                    x1, y1 = np.max(plate_box[:, 0]), np.max(plate_box[:, 1])
                    plate_bbox = [x0, y0, x1, y1]
                    iou = bb_intersection_over_union(bbox, plate_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        text = plate['text']

                if max_iou < 1e-1:
                    continue

                # Формируем название выходного файла
                output_file = os.path.splitext(file)
                output_file = '{}.ebox.{}.{}{}'.format(output_file[0], k, text, output_file[1])
            else:
                # Формируем название выходного файла
                output_file = os.path.splitext(file)
                output_file = '{}.ebox.{}{}'.format(output_file[0], k, output_file[1])

            # Записываем результат
            cv2.imwrite(output_file, image_plate)

        # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        # plt.show()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--detection_model', type=str, help='path to detection model')
    parser.add_argument('--device', type=str, default='cuda', help='device used to process data')
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--json_path', type=str, default=None, help='path to json data')
    parser.add_argument('--score_threshold', type=float, default=0.95, help='score threshold for mask-rcnn')
    parser.add_argument('--mask_threshold', type=float, default=0.05, help='mask threshold for mask binarization')

    args = parser.parse_args()
    main(args)