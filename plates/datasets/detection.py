import os
import json

import cv2
import numpy as np
from torch.utils.data import Dataset


# def detection_collate(batch):
#     return tuple(zip(*batch))

# def detection_collate_list(batch):
#     return tuple(map(list, zip(*batch)))

def detection_collate(batch):
    return tuple(map(list, zip(*batch)))


class DetectionDataset(Dataset):
    def __init__(self, data_path, json_path=None, transforms=None):
        """
        Для тренировки необходимо подавать json_file
        Для inference не надо подавать json_file
        """

        super(DetectionDataset, self).__init__()

        # Сохраняем входные параметры
        self.data_path = data_path
        self.json_path = json_path
        self.transforms = transforms

        if self.json_path is None:
            # Читаем все файлы в директории
            self.images = os.listdir(self.data_path)
            # Фильтруем маски и вырезанные номеры
            self.images = filter(lambda x: ('.mask.' not in x) and ('.box' not in x), self.images)
            # Сортируем файлы
            self.images = sorted(self.images, key=lambda x: int(os.path.splitext(x)[0]))
            # Добавляем папку
            self.images = map(lambda x: os.path.join(self.data_path, x), self.images)
            # Переводим в список
            self.images = list(self.images)
            # Переводим в массив списков с ключем file
            raw_info = [{'file': x} for x in self.images]
        else:
            basepath = os.path.dirname(self.json_path)
            # Читаем json файл
            with open(json_path, 'r') as in_file:
                raw_info = json.load(in_file)
                for i in range(len(raw_info)):
                    raw_info[i]['file'] = os.path.join(basepath, raw_info[i]['file'])

        # Отфильтруем плохие изображения (нулевой вес)
        self.info = []
        for info in raw_info:
            if os.path.getsize(info['file']) != 0:
                self.info.append(info)

    def __len__(self):
        return len(self.info)

    def get_height_and_width(self, item):
        image = cv2.imread(self.info[item]['file']).astype(np.float32)
        height, width, _ = image.shape
        return height, width

    def __getitem__(self, item):
        image = cv2.imread(self.info[item]['file']).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

        target = None

        if 'nums' in self.info[item]:
            boxes = []
            labels = []
            image_id = [item]
            area = []
            masks = []

            # Проходимся по всем bbox'ам
            for box in self.info[item]['nums']:
                bbox = np.array(box['box'])

                # Находим крайние точки
                x0, y0 = np.min(bbox[:, 0]), np.min(bbox[:, 1])
                x1, y1 = np.max(bbox[:, 0]), np.max(bbox[:, 1])
                boxes.append([x0, y0, x1, y1])

                # Формируем label
                labels.append(1)

                # Формируем area
                area.append((x1 - x0) * (y1 - y0))

                # Формируем mask
                mask = np.zeros(shape=image.shape[:-1], dtype=np.uint8)
                cv2.fillConvexPoly(mask, bbox, (255, 255, 255))
                mask = mask / 255.
                masks.append(mask)

            target = {
                'boxes': np.array(boxes).astype(np.float32),
                'labels': np.array(labels).astype(np.int64),
                'image_id': np.array(image_id).astype(np.int64),
                'area': np.array(area).astype(np.float32),
                'masks': np.array(masks).astype(np.uint8)
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target