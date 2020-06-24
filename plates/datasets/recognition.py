import os
import re
import json
import glob

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt

abc = "0123456789ABCEHKMOPTXY"
mapping = {
    'А': 'A',
    'В': 'B',
    'С': 'C',
    'Е': 'E',
    'Н': 'H',
    'К': 'K',
    'М': 'M',
    'О': 'O',
    'Р': 'P',
    'Т': 'T',
    'Х': 'X',
    'У': 'Y',
}

def convert_to_eng(text, mapping=mapping):
    return ''.join([mapping.get(a, a) for a in text])


def text_to_seq(text):
    seq = [abc.find(c) + 1 for c in text]
    return seq


def recognition_collate(batch):
    images = list()
    seqs = list()
    seq_lens = list()
    for sample in batch:
        images.append(torch.from_numpy(sample["image"].transpose((2, 0, 1))).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"images": images, "seqs": seqs, "seq_lens": seq_lens}
    return batch



class GeneratedRecognitionDataset(Dataset):
    """Сгенерированный датасет"""

    def __init__(self, data_path, transforms=None):
        super().__init__()

        self.data_path = data_path
        self.transforms = transforms

        self.images = sorted([f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))])

        # Фильтруем изображения - должны открываться
        def check_image_opens(path):
            from PIL import Image
            try:
                Image.open(path).tobytes()
                return True
            except IOError:
                return False

        self.images = list(filter(lambda x: check_image_opens(os.path.join(self.data_path, x)), self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Читаем изображение
        # image = cv2.imread(os.path.join(self.data_path, self.images[item])).astype(np.float32)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

        import PIL
        image_pil = PIL.Image.open(os.path.join(self.data_path, self.images[item])).convert('RGB')
        image = np.array(image_pil)
        image = image / 255.

        # Получаем текст из названия изображения
        text = convert_to_eng(os.path.splitext(self.images[item])[0].upper())

        # Применяем трансформации к изображению
        if self.transforms is not None:
            image = self.transforms(image)

        # Формируем выходной словарь
        seq = text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)

        return output


class ExtractedRecognitionDataset(Dataset):
    """Датасет с выделенными моделью изображениями (суффикс ebox)"""

    def __init__(self, data_path, mask='*.ebox.*', transforms=None):
        super().__init__()

        self.data_path = data_path
        self.mask = mask
        self.transforms = transforms

        self.images = glob.glob(os.path.join(self.data_path, self.mask), recursive=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Читаем изображение
        image = cv2.imread(os.path.join(self.data_path, self.images[item])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

        # Получаем текст из названия изображения (если он есть)
        filename = os.path.basename(self.images[item])
        text = re.search(r'(?<=box\.\d{1}\.)\w+(?=\.)', filename)
        if text is not None:
            text = text.group(0)

        # Применяем трансформации к изображению
        if self.transforms is not None:
            image = self.transforms(image)

        # Формируем выходной словарь
        output = dict(image = image)
        if text is not None:
            seq = text_to_seq(text)
            seq_len = len(seq)
            output.update(seq=seq, seq_len=seq_len, text=text)

        return output



if __name__ == '__main__':
    # transforms = Compose([
    #     Resize((520, 115)),
    #     ToTensor(),
    # ])
    #
    # d = GeneratedRecognitionDataset(
    #     '/home/kovalexal/Spaces/dev/car_plates_ocr/data/generated_60k',
    #     transforms
    # )
    e = ExtractedRecognitionDataset('/home/kovalexal/Spaces/dev/car_plates_ocr/data/train')
    print(e[0])



# class RecognitionDataset(Dataset):
#     def __init__(self, data_path, json_path, abc="0123456789ABCEHKMOPTXY", transforms=None):
#         """
#         Для тренировки необходимо подавать json_file
#         Для inference не надо подавать json_file
#         """
#
#         super(RecognitionDataset, self).__init__()
#
#         # Сохраняем входные параметры
#         self.data_path = data_path
#         self.json_path = json_path
#         self.abc = abc
#         self.transforms = transforms
#
#         # Читаем данные
#         self.info = []
#         basepath = os.path.dirname(self.json_path)
#         # Читаем json файл
#         with open(json_path, 'r') as in_file:
#             info = json.load(in_file)
#             for i in range(len(info)):
#                 info[i]['file'] = os.path.join(basepath, info[i]['file'])
#                 for bbox in info[i]['nums']:
#                     bbox['file'] = info[i]['file']
#                     self.info.append(bbox)
#
#     def __len__(self):
#         return len(self.info)
#
#     @staticmethod
#     def order_points(pts):
#         # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#
#         # initialzie a list of coordinates that will be ordered
#         # such that the first entry in the list is the top-left,
#         # the second entry is the top-right, the third is the
#         # bottom-right, and the fourth is the bottom-left
#         rect = np.zeros((4, 2), dtype="float32")
#
#         # the top-left point will have the smallest sum, whereas
#         # the bottom-right point will have the largest sum
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#
#         # now, compute the difference between the points, the
#         # top-right point will have the smallest difference,
#         # whereas the bottom-left will have the largest difference
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#
#         # return the ordered coordinates
#         return rect
#
#     @staticmethod
#     def four_point_transform(image, pts):
#         # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#
#         # obtain a consistent order of the points and unpack them
#         # individually
#         rect = order_points(pts)
#         (tl, tr, br, bl) = rect
#
#         # compute the width of the new image, which will be the
#         # maximum distance between bottom-right and bottom-left
#         # x-coordiates or the top-right and top-left x-coordinates
#         widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#         widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#         maxWidth = max(int(widthA), int(widthB))
#
#         # compute the height of the new image, which will be the
#         # maximum distance between the top-right and bottom-right
#         # y-coordinates or the top-left and bottom-left y-coordinates
#         heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#         heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#         maxHeight = max(int(heightA), int(heightB))
#
#         # now that we have the dimensions of the new image, construct
#         # the set of destination points to obtain a "birds eye view",
#         # (i.e. top-down view) of the image, again specifying points
#         # in the top-left, top-right, bottom-right, and bottom-left order
#         dst = np.array([
#             [0, 0],
#             [maxWidth - 1, 0],
#             [maxWidth - 1, maxHeight - 1],
#             [0, maxHeight - 1]], dtype="float32")
#
#         # compute the perspective transform matrix and then apply it
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#
#         # return the warped image
#         return warped
#
#     def text_to_seq(self, text):
#         seq = [self.abc.find(c) + 1 for c in text]
#         return seq
#
#     def __getitem__(self, item):
#         image = cv2.imread(self.info[item]['file']).astype(np.float32) / 255.
#         box = np.array(self.info[item]['box'])
#         text = self.info[item]['text']
#         image_number = self.four_point_transform(image, box)
#
#         #         plt.imshow(image_number)
#         #         print(text)
#         #         print(self.text_to_seq(text))
#
#         return image_number, self.text_to_seq(text)