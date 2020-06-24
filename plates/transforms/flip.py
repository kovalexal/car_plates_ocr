import cv2
import numpy as np


class RandomHorisontalFlip(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        if np.random.uniform() > self.p:
            return image, target

        height, width, _ = image.shape

        # Flip image
        image = cv2.flip(image, 1)

        # Swap mask
        target['masks'] = target['masks'][:, :, ::-1].copy()

        # Swap box coords
        target['boxes'][:, [0, 2]] = target['boxes'][:, [2, 0]]
        target['boxes'][:, [0, 2]] = width - target['boxes'][:, [0, 2]]
        target['boxes'] = target['boxes'].copy()

        return image, target