import cv2

class Resize(object):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __call__(self, item):
        return cv2.resize(item, self.shape, interpolation=cv2.INTER_CUBIC)