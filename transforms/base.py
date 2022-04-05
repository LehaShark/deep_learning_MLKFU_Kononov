from enum import Enum
import cv2


class InterpolationMode(Enum):
    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4


class Transform(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()
