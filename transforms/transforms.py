import math
import random
from collections.abc import Sequence
import cv2
import numpy as np
from transforms.registry import Registry_Assembly
from transforms.base import Transform, InterpolationMode

REGISTRY = Registry_Assembly('transforms')


@REGISTRY.register_module
class Normalize(Transform):
    def __init__(self, mean, std, in_range: bool = True):
        self.mean = mean
        self.std = std
        self.in_range = in_range

    def forward(self, images):
        if not isinstance(images, np.ndarray):
            raise TypeError(f'Input array of images should be a ndarray. Got {type(images)}.')

        if not np.issubdtype(images.dtype, np.floating):
            raise TypeError(f'Input array of images should be a float array. Got {images.dtype}.')

        if images.ndim < 3:
            raise ValueError(f'Expected array to be a tensor image of size (N, ..., H, W). Got size = {images.size()}.')

        dtype = images.dtype
        self.mean = np.array(self.mean, dtype=dtype)
        self.std = np.array(self.std, dtype=dtype)

        if (self.std == 0).any():
            raise ValueError(f'std after conversion to {dtype} leading to division by zero.')

        images = (images - self.mean) / self.std
        ext = round(1 - self.mean / self.std, 1)
        images[images > ext] = ext
        images[images < -ext] = -ext

        return images + ext if self.in_range else images

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


@REGISTRY.register_module
class ToOneHot(Transform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def forward(self, targets):
        if hasattr(targets, '__len__'):
            one_hot = np.zeros((len(targets), self.num_classes), dtype=np.int32)
            targets = np.arange(len(targets)), np.array(targets).reshape(-1)
        else:
            one_hot = np.zeros(self.num_classes, dtype=np.int32)

        one_hot[targets] = 1
        return one_hot

    def __repr__(self):
        return self.__class__.__name__ + f'(num_classes={self.num_classes})'


@REGISTRY.register_module
class Pad(Transform):
    def __init__(self, padding, padding_mode="constant", save_size=False, interpolation=InterpolationMode.BILINEAR):

        # if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        #     raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                             f"{len(padding)} element tuple")

        self.padding = padding
        self.padding_mode = padding_mode
        self.save_size = save_size
        self._interpolation = interpolation

    def forward(self, images):

        if len(images.shape) == 3:
            height, width = images.shape[1:]
            images = np.array([np.pad(img, self.padding, self.padding_mode) for img in images])
        else:
            height, width, color = images.shape[1:]
            images = np.array([[np.pad(img[..., c], self.padding, self.padding_mode)
                                for c in range(color)]
                               for img in images])

            images = images.transpose((0, 2, 3, 1))

        if self.save_size:
            images = np.array([cv2.resize(img, (width, height),
                                          interpolation=self._interpolation.value)
                               for img in images])
        return images

    def __repr__(self):
        return self.__class__.__name__ + f'(padding={self.padding}, padding_mode={self.padding_mode}' \
               + (f', save_size={self.save_size}, interpolation={self._interpolation.name})' if self.save_size else ')')


@REGISTRY.register_module
class GaussianNoise(Transform):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def forward(self, images):
        return images + np.random.normal(self.mean, self.std ** 0.5, images.shape)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


@REGISTRY.register_module
class SaltAndPepperNoise(Transform):
    def __init__(self,
                 threshold: float = 0.005,
                 lowerValue: int = 5,
                 upperValue: int = 250,
                 normalize: bool = False):
        self.threshold = threshold
        self.lowerValue = lowerValue
        self.upperValue = upperValue
        self.normalize = normalize

    def forward(self, images):
        random_matrix = np.random.random(images.shape)
        images[random_matrix >= (1 - self.threshold)] = self.upperValue / (255 if self.normalize else 1)
        images[random_matrix <= self.threshold] = self.lowerValue / (255 if self.normalize else 1)
        return images

    def __repr__(self):
        return self.__class__.__name__ + f'(threshold={self.threshold}, lowerValue={self.lowerValue},' \
                                         f' upperValue={self.upperValue}, normalize={self.normalize})'
