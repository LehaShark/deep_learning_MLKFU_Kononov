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

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

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
class RandomCrop(Transform):

    def __init__(self, size, save_size=False, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.save_size = save_size
        self._interpolation = interpolation

    def forward(self, images):
        height, width = images.shape[1:][:2]

        crop_height, crop_width = self.size if hasattr(self.size, "__len__") else self.size, self.size

        i = random.randint(0, height - crop_height) if height != crop_height else 0
        j = random.randint(0, width - crop_width) if width != crop_width else 0
        images = images[:, i:i + crop_height, j:j + crop_width, ...]

        if self.save_size:
            images = np.array([cv2.resize(img, (width, height),
                                          interpolation=self._interpolation.value)
                               for img in images])
        return images

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size}" \
               + (f', save_size={self.save_size}, interpolation={self._interpolation.name})' if self.save_size else ')')


@REGISTRY.register_module
class Resize(Transform):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.scale = (None, None)

        if isinstance(size, int):
            self.size = size, size
        else:
            if len(size) == 1:
                if isinstance(size[0], int):
                    self.size = size, size
                else:
                    self.size = 0, 0
                    self.scale = size, size
            else:
                if isinstance(size[0], float) or isinstance(size[1], float):
                    self.size = 0, 0
                    self.scale = size
                else:
                    self.size = size

        self.interpolation = interpolation

    def forward(self, images):
        return np.array([cv2.resize(img, self.size,
                                    fx=self.scale[0],
                                    fy=self.scale[1],
                                    interpolation=self.interpolation.value)
                         for img in images])

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size}, interpolation={self.interpolation.name})'


@REGISTRY.register_module
class RandomAffine(Transform):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = tuple(float(d) for d in sorted(degrees if hasattr(degrees, "__len__") else (-degrees, degrees)))

        if translate is not None:
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be in [0; 1] range")
        self.translate = translate

        if scale is not None:
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be greater than 0")
        self.scale = scale

        self.shear = tuple(float(s) for s in sorted(shear if hasattr(shear, "__len__") else (-shear, shear))) \
            if shear is not None else shear

    def forward(self, images):
        height, width = images.shape[1:][:2]
        rot = math.radians(random.uniform(float(self.degrees[0]), float(self.degrees[1])))
        scale = random.uniform(*self.scale) if self.scale is not None else 1.
        center_x, center_y = width // 2, height // 2

        if self.translate is not None:
            max_dx, max_dy = float(self.translate[0] * width), float(self.translate[1] * height)
            translate_x, translate_y = random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)
        else:
            translate_x, translate_y = 0, 0

        if self.shear is not None:
            shear_x = random.uniform(*self.shear[:2])
            shear_y = random.uniform(*self.shear[2:]) if len(self.shear) == 4 else 0
            shear_x, shear_y = math.radians(shear_x), math.radians(shear_y)
        else:
            shear_x, shear_y = 0, 0

        matrix = np.array([
            -math.sin(rot - shear_y) * math.tan(shear_x) / math.cos(shear_y) + math.cos(rot),
            math.cos(rot - shear_y) * math.tan(shear_x) / math.cos(shear_y) + math.sin(rot),
            0.0,

            -math.sin(rot - shear_y) / math.cos(shear_y),
            math.cos(rot - shear_y) / math.cos(shear_y),
            0.0
        ], dtype=np.float32) / scale

        matrix[2] += center_x - matrix[0] * (center_x + translate_x) - matrix[1] * (center_y + translate_y)
        matrix[5] += center_y - matrix[3] * (center_x + translate_x) - matrix[4] * (center_y + translate_y)

        return np.array([cv2.warpAffine(img, matrix.reshape(-1, 3), (width, height)) for img in images])

    def __repr__(self):
        return self.__class__.__name__ + f'(degrees={self.degrees}, translate={self.translate},' \
                                         f'scale={self.scale}, shear={self.shear})'


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
