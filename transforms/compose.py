import random
from collections.abc import Sequence
from transforms.registry import Registry

REGISTRY = Registry('composes')


@REGISTRY.register_module
class Compose(object):
    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


@REGISTRY.register_module
class RandomApply(Compose):

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p

    def __call__(self, data):
        if self.p < random.random():
            return data
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
