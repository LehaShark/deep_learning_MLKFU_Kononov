import numpy as np
import torch.nn

from tensor import Tensor, Parameter
from netlib.module import Module
from enum import Enum

class InitializationType(Enum):
    NORMAL_HE = ('normal', 'he')
    NORMAL_XAVIER = ('normal', 'xavier')
    NORMAL_GLOROT = ('normal', 'glorot')
    UNIFORM_HE = ('uniform', 'he')
    UNIFORM_XAVIER = ('uniform', 'xavier')
    UNIFORM_GLOROT = ('uniform', 'glorot')

torch.nn.MaxPool2d

class Layer(Module):
    def __init__(self, initialization_type):
        super().__init__()
        self.initialization_type = initialization_type

    def state_dict(self):
        return dict(self.named_parameters())

    def load_state_dict(self, state: dict):
        for name, param in self.__dict__.items():
            if isinstance(param, Tensor):
                setattr(self, name, state[name])

    def named_parameters(self):
        for name, param in self.__dict__.items():
            if isinstance(param, Tensor):
                yield name, param

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def reset_parameters(self, initialization_type: InitializationType = None):
        if initialization_type is not None:
            self.initialization_type = initialization_type

        for name, param in self.__dict__.items():
            if isinstance(param, Parameter):
                self._init_param(param)

    def _init_param(self, param, mu=0, scale=1, diff=1):
        shape = param.shape
        m, *n = shape if len(shape) > 1 else (1, shape)
        n = np.prod(n).item()
        distrib, init_type = self._initialization_type.value

        if init_type == 'he':
            diff = 3
            scale = 2 / n
        elif init_type == 'xavier':
            diff = 3
            scale = 1 / n
        elif init_type == 'glorot':
            diff = 6
            scale = 1 / (n + m)

        if distrib == 'normal':
            dsn = np.sqrt(scale)
            param.data = np.random.normal(mu, dsn, shape)
        elif distrib == 'uniform':
            l = np.sqrt(scale * diff)
            param.data = np.random.uniform(-l, l, shape)