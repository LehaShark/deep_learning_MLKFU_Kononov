from enum import Enum
import numpy as np
from netlib.layer import Layer
from tensor import Parameter


class InitializationType(Enum):
    NORMAL_HE = ('normal', 'he')
    NORMAL_XAVIER = ('normal', 'xavier')
    NORMAL_GLOROT = ('normal', 'glorot')
    UNIFORM_HE = ('uniform', 'he')
    UNIFORM_XAVIER = ('uniform', 'xavier')
    UNIFORM_GLOROT = ('uniform', 'glorot')


class Linear(Layer):
    def __init__(self, input_shape, output_shape,
                 use_bias=True,
                 initialization_type: InitializationType = InitializationType.UNIFORM_HE,
                 regularization_type=None,
                 weight_decay=None,
                 dtype=np.float32):

        super().__init__(input_shape, output_shape)
        self.weight = Parameter(np.empty((input_shape, output_shape), dtype=dtype))
        self.bias = Parameter(np.empty(output_shape, dtype=dtype)) if use_bias else None

        self._initialization_type = initialization_type
        self.reset_parameters()

    def reset_parameters(self, initialization_type: InitializationType = None):
        if initialization_type is not None:
            self._initialization_type = initialization_type
        self._init_param(self.weight)
        if self.bias is not None:
            self._init_param(self.bias)

    def forward(self, x):
        return np.dot(x, self.weight.data) + (self.bias.data if self.bias is not None else 0)

    def backward(self, dy):
        self.weight.grad = np.dot(self._saved_input.T, dy)
        if self.bias is not None:
            self.bias.grad = np.sum(dy, axis=0)
        return np.dot(dy, self.weight.data.T)

    def _init_param(self, param, mu=0, scale=1, diff=1):
        dtype = param.dtype
        shape = param.shape
        n, m, *_ = *shape, 0
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

    def __repr__(self) -> str:
        return '{}(in_features={}, out_features={}, bias={})'.format(
            type(self).__name__, self.input_shape, self.output_shape, self.bias is not None
        )


