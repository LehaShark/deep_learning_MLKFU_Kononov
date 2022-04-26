import numpy as np
from netlib.layer import Layer, InitializationType
from tensor import Parameter


class Linear(Layer):
    def __init__(self, input_shape, output_shape,
                 use_bias=True,
                 initialization_type: InitializationType = InitializationType.UNIFORM_HE,
                 dtype=np.float32):

        super().__init__(initialization_type)
        self.weight = Parameter(np.empty((output_shape, input_shape), dtype=dtype))
        self.bias = Parameter(np.empty(output_shape, dtype=dtype)) if use_bias else None

        self._initialization_type = initialization_type
        self.reset_parameters()

    def forward(self, x):
        return np.dot(x, self.weight.data.T) + (self.bias.data if self.bias is not None else 0)

    def backward(self, dy):
        self.weight.grad = np.dot(self._saved_input.T, dy).T
        if self.bias is not None:
            self.bias.grad = np.sum(dy, axis=0)
        return np.dot(dy, self.weight.data)

    def __repr__(self) -> str:
        return '{}(in_features={}, out_features={}, bias={})'.format(
            type(self).__name__, self.input_shape, self.output_shape, self.bias is not None
        )