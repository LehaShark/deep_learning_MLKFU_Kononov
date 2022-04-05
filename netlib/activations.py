import numpy as np

from netlib import functions as f
from netlib.module import Module
from transforms.registry import Registry

REGISTRY = Registry('activations')


@REGISTRY.register_module
class LogSoftmax(Module):
    def forward(self, input_):
        return f.log_softmax(input_)

    def backward(self, grad_output):
        grad = f.softmax(self._saved_input) / self._saved_input.shape[0]
        return grad + grad_output


@REGISTRY.register_module
class ReLU(Module):
    def forward(self, input_):
        return f.relu(input_)

    def backward(self, grad_output):
        return grad_output * (self._saved_input > 0)


@REGISTRY.register_module
class Sigmoid(Module):
    def forward(self, input_):
        return f.sigmoid(input_)

    def backward(self, grad_output):
        return grad_output * f.sigmoid(self._saved_input) * (1 - f.sigmoid(self._saved_input))


@REGISTRY.register_module
class Tanh(Module):
    def forward(self, input_):
        return np.tanh(input_)

    def backward(self, grad_output):
        return grad_output * (1 - np.tanh(self._saved_input) ** 2)


@REGISTRY.register_module
class Softmax(Module):
    def forward(self, input_):
        return f.softmax(input_)

    def backward(self, grad_output):
        grad = grad_output - grad_output
        grad[grad == 0] = np.random.normal(0, 1e-30, grad[grad == 0].shape)

        return grad