import numpy as np
from netlib import functions as f
from netlib.module import Module
from transforms.registry import Registry

REGISTRY = Registry('activations')


class Loss(Module):
    def __init__(self):
        super().__init__()
        self._saved_target_indexes = None

    def forward(self, input_, target):
        if self._saved_tensor is not None:
            self._saved_target_indexes = np.where(target == 1) if input_.shape == target.shape \
                else (np.arange(input_.shape[0]), target)

        return self.get_loss(input_, target)

    def get_loss(self, input_, target):
        raise NotImplementedError()


@REGISTRY.register_module
class CrossEntropyLoss(Loss):

    def get_loss(self, input_, target):
        return f.cross_entropy(input_, target)

    def backward(self, dl=1):
        grad = f.softmax(self._saved_input)
        grad[self._saved_target_indexes] = grad[self._saved_target_indexes] - 1

        return grad / grad.shape[0]


@REGISTRY.register_module
class NLLLoss(Loss):

    def get_loss(self, input_, target):
        return f.nll_loss(input_, target)

    def backward(self, dl=1):
        grad = np.zeros_like(self._saved_input)
        grad[self._saved_target_indexes] = -1 / grad.shape[0]
        return grad