import numpy as np
import torch.nn

from netlib.module import Module
from transforms.registry import Registry

REGISTRY = Registry('activations')

class Pooling(Module):

    def __init__(self, kernel_size: int = None,
                 stride: int = None,
                 padding: int = 0,
                 dilation: int = (1, 1)):

        super().__init__()
        self.kernel_size = kernel_size
        if stride is not None:
            self.stride = stride
        else:
            self.stride = self.kernel_size

        self.padding = padding
        self.dilation = dilation
        self.indexes = None

    def _pooling(self, input_, kernel_size, stride: tuple = None, indexes: tuple = None):
        batch_size, depth, h, w = input_.shape
        kernel_h, kernel_w = kernel_size

        if stride is not None:
            stride = stride
        else:
            stride = kernel_h, kernel_w

        output_h, output_w = (h - kernel_h) // stride[0] + 1, (w - kernel_w) // stride[1] + 1
        if indexes is None:
            i = self._get_kernel_indexes(output_h, kernel_h, stride[0], 1, repeat_axis=1)
            i = np.repeat(i, output_w, axis=0).reshape(-1)

            j = self._get_kernel_indexes(output_w, kernel_w, stride[1], 1, repeat_axis=0)
            j = np.tile(j.reshape(-1), output_h)
        else:
            i, j = indexes

        return input_[..., i, j].reshape(batch_size, depth, output_h, output_w, kernel_h * kernel_w), (i, j)

# torch.nn.AvgPool2d
@REGISTRY.register_module
class MaxPool(Pooling):
    # def __init__(self, kernel_size):
    #     super().__init__(kernel_size=kernel_size)

    def forward(self, input_):
        if self.indexes is None:
            output, self.indexes = self._max_pool(input_, self.kernel_size, self.stride, return_indexes=True)
        else:
            output = self._max_pool(input_, self.kernel_size, self.stride, indexes=self.indexes)

        return output

    def backward(self, grad_out):
        i, j = self.indexes
        input_reshaped = self._saved_input[..., i, j].reshape((-1, np.prod(self.kernel_size)))
        grad = np.zeros_like(input_reshaped)
        grad[np.arange(input_reshaped.shape[0]), input_reshaped.argmax(-1)] = grad_out.flatten()

        tensor = np.zeros_like(self._saved_input)
        np.add.at(tensor, (slice(None), slice(None), i, j), grad.reshape((*self._saved_input.shape[:2], -1)))

    def _max_pool(self, input_, kernel_size, stride: tuple = None, indexes: tuple = None, return_indexes: bool = False):
        output, (i, j) = self._pooling(input_, kernel_size, stride, indexes)
        output = output.max(axis=-1)

        if return_indexes:
            return output, (i, j)
        return output

# torch.nn.AvgPool2d
@REGISTRY.register_module
class AvgPool(Pooling):
    # def __init__(self):
    #     super().__init__()

    def forward(self, input_):
        if self.indexes is None:
            output, self.indexes = self._avg_pool(input_, self.kernel_size, self.stride, return_indexes=True)
        else:
            output = self._avg_pool(input_, self.kernel_size, self.stride, indexes=self.indexes)

        return output

    def backward(self, grad_out):
        i, j = self.indexes
        kernel_size = np.prod(self.kernel_size)
        grad = np.ones(self._saved_input.size).reshape((-1, kernel_size)) * grad_out.reshape(-1, 1) / kernel_size

        tensor = np.zeros_like(self._saved_input)
        np.add.at(tensor, (slice(None), slice(None), i, j), grad.reshape((*self._saved_input.shape[:2], -1)))

        return tensor

    def _avg_pool(self, input_, kernel_size, stride: tuple = None, indexes: tuple = None, return_indexes: bool = False):
        output, (i, j) = self._pooling(input_, kernel_size, stride, indexes)
        output = output.mean(axis=-1)

        if return_indexes:
            return output, (i, j)
        return output


if __name__ == '__main__':
    prepared_input = np.random.rand(4, 8)
    AvgPool.backward(AvgPool(), prepared_input)
