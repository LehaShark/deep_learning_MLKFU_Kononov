import copy

import numpy as np


class Module(object):
    def __init__(self):
        self._saved_tensor = None

    @property
    def _saved_input(self):
        return self._saved_tensor.data

    def set_grad(self, grad, retain_graph: bool = False):
        if self._saved_tensor.is_optimize:
            self._saved_tensor.grad = grad
        if not retain_graph:
            self._saved_tensor.grad_fn = None

    def __call__(self, tensor, *args, **kwargs):

        output = copy.deepcopy(tensor)

        if tensor.requires_grad:
            if output.grad_fn is None:
                output.start_graph()

            output.grad_fn.append(self)
            self._saved_tensor = tensor

        output.data = self.forward(tensor.data, *args, **kwargs)
        return output

    # todo: check for maxpool
    def _get_kernel_indexes(self, num_kernels, axis_length, stride: int, depth: int, repeat_axis):
        idx = np.tile(np.arange(axis_length), num_kernels).reshape(-1, axis_length)
        idx += stride * np.arange(num_kernels).reshape(-1, 1)

        idx = np.tile(np.repeat(idx, axis_length, axis=repeat_axis), depth)
        return np.repeat(idx, num_kernels, axis=0).reshape(-1) if repeat_axis == 1 else np.tile(idx.reshape(-1), num_kernels)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()
