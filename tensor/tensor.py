from typing import Any
import numpy as np
from collections import deque


class Tensor(object):
    def __init__(self, x, requires_grad=False, dtype=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        self.data = x if dtype is None else x.astype(dtype)

        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._optimize = False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_optimize(self):
        return self._optimize

    def set4_optimization(self):
        self._optimize = True

    @property
    def shape(self):
        return self.data.shape

    def start_graph(self):
        self.grad_fn = deque()

    def backward(self, retain_graph: bool = False):
        grad = None
        grad_fn = self.grad_fn.copy() if retain_graph else None

        for _ in range(len(self.grad_fn)):
            obj = self.grad_fn.pop()
            grad = obj.backward(grad) if grad is not None else obj.backward()
            obj.set_grad(grad, retain_graph)

        self.grad_fn = grad_fn

    def __getitem__(self, index=None):
        return self.data[index]

    def __deepcopy__(self, memo):
        my_copy = type(self)(self.data, self.requires_grad)
        memo[id(self)] = my_copy
        my_copy.data = np.copy(self.data)
        my_copy.grad = np.copy(self.grad) if self.grad is not None else None
        my_copy.grad_fn = self.grad_fn.copy() if self.grad_fn is not None else None
        return my_copy

class Parameter(Tensor):
    def __init__(self, x, dtype=None):
        super().__init__(x, True, dtype)
        self._optimize = True


class no_grad:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.requires_grad = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.model.requires_grad = True


if __name__ == '__main__':
    x = np.random.random((3, 2)) - 0.5
    my_tensor = Tensor(x)
    print(my_tensor)
