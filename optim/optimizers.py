import numpy as np


class Optimizer(object):
    def __init__(self, params: dict):
        self.params = dict(params)
        for name, param in self.params.items():
            param.set4_optimization()


    def _update_rule(self):
        raise NotImplementedError()

    def step(self):
        self._update_rule()

    def zero_grad(self):
        for name, param in self.params.items():
            param.grad = np.zeros_like(param.data)


class SGD(Optimizer):
    def __init__(self, params: dict, lr: float, momentum: float = 0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [0]*len(self.params)

    def _update_rule(self):
        for i, (name, param) in enumerate(self.params.items()):
            self._velocity[i] = self.momentum * self._velocity[i] - self.lr * param.grad
            self.params[name].data = param.data + self._velocity[i]
