from tensor import Tensor
from netlib.module import Module


class Layer(Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

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
