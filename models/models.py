from models.base import Model
from netlib import Linear
from netlib import LogSoftmax, ReLU
from losses import NLLLoss
from netlib.module import Module


class MLP(Model):
    def __init__(self, cfg, activations: tuple = None):
        super().__init__()
        for i, (shape, params) in enumerate(zip(cfg.layers_shapes, cfg.layers_params)):
            setattr(self, 'Linear' + str(i), Linear(*shape, **params))
            if activations is not None and len(activations) > i:
                setattr(self, type(activations[i]).__name__ + str(i), activations[i])

    def forward(self, tensor):
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                tensor = module(tensor)

        return tensor

    def __repr__(self) -> str:
        return '{}(\n\t{})'.format(type(self).__name__,
                                   ',\n\t'.join(module.__repr__() for module in self.__dict__.values()
                                                if isinstance(module, Module)))

    # def __repr__(self):
    #     string = '{' + ''.join(name.__repr__() + '\n' for name in self.__dict__.keys()) + '\n}'
    #     return string


class Config:
    def __init__(self, layers_shapes, layers_params):
        self.layers_shapes = layers_shapes
        self.layers_params = layers_params


if __name__ == "__main__":
    cfg = Config(layers_shapes=((28 * 28, 200), (200, 10)),
                 layers_params=(dict(), dict()))
    # activation = (LogSoftmax, )
    activation = (ReLU(), LogSoftmax())

    model = MLP(cfg, activation)
