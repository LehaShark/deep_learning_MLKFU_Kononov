from models.base import Model
from netlib import Linear
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

    def __repr__(self):
        string = '{' + ''.join(name.__repr__() + '\n' for name in self.__dict__.keys()) + '\n}'
        return string
