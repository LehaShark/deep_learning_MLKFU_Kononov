from models.base import Model
from netlib import Linear
from netlib import LogSoftmax, ReLU
from losses import NLLLoss
from netlib.conv import Convolution
from netlib.module import Module
from netlib.pooling import MaxPool


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




class ConvNN(Model):
    def __init__(self, cfg, activations: tuple = None):
        super().__init__()
        self.conv1 = Convolution(1, 16, kernel_size=5, stride=1)
        self.conv2 = Convolution(16, 32, kernel_size=5, stride=1)
        self.max_pool = MaxPool(2)
        self.fc1 = Linear(32 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)
        self.relu = ReLU()

    def forward(self, tensor):
        tensor = self.conv1(tensor)
        tensor = self.relu(tensor)
        tensor = self.max_pool(tensor)

        tensor = self.conv2(tensor)
        tensor = self.relu(tensor)
        tensor = self.max_pool(tensor)

        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor = self.fc1(tensor)
        tensor = self.relu(tensor)
        tensor = self.fc2(tensor)

        return tensor

    def __repr__(self) -> str:
        return '{}(\n\t{})'.format(type(self).__name__,
                                   ',\n\t'.join(module.__repr__() for module in self.__dict__.values()
                                                if isinstance(module, Module)))


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
