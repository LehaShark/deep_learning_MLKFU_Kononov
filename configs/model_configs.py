import numpy as np

from configs.base import ModelConfig


class MLPConfig(ModelConfig):
    def __init__(self, input_size, output_size):
        hidden1 = 200
        input_size = int(np.prod(np.array(input_size))) if len(input_size) > 1 else input_size
        input_layer_shape = (input_size, hidden1)
        input_layer_params = dict()

        output_layer_shape = (hidden1, output_size)
        output_layer_params = dict()

        epochs = 1
        lr = .01
        momentum = 0

        activations = (('Tanh', dict()),)
        criterion = ('CrossEntropyLoss', dict())

        super().__init__(layers_shapes=(input_layer_shape, output_layer_shape),
                         layers_params=(input_layer_params, output_layer_params),
                         epochs=epochs,
                         lr=lr,
                         activations=activations,
                         criterion=criterion,
                         momentum=momentum)
