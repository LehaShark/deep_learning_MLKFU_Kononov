import numpy as np

from configs.base import ModelConfig
from netlib.linear import InitializationType


class MLPConfig(ModelConfig):
    def __init__(self, input_size, output_size,
                 hidden_neurons=200,
                 num_layers=2,
                 layers_kwargs=None,
                 activation='ReLU',
                 activation_kwargs=None,
                 criterion='CrossEntropyLoss',
                 criterion_kwargs=None):
        input_size = int(np.prod(np.array(input_size))) if len(input_size) > 1 else input_size
        layers_shape, layers_params = get_layers_settings(input_size, output_size, hidden_neurons,
                                                          num_layers, layers_kwargs)
        activations = get_activations(activation, activation_kwargs)
        super().__init__(layers_shapes=layers_shape,
                         layers_params=layers_params,
                         epochs=10,
                         lr=.01,
                         activations=activations,
                         criterion=(criterion, criterion_kwargs if criterion_kwargs is not None else dict()),
                         momentum=0)

        init_type = layers_kwargs['initialization_type'] if layers_kwargs is not None \
            else InitializationType.UNIFORM_HE

        self.experiment_name = 'num_lays={}_neurons={}_acts={}_init={}'.format(
            num_layers, hidden_neurons, activation, init_type
        )


def get_layers_settings(input_size, output_size, hidden, num_layers, layers_kwargs=None):
    if layers_kwargs is None:
        layers_kwargs = dict()

    if num_layers == 0:
        layers_shape = (input_size, output_size)
        layers_params = layers_kwargs
    else:
        input_layer_shape = (input_size, hidden)
        output_layer_shape = (hidden, output_size)
        layers_shape = (input_layer_shape, output_layer_shape)
        layers_params = (layers_kwargs, layers_kwargs)
        if num_layers > 2:
            hid_shapes = []
            hid_params = []
            for _ in range(num_layers - 2):
                hid_shapes.append((hidden, hidden))
                hid_params.append(layers_kwargs)
            layers_shape = (input_layer_shape, *hid_shapes, output_layer_shape)
            layers_params = (layers_kwargs, *hid_params, layers_kwargs)
    return tuple(layers_shape), tuple(layers_params)


def get_activations(activations=None, kwargs=None):
    if activations is None:
        return None
    if kwargs is None:
        kwargs = dict()
    output = []
    if isinstance(activations, str):
        output.append((activations, kwargs))
    else:
        for activation in activations:
            output.append((activation, kwargs))

    return tuple(output)