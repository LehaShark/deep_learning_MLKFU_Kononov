import os
import time

import numpy as np

from configs.base import ModelConfig
from netlib.linear import InitializationType
from torch import nn


class ConvNNConfig(ModelConfig):
    def __init__(self,
                 layers_dict=None,
                 activation='ReLU',
                 activation_kwargs=None,
                 last_activation: bool = False,
                 pooling='MaxPool2d',
                 pooling_kwargs=None,
                 criterion='CrossEntropyLoss',
                 criterion_kwargs=None,
                 image_shape=(32, 32)):
        if layers_dict is None:
            layers_dict = dict(Conv2d=(dict(input_depth=1, output_depth=16, kernel_shape=5),
                                       dict(input_depth=16, output_depth=32, kernel_shape=5)))

        out_image_shape = image_shape
        for lay in layers_dict['Conv2d']:
            kernel = lay['kernel_shape']
            pad = lay['padding'] if 'padding' in lay else 0
            stride = lay['stride'] if 'stride' in lay else 1

            out_image_shape = (out_image_shape[0] - kernel + pad * 2) / stride + 1, \
                              (out_image_shape[1] - kernel + pad * 2) / stride + 1
            out_image_shape = out_image_shape[0] // 2, out_image_shape[1] // 2

        if 'Linear' not in layers_dict:
            fif = int(np.prod(out_image_shape) * layers_dict['Conv2d'][-1]['output_depth'])
            layers_dict['Linear'] = (dict(input_features=fif, output_features=128),
                                     dict(input_features=128, output_features=10))

        if hasattr(activation, '__len__') and not isinstance(activation, str):
            activation_kwargs = activation_kwargs if activation_kwargs is not None else [dict()] * len(activation)
            activations = [(a, k) for a, k in zip(activation, activation_kwargs)]
        else:
            activations = [(activation, (activation_kwargs if activation_kwargs is not None else dict()))] * 2

        super().__init__(layers_dict=layers_dict,
                         epochs=20,
                         lr=.001,
                         activations=activations,
                         criterion=(criterion, criterion_kwargs if criterion_kwargs is not None else dict()),
                         momentum=0,
                         last_activation=last_activation)
        self._pooling = pooling, pooling_kwargs if pooling_kwargs is not None else dict(kernel_shape=2)

        init_type = layers_dict['Conv2d'][0]['initialization_type'].name \
            if 'initialization_type' in layers_dict['Conv2d'][0] is not None \
            else InitializationType.NORMAL_XAVIER.name

        self.experiment_name = 'num_convs={}_acts={}_init={}_{}'.format(
            len(self.layers_dict['Conv2d']), activation, init_type, time.time()
        )

        self.SAVE_PATH = os.path.join(self.ROOT_DIR, 'checkpoints', type(self).__name__,
                                      f'lr{self.lr}_' + self.experiment_name)

        self.LOAD_PATH = os.path.join(self.ROOT_DIR, 'checkpoints', type(self).__name__,
                                      'lr0.001_start_filters32_1651509451.4862506_overfitted_on_batch', '69.pth')

    @property
    def get_pooling(self):
        return self._pooling


class MLPConfig(ModelConfig):
    def __init__(self, input_size, output_size,
                 hidden_neurons=200,
                 num_layers=2,
                 layers_kwargs=None,
                 activation='ReLU',
                 activation_kwargs=None,
                 criterion='CrossEntropyLoss',
                 criterion_kwargs=None):
        input_size = int(np.prod(np.array(input_size))) if hasattr(input_size, '__len__') and len(input_size) > 1 \
            else input_size
        layers_shape, layers_params = get_layers_settings(input_size, output_size, hidden_neurons,
                                                          num_layers, layers_kwargs)
        layers_dict = dict(Linear=tuple(zip(layers_shape, layers_params)))
        activations = get_activations(activation, activation_kwargs)
        super().__init__(layers_dict=layers_dict,
                         epochs=50,
                         lr=.001,
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
