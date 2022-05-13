import itertools
import unittest
import numpy as np
import torch
from torch import nn
from models import ConvNN

from losses import CrossEntropyLoss, NLLLoss
from netlib import Linear, ReLU, LogSoftmax, Tanh, Sigmoid, Convolution, MaxPool, Flatten
from tensor import Tensor, Parameter


def build_models(linear_shapes,
                 activations=None,
                 conv_kwargs=None,
                 torch_conv_kwargs=None,
                 depths=None,
                 poolings=None,
                 pooling_kernel=2,
                 init_type=None):
    model = []
    model_torch = []
    last_depth = 1

    if None not in [depths, conv_kwargs, torch_conv_kwargs]:
        for i, (in_d, out_d, k, torch_k) in enumerate(zip(depths[:-1], depths[1:], conv_kwargs, torch_conv_kwargs)):

            layer = Convolution(in_d, out_d, **k) if init_type is None\
                else Convolution(in_d, out_d, initialization_type=init_type, **k)
            torch_layer = nn.Conv2d(in_d, out_d, **torch_k)
            torch_layer.weight = nn.Parameter(torch.from_numpy(layer.weight.data))
            torch_layer.bias = nn.Parameter(torch.from_numpy(layer.bias.data))

            model.append(layer)
            model_torch.append(torch_layer)

            if activations is not None:
                model.append(activations[0]())
                model_torch.append(activations[1]())

            if poolings is not None:
                model.append(poolings[0](pooling_kernel))
                model_torch.append(poolings[1](pooling_kernel))

            last_depth = out_d

    for i, (in_shape, out_shape) in enumerate(zip(linear_shapes[:-1], linear_shapes[1:])):
        if i == 0:
            model.append(Flatten())
            model_torch.append(nn.Flatten())
            layer = Linear(in_shape * last_depth, out_shape)
            torch_layer = nn.Linear(in_shape * last_depth, out_shape)
        else:
            if activations is not None:
                model.append(activations[0]())
                model_torch.append(activations[1]())
            layer = Linear(in_shape, out_shape)
            torch_layer = nn.Linear(in_shape, out_shape)

        torch_layer.weight = nn.Parameter(torch.from_numpy(layer.weight.data))
        torch_layer.bias = nn.Parameter(torch.from_numpy(layer.bias.data))
        model.append(layer)
        model_torch.append(torch_layer)

    return model, model_torch


def test_models(unit_test,
                models,
                test_input,
                test_targets,
                loss=(CrossEntropyLoss, nn.CrossEntropyLoss),
                last_activation=None):
    output = Tensor(test_input, requires_grad=True)
    output_torch = torch.tensor(test_input, requires_grad=True)

    for i, (layer, layer_torch) in enumerate(zip(*models)):
        # forward
        output_torch = layer_torch(output_torch)
        output = layer(output)

        unit_test.assertEqual(np.array_equal(output.shape, output_torch.detach().numpy().shape), True,
                              (output.shape, output_torch.detach().numpy().shape))
        unit_test.assertEqual(np.allclose(output.data, output_torch.detach().numpy(), atol=1e-6), True)

    if last_activation is not None:
        output = last_activation[0]()(output)
        output_torch = last_activation[1]()(output_torch)

    # loss
    loss_ = loss[0]()
    loss_torch = loss[1]()

    result = loss_(output, test_targets)
    result_torch = loss_torch(output_torch, torch.from_numpy(test_targets))

    unit_test.assertEqual(np.allclose(result.data, result_torch.detach().numpy(), atol=1e-6), True)

    # grads
    result.backward()
    result_torch.backward()

    for layer, layer_torch in zip(*models):
        if hasattr(layer, 'weight') and hasattr(layer_torch, 'weight'):
            unit_test.assertEqual(np.allclose(layer.weight.grad,
                                              layer_torch.weight.grad.detach().numpy(), atol=1e-6),
                                  True)
            unit_test.assertEqual(np.allclose(layer.bias.grad, layer_torch.bias.grad.detach().numpy(), atol=1e-6), True)
