import itertools
import unittest
import numpy as np
import torch
from torch.nn import functional
from torch import nn
from torch.autograd import gradcheck

from losses import CrossEntropyLoss, NLLLoss
from netlib import Linear, ReLU, LogSoftmax, Tanh, Sigmoid
from tensor import Tensor


def test_base(unittest,
              loss=(CrossEntropyLoss, nn.CrossEntropyLoss),
              activation: list = None,
              layers_shape: tuple = (28 * 28, 10),
              batch_size=32):
    img_shape, classes = layers_shape[0], layers_shape[-1]
    test_input = np.random.normal(size=(batch_size, img_shape))
    test_targets = np.random.randint(0, classes, size=batch_size).astype(np.int64)

    fc_layers = []
    fc_layers_torch = []
    for input_shape, output_shape in zip(layers_shape[:-1], layers_shape[1:]):
        fc = Linear(input_shape, output_shape)
        fc_torch = nn.Linear(input_shape, output_shape)
        fc_torch.weight = nn.Parameter(torch.from_numpy(fc.weight.data.T))
        fc_torch.bias = nn.Parameter(torch.from_numpy(fc.bias.data.T))

        fc_layers.append(fc)
        fc_layers_torch.append(fc_torch)

    output = Tensor(test_input, requires_grad=True)
    output_torch = torch.tensor(test_input, requires_grad=True)

    layres_queue = []
    layres_queue_torch = []
    for i, (layer, layer_torch) in enumerate(zip(fc_layers, fc_layers_torch)):

        # forward
        output = layer(output)
        output_torch = layer_torch(output_torch)

        layres_queue.append(layer)
        layres_queue_torch.append(layer_torch)
        unittest.assertEqual(np.array_equal(output.shape, output_torch.detach().numpy().shape), True,
                             (output.shape, output_torch.detach().numpy().shape))
        unittest.assertEqual(np.allclose(output.data, output_torch.detach().numpy(), atol=1e-6), True)

        # activations
        if activation is not None and len(activation) > i:
            activation_ = activation[i][0]()
            activation_torch = activation[i][1]()

            output = activation_(output)
            output_torch = activation_torch(output_torch)

            layres_queue.append(activation_)
            layres_queue_torch.append(activation_torch)
            unittest.assertEqual(np.allclose(output.data, output_torch.detach().numpy(), atol=1e-6), True)

    # loss
    loss_ = loss[0]()
    loss_torch = loss[1]()

    result = loss_(output, test_targets)
    result_torch = loss_torch(output_torch, torch.from_numpy(test_targets))

    unittest.assertEqual(np.allclose(result.data, result_torch.detach().numpy(), atol=1e-6), True)

    # grads
    result.backward()
    result_torch.backward()

    for layer, layer_torch in zip(fc_layers, fc_layers_torch):
        unittest.assertEqual(np.allclose(layer.weight.grad, layer_torch.weight.grad.T.detach().numpy(), atol=1e-6),
                             True)
        unittest.assertEqual(np.allclose(layer.bias.grad, layer_torch.bias.grad.detach().numpy(), atol=1e-6), True)


class TestMLP(unittest.TestCase):

    def test1(self):
        test_base(self)

    def test_fc_size(self):
        """
        в цикле с разным размером полносвязного слоя проверить значения выхода полносвязного слоя и градиентов
        """
        img_shapes = (28 * 28, 32 * 32, 24 * 24)
        num_classes = (10, 2)

        for img_shape, classes in itertools.product(img_shapes, num_classes):
            test_base(self, layers_shape=(img_shape, classes))

    def test_activation_function(self):
        """
        в цикле по разным функциям активации проверить значения выхода функции активации и градиентов
        """
        layers_shape = (28 * 28, 200, 10)
        activations = [[(ReLU, nn.ReLU)],
                       [(Tanh, nn.Tanh)],
                       [(Sigmoid, nn.Sigmoid)]]
        last_activations = [(LogSoftmax, nn.LogSoftmax)]

        entropy_loss = (CrossEntropyLoss, nn.CrossEntropyLoss)
        nll_loss = (NLLLoss, nn.NLLLoss)

        for activation, loss in itertools.product(activations, [entropy_loss, nll_loss]):
            if loss is nll_loss:
                activation.extend(last_activations)
            test_base(self, layers_shape=layers_shape, loss=loss, activation=activation)

    def test_nrof_layers(self):
        """
        в цикле по количеству слоев проверить -//-
        """
        layers_shapes = ((28 * 28, 10), (28 * 28, 200, 10), (28 * 28, 200, 200, 10), (28 * 28, 200, 200, 200, 10))
        activation = (ReLU, nn.ReLU)

        activations = [[activation for _ in range(len(shape) - 2)] for shape in layers_shapes]

        for shape, act in zip(layers_shapes, activations):
            test_base(self, layers_shape=shape, activation=act)


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    unittest.main()
