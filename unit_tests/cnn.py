import itertools
import unittest
import numpy as np
import torch
from torch import nn
from models import ConvNN

from losses import CrossEntropyLoss, NLLLoss
from netlib import Linear, ReLU, LogSoftmax, Tanh, Sigmoid, Convolution, MaxPool, Flatten
from tensor import Tensor, Parameter

from base import build_models, test_models


class TestConvNN(unittest.TestCase):

    def test1(self):
        batch_size = 32
        img_shape = 28, 28
        classes = 10
        test_input = np.random.normal(size=(batch_size, 1, *img_shape))
        test_targets = np.random.randint(0, classes, size=batch_size).astype(np.int64)

        linear_shapes = (7 * 7, classes)
        activations = (ReLU, nn.ReLU)
        conv_kwargs = [dict(kernel_size=5, stride=1, padding=2)] * 2
        torch_conv_kwargs = [dict(kernel_size=5, stride=1, padding=2)] * 2
        layers_depth = (1, 16, 32)
        poolings = (MaxPool, nn.MaxPool2d)
        models = build_models(linear_shapes, activations, conv_kwargs, torch_conv_kwargs, layers_depth, poolings)
        test_models(self, models, test_input, test_targets)



if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    unittest.main()