from abc import ABC

import numpy as np

from netlib.layer import InitializationType, Layer
from netlib.module import Module
from tensor import Parameter

class BaseConvClass(Layer, ABC):
    def __init__(self, input_depth: int, output_depth: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 use_bias: bool = True,
                 initialization_type: InitializationType = InitializationType.UNIFORM_HE,
                 dtype=np.float32):
        super(BaseConvClass, self).__init__(initialization_type)
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._indexes = None

        self.weight = Parameter(np.empty((self.output_depth, self.input_depth, self.kernel_size, self.kernel_size), dtype=dtype))
        self.bias = Parameter(np.empty(output_depth, dtype=dtype)) if use_bias else None
        self._initialization_type = initialization_type
        self.reset_parameters()


    def _output_len(self, length_axis, length_kernel, stride):
        return (length_axis - length_kernel) // stride + 1


    def _pad(self, input, padding: int):
        shape = input.shape
        padded_input = np.zeros((*shape[:-2], shape[-2] + padding * 2, shape[-1] + padding * 2))
        padded_input[..., padding:-padding, padding:-padding] = input

        return padded_input

    def __repr__(self):
        s = ('{input_depth}, {output_depth}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Convolution(BaseConvClass):
    # def __init__(self, input_depth, output_depth):
    #     super().__init__(input_depth=input_depth, output_depth=output_depth)

    def forward(self, input_):
        output = self._conv(input_, self.weight.data, self.bias.data, self.stride, self.padding)
        return output

    def backward(self, grad_out):
        k, i, j = self._indexes
        grad_out_reshaped = grad_out.reshape(self.output_depth, -1)
        weights_reshaped = self.weight.data.reshape(self.output_depth, -1)

        input_ = self._pad(self._saved_input, self.padding)

        # if sum(self.padding) > 0 else self._saved_input

        input_kernels = input_[..., k, i, j].reshape(-1, np.prod((*self.kernel_size, self.input_depth)))
        self.weight.grad = np.dot(grad_out_reshaped, input_kernels).reshape(self.weight.shape)

        # grad = np.dot(weights_reshaped.T, grad_out_reshaped)
        zeros = np.zeros_like(input_)
        zeros[..., k, i, j] += np.dot(weights_reshaped.T, grad_out_reshaped).T.reshape((self._saved_input.shape[0], -1))

        if self.bias is not None:
            self.bias.grad = grad_out.sum((2, 3)).mean(0)

        return zeros[..., self.padding:-self.padding, self.padding:-self.padding]

    def _conv(self, input, kernel, bias=None, stride: int = 1, padding: int = 0):
        input = self._pad(input, padding)
        _, _, input_height, input_width = input.shape
        count_kernels, kernel_depth, kernel_height, kernel_width = kernel.shape
        output_width = self._output_len(input_width, kernel_width, stride)
        output_height = self._output_len(input_height, kernel_height, stride)

        k = np.tile(np.repeat(np.arange(kernel_depth), kernel_height * kernel_width), output_width * output_height)
        i = self._get_kernel_indexes(output_width, kernel_height, stride, kernel_depth, repeat_axis=1)
        j = self._get_kernel_indexes(output_height, kernel_width, stride, kernel_depth, repeat_axis=0)

        kernel_shape = kernel.reshape(-1, kernel_depth*kernel_width*kernel_height)
        matrix = input[:, k, i, j].reshape(-1, kernel_depth*kernel_width*kernel_height).T
        out = np.dot(kernel_shape, matrix)

        self._indexes = (k, i, j)
        return out.reshape((count_kernels, -1, output_width, output_height)).transpose(1, 0, 2, 3)




# def get_matrix(self, tensor_shape, kernel_height, kernel_width, padding: int = 1, stride: int = 1):
    #     """
    #     :param x_shape (tuple): shape of input tensor
    #     :param kernel_height (int): filter size
    #     :param kernel_width (int): filter size
    #     :param padding (int)
    #     :param stride (int)
    #     :return: k, i, j - indices
    #     """
    #
    #     # tensor = np.arange(64 * 2 * 14 * 14).reshape((64, 2, 14, 14))
    #     # tensor = np.arange(1*2*3*3).reshape((1, 2, 3, 3))
    #
    #     # kernel_size = 3
    #     # kernel = np.arange()
    #
    #     _, tensor_depth, tensor_height, tensor_width = tensor_shape
    #
    #     output_width = self._output_len(tensor_width, kernel_width, stride)
    #     output_height = self._output_len(tensor_height, kernel_height, stride)
    #
    #     k_base = np.arange(tensor_depth)
    #     i_base = np.tile(np.arange(kernel_width), output_width).reshape(-1, kernel_width)
    #     j_base = np.tile(np.arange(kernel_height), output_height).reshape(-1, kernel_height)
    #
    #     i_base += stride * np.arange(output_width).reshape(-1, 1)
    #     j_base += stride * np.arange(output_height).reshape(-1, 1)
    #
    #     k = k_base.repeat(kernel_width * kernel_height)
    #     i = np.tile(np.repeat(i_base, kernel_width, axis=1), tensor_depth)  # .reshape(-1, tensor_depth * kernel_size**2)
    #     j = np.tile(np.repeat(j_base, kernel_height, axis=0), tensor_depth)
    #
    #     k = np.tile(k, output_width * output_height)
    #     i = np.repeat(i, output_width, axis=0).reshape(-1)
    #     j = np.tile(j.reshape(-1), output_height)
    #
    #     matrix = tensor[:, k, i, j].reshape(-1, tensor_depth * kernel_height * kernel_width).T
    #
    #     # output = np.dot(matrix)
    # def tensor2matrix(self, x, kernel_height, kernel_width, padding=1, stride=1):
    #     """
    #     :param x: input tensor
    #     :param kernel_height (int): filter size
    #     :param kernel_width (int): filter size
    #     :param padding (int)
    #     :param stride (int)
    #     :return: converted matrix
    #     """
    #     batch_size, x_depth, x_height, x_width = x.shape
    #     output_width = self._output_len(x_width, kernel_width, stride)
    #     output_height = self._output_len(x_height, kernel_height, stride)
    #
    #     k, i, j = self.get_indices(x.shape, kernel_height, kernel_width, padding, stride)
    #     x_col = x[:, k, i, j].reshape(batch_size, x_depth*kernel_height*kernel_width, output_height*output_width)
    #     result = x_col.transpose(1, 2, 0).reshape(x_depth*kernel_height*kernel_width, -1)
    #     return result




if __name__ == "__main__":
    tensor = np.arange(64 * 2 * 14 * 14).reshape((64, 2, 14, 14))
    bcc = BaseConvClass()
    tensor = bcc.tensor2matrix(tensor, 3, 3)
    print(1)