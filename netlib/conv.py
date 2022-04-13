from Layers.InitClass import BaseLayerClass

class BaseConvClass(BaseLayerClass):
    def get_indices(self, x_shape, kernel_height, kernel_width, padding=1, stride=1):
        """
        :param x_shape (tuple): shape of input tensor
        :param kernel_height (int): filter size
        :param kernel_width (int): filter size
        :param padding (int)
        :param stride (int)
        :return: k, i, j - indices
        """

    def tensor2matrix(self, x, kernel_height, kernel_width, padding=1, stride=1):
        """
        :param x: input tensor
        :param kernel_height (int): filter size
        :param kernel_width (int): filter size
        :param padding (int)
        :param stride (int)
        :return: converted matrix
        """

class Convolution(BaseConvClass):
    def __init__(self, kernel_size, nrof_filters, kernel_depth, zero_pad, stride, use_bias, initialization_type):
        pass

    def trainable(self):
        return True

    def init_weights(self):
        pass

    def __call__(self, input, phase='eval'):
        """
        :param input: input tensor, shape=(batch_size, depth, height, width)
        :param phase: ['train', 'eval']
        :return: output of convolutional layer
        """

class Pooling(BaseConvClass):
    def __init__(self, kernel_size, stride, type='Max'):
        """
        :param kernel_size (int): kernel size
        :param stride (int): stride
        :param type (string): ['Max', 'Avg']
        """

    def __call__(self, x, phase):
        """
       :param input: input tensor, shape=(batch_size, depth, height, width)
       :param phase: ['train', 'eval']
       :return: output of pooling layer
       """
