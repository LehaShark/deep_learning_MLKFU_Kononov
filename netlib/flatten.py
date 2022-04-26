from netlib.module import Module


class Flatten(Module):

    def forward(self, tensor):
        return tensor.reshape(tensor.shape[0], -1)

    def backward(self, *args, **kwargs):
        pass

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
