from netlib.layer import Layer
from tensor import Tensor


class Model(object):
    def __init__(self):
        self.requires_grad = True

    def __call__(self, tensor, *args, **kwargs):
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor, requires_grad=self.requires_grad)
        return self.forward(tensor, *args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def state_dict(self):
        return dict(self.named_parameters())

    def load_state_dict(self, state: dict):
        self_state = self.state_dict()
        if len(self_state) > len(state):
            bigger, smaller = self_state, state
        else:
            bigger, smaller = state, self_state

        for name, param in smaller.items():
            if name in bigger.keys():
                layer_name, type_name = name.split('.')
                layer = getattr(self, layer_name)
                setattr(layer, type_name, state[name])

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self):
        for name, param in self.__dict__.items():
            if isinstance(param, Layer):
                for layer_name, layer_param in param.named_parameters():
                    yield name + "." + layer_name, layer_param
