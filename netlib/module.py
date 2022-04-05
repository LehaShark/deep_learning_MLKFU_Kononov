import copy


class Module(object):
    def __init__(self):
        self._saved_tensor = None

    @property
    def _saved_input(self):
        return self._saved_tensor.data

    def set_grad(self, grad, retain_graph: bool = False):
        if self._saved_tensor.is_optimize:
            self._saved_tensor.grad = grad
        if not retain_graph:
            self._saved_tensor.grad_fn = None

    def __call__(self, tensor, *args, **kwargs):

        output = copy.deepcopy(tensor)

        if tensor.requires_grad:
            if output.grad_fn is None:
                output.start_graph()

            output.grad_fn.append(self)
            self._saved_tensor = tensor

        output.data = self.forward(tensor.data, *args, **kwargs)
        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()
