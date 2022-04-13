import itertools

from executors.main import main
from configs import MNISTConfig, CIFAR10Config, MLPConfig
from netlib.linear import InitializationType

if __name__ == "__main__":
    data_config = MNISTConfig(with_augs=False)
    print(type(data_config).__name__)
    inits = [getattr(InitializationType, i) for i in list(InitializationType.__dict__) if i[0] not in '_']
    hidden_neurons = [64, 128]
    activations = ['ReLU', 'Sigmoid', 'Tanh']
    num_layers = (2, 3)

    # init_type, hidden_neurons, activations, num_layers
    products_1 = list(itertools.product(inits, hidden_neurons, (None,), (1,)))
    products_2 = list(itertools.product(inits, hidden_neurons, activations, (2,)))
    products_3 = list(itertools.product(inits, hidden_neurons, list(itertools.product(activations, activations)), (3,)))
    products = (*products_1, *products_2, *products_3)
    for params in products:
        init, neurons, activations, layers = params
        kwargs = dict(initialization_type=init)
        model_config = MLPConfig(data_config.img_shape, 10,
                                 hidden_neurons=neurons,
                                 num_layers=layers,
                                 layers_kwargs=kwargs,
                                 activation=activations)
        main(data_config, model_config, 'overfit', overfit_mode=True)
