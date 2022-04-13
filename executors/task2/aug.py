import itertools

from configs import MNISTConfig, CIFAR10Config, MLPConfig
from executors.main import main
from netlib.linear import InitializationType

if __name__ == "__main__":
    data_config = MNISTConfig()
    print(type(data_config).__name__)
    inits = [getattr(InitializationType, i) for i in list(InitializationType.__dict__) if i[0] not in '_']
    activations = ['ReLU', 'Sigmoid', 'Tanh']
    products = list(itertools.product(inits, activations))
    for params in products:
        init, activations = params
        kwargs = dict(initialization_type=init)
        model_config = MLPConfig(data_config.img_shape, 10,
                                 layers_kwargs=kwargs,
                                 activation=activations)
        main(data_config, model_config, 'with_augs,')



    # dataset_config = MNISTConfig()
    # print(type(dataset_config).__name__)
    # main(dataset_config, MLPConfig(dataset_config.img_shape, 10), 'aug')
