from netlib.linear import InitializationType
from configs import ConvNNConfig, CIFAR10Config, MNISTConfig

from main import main

if __name__ == "__main__":  # , CIFAR10Config(with_augs=False)):
    # for data_config, logname, shape in zip([CIFAR10Config(with_augs=False), CIFAR10Config(),
    #                                         MNISTConfig(with_augs=False), MNISTConfig()],
    #                                        ['wo_augs_cifar', 'with_augs_cifar', 'wo_augs_mnist', 'with_augs_mnist'],
    #                                        [(32, 32), (32, 32), (28, 28), (28, 28)]):
    for data_config, logname, shape in zip([CIFAR10Config()],
                                           ['with_augs_cifar'],
                                           [(32, 32)]):
        # data_config = MNISTConfig(with_augs=False)

        model_config = ConvNNConfig(image_shape=shape)
        main(data_config, model_config, logname)
