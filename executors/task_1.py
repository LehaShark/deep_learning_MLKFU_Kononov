import numpy as np
from datasets import DATASETS_REGISTRY
from dataloaders import DATALOADERS_REGISTRY
from transforms import COMPOSES_REGISTRY, TRANSFORMS_REGISTRY
from configs.cifar10_config import CIFAR10Config
from configs.mnist_config import MNISTConfig

def main(config, first_batch: bool = False, counter:int = 0):
    np.random.seed(1)
    registries = dict(transforms_registry=TRANSFORMS_REGISTRY,
                      composes_registry=COMPOSES_REGISTRY,
                      datasets_registry=DATASETS_REGISTRY,
                      dataloader_registry=DATALOADERS_REGISTRY)

    train_dataset, train_dataloader = config.train.get_data(**registries)
    test_dataset, test_dataloader = config.test.get_data(**registries)

    if config.show_dataset:
        train_dataset.show_statistics()
        test_dataset.show_statistics()

    for i, (images, labels) in enumerate(train_dataloader.batch_generator()):
        if first_batch:
            if i == 0:
                train_dataloader.show_batch(first_batch, counter)
        elif config.show_batch and i % config.show_each == 0:
            train_dataloader.show_batch()


if __name__ == "__main__":
    for dataset_config in (MNISTConfig, CIFAR10Config):
        main(dataset_config())