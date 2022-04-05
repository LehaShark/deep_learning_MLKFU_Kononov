import numpy as np
from datasets import DATASETS_REGISTRY
from dataloaders import DATALOADERS_REGISTRY
from transforms import COMPOSES_REGISTRY, TRANSFORMS_REGISTRY
from configs import MNISTConfig, CIFAR10Config


def main(config, first_batch: bool = False, counter:int = 0):
    np.random.seed(1)
    registries = dict(transforms_registry=TRANSFORMS_REGISTRY,
                      composes_registry=COMPOSES_REGISTRY,
                      datasets_registry=DATASETS_REGISTRY,
                      dataloader_registry=DATALOADERS_REGISTRY)

    train_dataset, train_dataloader = config.train.get_data(**registries)
    test_dataset, test_dataloader = config.test.get_data(**registries)


    for i, (images, labels) in enumerate(train_dataloader.batch_generator()):
        if first_batch:
            if i == 0:
                train_dataloader.show_batch(first_batch, counter)
        elif config.show_batch and i % config.show_each == 0:
            train_dataloader.show_batch()

    for i, (images, labels) in enumerate(test_dataloader.batch_generator()):
        if first_batch:
            if i == 0:
                test_dataloader.show_batch(first_batch, counter)
        elif config.show_batch and i % config.show_each == 0:
            test_dataloader.show_batch()

def main_fb(dataset_config):
    counter = 1
    while True:
        config = dataset_config(transforms_counter=counter)
        main(config, first_batch=True, counter=counter)

        if counter == config.num_transform:
            break
        else:
            counter += 1


if __name__ == "__main__":
    for dataset_config in (MNISTConfig,
                           CIFAR10Config,
                           ):
        main(dataset_config())