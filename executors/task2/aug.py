import numpy as np
from datasets import DATASETS_REGISTRY
from dataloaders import DATALOADERS_REGISTRY
from epoch import Epoch
from models.models import MLP
from optim import SGD
import tensor
from transforms import COMPOSES_REGISTRY, TRANSFORMS_REGISTRY
from losses import LOSSES_REGISTRY
from configs import MNISTConfig, CIFAR10Config, MLPConfig
from netlib import ACTIVATIONS_REGISTRY


def main(dataset_config, model_config):
    np.random.seed(1)
    dataset_registries = dict(transforms_registry=TRANSFORMS_REGISTRY,
                              composes_registry=COMPOSES_REGISTRY,
                              datasets_registry=DATASETS_REGISTRY,
                              dataloader_registry=DATALOADERS_REGISTRY)

    train_dataset, train_dataloader = dataset_config.train.get_data(**dataset_registries)
    test_dataset, test_dataloader = dataset_config.test.get_data(**dataset_registries)

    activations = tuple([ACTIVATIONS_REGISTRY.get(acts) for acts in model_config.get_activations])
    criterion = LOSSES_REGISTRY.get(model_config.get_criterion)
    model = MLP(model_config, activations)
    optimizer = SGD(model.named_parameters(), model_config.lr, model_config.momentum)

    if dataset_config.show_dataset:
        train_dataset.show_statistics()
        test_dataset.show_statistics()


    epoch_manager = Epoch(model=model, criterion=criterion, optimizer=optimizer)
    for i_epoch in range(model_config.epochs):
        epoch_manager.step(train_dataloader, is_train=True, i_epoch=i_epoch)

    with tensor.no_grad(model):
        epoch_manager.step(test_dataloader, is_train=False)


if __name__ == "__main__":
    dataset_config = MNISTConfig()
    print(type(dataset_config).__name__)
    main(dataset_config, MLPConfig(dataset_config.img_shape, 10))
