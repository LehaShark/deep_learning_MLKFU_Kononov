import itertools
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from datasets import DATASETS_REGISTRY
from dataloaders import DATALOADERS_REGISTRY
from epoch import Epoch, PhaseKeysDict
from models import MLP, ConvNN
from optim import SGD
from transforms import COMPOSES_REGISTRY, TRANSFORMS_REGISTRY
from losses import LOSSES_REGISTRY
from netlib import ACTIVATIONS_REGISTRY, POOLING_REGISTRY


def main(dataset_config, model_config, logname, overfit_mode: bool = False):
    np.random.seed(1)
    dataset_registries = dict(transforms_registry=TRANSFORMS_REGISTRY,
                              composes_registry=COMPOSES_REGISTRY,
                              datasets_registry=DATASETS_REGISTRY,
                              dataloader_registry=DATALOADERS_REGISTRY)

    model_registires = dict(activations_registry=ACTIVATIONS_REGISTRY,
                            losses_registry=LOSSES_REGISTRY)

    train_dataset, train_dataloader = dataset_config.train.get_data(**dataset_registries)
    test_dataset, test_dataloader = dataset_config.test.get_data(**dataset_registries)

    activations = None
    if model_config.get_activations is not None:
        activations = tuple([ACTIVATIONS_REGISTRY.get(acts) for acts in model_config.get_activations])
    pooling = POOLING_REGISTRY.get(model_config.get_pooling) if model_config.get_pooling is not None else None

    model = ConvNN(model_config, activations, pooling)
    criterion = LOSSES_REGISTRY.get(model_config.get_criterion)
    optimizer = SGD(model.named_parameters(), model_config.lr, model_config.momentum)
    writer = SummaryWriter(log_dir=os.path.join(model_config.ROOT_DIR, 'logs', logname, model_config.experiment_name))

    if overfit_mode:
        train_dataloader.set_overfit_mode()
        test_dataloader.set_overfit_mode()

    dataloaders = {dataset_config.train_key: train_dataloader,
                   dataset_config.test_key: test_dataloader}

    phase_keys = PhaseKeysDict(train=dataset_config.train_key, test=dataset_config.test_key)

    if dataset_config.show_dataset:
        train_dataset.show_statistics()
        test_dataset.show_statistics()

    epoch_manager = Epoch(model=model, criterion=criterion, optimizer=optimizer, writer=writer,
                                 dataloaders=dataloaders, phase_keys=phase_keys)

    for i_epoch in range(model_config.epochs):
        epoch_manager.train(i_epoch, show_each=100)
        epoch_manager.save_model(model_config.SAVE_PATH+'_'+logname, i_epoch)
        epoch_manager.test(show_each=100)
