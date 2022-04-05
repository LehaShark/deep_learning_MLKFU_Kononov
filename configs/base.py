from netlib.module import Module
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataConfig(object):
    def __init__(self, dataset: tuple, dataloader: tuple, transform: tuple = None,
                 target_transform: tuple = None):
        self.dataset = dataset
        self.dataloader = dataloader
        self.transform = transform
        self.target_transform = target_transform

    def get_data(self, transforms_registry, composes_registry, datasets_registry, dataloader_registry):

        def _registry_transforms(transforms):
            if transforms is None:
                return None

            if isinstance(transforms[1], (list, tuple)):
                tf = [transforms_registry.get(tfm) for tfm in transforms[1]]
                return composes_registry.get((transforms[0], dict(transforms=tf)))

            return transforms_registry.get(transforms)

        dataset = datasets_registry.get((self.dataset[0],
                                         dict(self.dataset[1],
                                              transform=_registry_transforms(self.transform),
                                              target_transform=_registry_transforms(self.target_transform))))

        dataloader = dataloader_registry.get((self.dataloader[0], dict(self.dataloader[1], dataset=dataset)))
        return dataset, dataloader


class DatasetConfig(object):
    def __init__(self,
                 img_shape: tuple,
                 train: DataConfig,
                 valid: DataConfig = None,
                 test: DataConfig = None,
                 show_dataset: bool = None,
                 show_batch: bool = None,
                 show_each: int = None):
        self.train = train
        self.valid = valid
        self.test = test
        self.img_shape = img_shape
        self.show_dataset = show_dataset
        self.show_batch = show_batch
        self.show_each = show_each


class ModelConfig(object):
    def __init__(self,
                 layers_shapes: tuple,
                 layers_params: tuple,
                 epochs: int,
                 lr: float,
                 momentum: float,
                 activations: tuple,
                 criterion: tuple):
        self.layers_shapes = layers_shapes
        self.layers_params = layers_params
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self._activations = activations
        self._criterion = criterion

    @property
    def get_activations(self):
        return self._activations

    @property
    def get_criterion(self):
        return self._criterion
