import os
from configs.base import DataConfig, DatasetConfig, ROOT_DIR

class CIFAR10Config(DatasetConfig):
    def __init__(self, transforms_counter: int = None, aug: bool = True):
        super().__init__()
        path = os.path.join(ROOT_DIR, "data/CIFAR10/")
        batch_size = 32
        img_shape = (32, 32, 3)
        transform = [('Pad', dict(padding=(2, 2), save_size=True), .5),
                     ('RandomCrop', dict(size=28, save_size=True), .5),
                     ('RandomAffine', dict(degrees=15,
                                           translate=(0.0, 0.15),
                                           scale=(0.9, 1.)), .5),
                     ('SaltAndPepperNoise', dict(threshold=0.01)),
                     ('Normalize', dict(mean=127, std=255))
                     ]

        self.num_transform = len(transform)
        if transforms_counter:
            transform = transform[-transforms_counter:]

        transform = transform if aug else [('Normalize', dict(mean=127, std=255))]

        super().__init__(train=DataConfig(dataset=('CIFAR10', dict(path=path, is_train=True)),
                                          transform=('Compose', transform),
                                          #target_transform=('ToOneHot', dict(num_classes=10)),
                                          dataloader=('DataLoader', dict(batch_size=batch_size))),
                         test=DataConfig(dataset=('CIFAR10', dict(path=path, is_train=False)),
                                         transform=('Normalize', dict(mean=127, std=255)),
                                         dataloader=('DataLoader', dict(batch_size=batch_size))),
                         img_shape=img_shape,
                         show_dataset=True,
                         show_batch=True,
                         show_each=100)
