import os
from configs.base import DataConfig, DatasetConfig
from dataloaders.dataloader import SampleType

base_url = "http://yann.lecun.com/exdb/mnist/"

class MNISTConfig(DatasetConfig):
    def __init__(self, transforms_counter: int = None, with_augs: bool = True):
        path = os.path.join("data/MNIST/")
        batch_size = 32
        img_shape = (28, 28)
        transform = [('Pad', dict(padding=(2, 2), save_size=True), .5),
                     ('RandomCrop', dict(size=24, save_size=True), .5),
                     ('RandomAffine', dict(degrees=(0, 15),
                                           translate=(0.0, 0.15),
                                           scale=(0.9, 1.)), .5),
                     ('GaussianNoise', dict(mean=127, std=255)),
                     ('Normalize', dict(mean=127, std=255))
                     ]

        self.num_transform = len(transform)
        if transforms_counter:
            transform = transform[-transforms_counter:]

        transform = transform if with_augs else [('Normalize', dict(mean=127, std=255))]

        super().__init__(train=DataConfig(dataset=('MNIST', dict(path=path, is_train=True)),
                                          transform=('Compose', transform),
                                          #target_transform=('ToOneHot', dict(num_classes=10)),
                                          dataloader=('DataLoader', dict(sample_type=SampleType.UPSAMPLE,
                                                                        # probabilities=[0.5, 0.2, 0.3, 0.7, 0.4, 0.9,
                                                                        #                0.6, 0.9, 0.8, 0.7],
                                                                        batch_size=batch_size,
                                                                        ))),
                         test=DataConfig(dataset=('MNIST', dict(path=path, is_train=False)),
                                         transform=('Normalize', dict(mean=127, std=255)),
                                         dataloader=('DataLoader', dict(sample_type=SampleType.UPSAMPLE,
                                                                        # probabilities=[0.5, 0.2, 0.3, 0.7, 0.4, 0.9,
                                                                        #                0.6, 0.9, 0.8, 0.7],
                                                                        batch_size=batch_size,
                                                                        ))),
                         img_shape=img_shape,
                         show_dataset=True,
                         show_batch=True,
                         show_each=100)
