from dataloaders.dataloader import SampleType
from configs.base import DataConfig, Config
from configs.base import ROOT_DIR
import os

base_url = "http://yann.lecun.com/exdb/mnist/"

class MNISTConfig(Config):
    def __init__(self, transforms_counter: int = None):

        path = os.path.join(ROOT_DIR, "data/MNIST/")
        batch_size = 32
        transform = [('Pad', dict(padding=(2, 2), save_size=True), .5),
                     ('GaussianNoise', dict(mean=127, std=255)),
                     ('Normalize', dict(mean=127, std=255))
                     ]

        self.num_transform = len(transform)
        if transforms_counter:
            transform = transform[-transforms_counter:]

        super().__init__(train=DataConfig(dataset=('MNIST', dict(path=path, is_train=True)),
                                          transform=('Compose', transform),
                                          target_transform=('ToOneHot', dict(num_classes=10)),
                                          dataloader=('DataLoader', dict(batch_size=batch_size,
                                                                         sample_type=SampleType.UPSAMPLE))),
                         test=DataConfig(dataset=('MNIST', dict(path=path, is_train=False)),
                                         dataloader=('DataLoader', dict(batch_size=batch_size))),
                         show_dataset=True,
                         show_batch=True,
                         show_each=100)