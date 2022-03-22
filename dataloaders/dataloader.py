import math
import os
from enum import Enum
# from configs.base import ROOT_DIR
import numpy as np
import matplotlib.pyplot as plt
from transforms.registry import Registry

REGISTRY = Registry('dataloaders')


class SampleType(Enum):
    DEFAULT = 0  # равномерно из всего датасета
    UPSAMPLE = 1  # равномерно из каждого класса, увеличивая размер каждого класса до максимального
    DOWNSAMPLE = 2  # равномерно из каждого класса, уменьшая размер каждого класса до минимального
    PROBABILITIES = 3  # случайно из каждого класса в зависимости от указанных вероятностей


def _shuffle_dataset(dataset):
    indexes = np.arange(len(dataset))
    indexes = np.random.permutation(indexes)
    dataset.data, dataset.targets = dataset.data[indexes], dataset.targets[indexes]
    return dataset


@REGISTRY.register_module
class DataLoader(object):
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False,
                 sample_type: SampleType = SampleType.UPSAMPLE, epoch_size=None, probabilities=None):
        self.dataset = dataset if not shuffle else _shuffle_dataset(dataset)
        self.batch_size = batch_size
        self.epoch_size = epoch_size if epoch_size else len(self.dataset) // self.batch_size
        self._current_batch = None
        self._current_idx = None

        if sample_type != SampleType.DEFAULT and shuffle:
            raise ValueError('sampler option is mutually exclusive with shuffle')
        elif sample_type != SampleType.DEFAULT:
            self._sample(sample_type, probabilities)

    def _sample(self, sample_type, probabilities):
        # shape = self.dataset.data[0].shape
        count = np.bincount(self.dataset.labels)
        threshold, repeats = np.ones_like(self.dataset.classes, dtype=int), np.ones_like(self.dataset.classes,
                                                                                         dtype=int)
        new_data = []
        new_labels = []

        if sample_type == SampleType.UPSAMPLE:
            threshold *= np.max(count)
        elif sample_type == SampleType.DOWNSAMPLE:
            threshold *= np.min(count)

        repeats = np.ceil(threshold / count).astype(np.int32)
        for i, _ in enumerate(self.dataset.classes):
            class_data = self.dataset.data[self.dataset.labels == i]
            class_data = np.repeat(class_data, repeats[i], 0)[:threshold[i]]
            new_data.append(class_data)
            new_labels.append(np.ones(threshold[i]) * i)

        self.dataset.data = np.concatenate(new_data)
        self.dataset.targets = np.concatenate(new_labels).astype(np.int32)
        self.dataset = _shuffle_dataset(self.dataset)

    def batch_generator(self):
        for i in range(self.epoch_size):
            self._current_idx = i * self.batch_size
            end = (i + 1) * self.batch_size
            self._current_batch = self.dataset[self._current_idx:end]
            yield self._current_batch

    def show_batch(self, first_batch: bool = False, counter: int = 0):
        x = int(math.sqrt(len(self._current_batch[0])))
        fig_size = (x, math.ceil(len(self._current_batch[0]) / x))

        fig = plt.figure(figsize=(16, 9))
        for i, img in enumerate(self._current_batch[0]):
            fig.add_subplot(*fig_size, i + 1)
            plt.imshow(img)

        if first_batch:
            path = os.path.join('figures',
                                type(self.dataset).__name__,
                                'first_batch')
            figname = str(counter) + '_' + ''.join(type(e).__name__ + '_' for e in self.dataset.transform.transforms) \
                if self.dataset.transform else 'None'
        else:
            path = os.path.join('figures',
                                type(self.dataset).__name__,
                                'train' if self.dataset.is_train else 'test',
                                ''.join(type(e).__name__ + '_' for e in self.dataset.transform.transforms)
                                if self.dataset.transform else 'None')
            figname = f'{self._current_idx}_batch.png'

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, 'transforms.txt'), "w+") as file:
            file.write(''.join(str(e) + '\n' for e in self.dataset.transform.transforms)
                       if self.dataset.transform else 'None')
        file.close()

        plt.savefig(os.path.join(path, figname))
        plt.close(fig)
