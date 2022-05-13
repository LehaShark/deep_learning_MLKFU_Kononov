import math
import os
import numpy as np
import matplotlib.pyplot as plt

# from configs.base import ROOT_DIR
from enum import Enum

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
                 sample_type: SampleType = SampleType.DEFAULT, epoch_size=None, probabilities=None,
                 overfit: bool = False):
        self.dataset = dataset if not shuffle else _shuffle_dataset(dataset)
        self.batch_size = batch_size
        self.epoch_size = epoch_size if epoch_size else len(self.dataset) // self.batch_size
        self._current_batch = None
        self._current_idx = None
        self.overfit = overfit
        self.sample_type = sample_type
        self.shuffle = shuffle
        self.probabilities = probabilities
        self.show_statistics = True

        self._need_sample = False
        if self.sample_type != SampleType.DEFAULT and self.shuffle:
            raise ValueError('sampler option is mutually exclusive with shuffle')
        elif self.sample_type != SampleType.DEFAULT:
            self._need_sample = True
    def _sample(self):
        count = np.bincount(self.dataset.labels)
        indexes = np.arange(len(self.dataset))
        threshold = np.ones_like(self.dataset.classes, dtype=int)
        # repeats = np.ones_like(self.dataset.classes, dtype=int)
        sample_indexes = []

        if self.sample_type == SampleType.UPSAMPLE:
            threshold *= np.max(count)
        elif self.sample_type == SampleType.DOWNSAMPLE:
            threshold *= np.min(count)
        elif self.sample_type == SampleType.PROBABILITIES and self.probabilities is not None:
            if len(self.probabilities) != len(self.dataset.classes):
                raise ValueError('probabilities len not equal number of classes')
            threshold = np.ceil(self.probabilities * count).astype(np.int32)
        else:
            raise ValueError('SampleType.PROBABILITIES is chosen, but probabilities is None')

        repeats = np.ceil(threshold / count).astype(np.int32)
        for i, _ in enumerate(self.dataset.classes):
            new_indexes = indexes[self.dataset.labels == i]
            np.random.shuffle(new_indexes)
            new_indexes = np.repeat(new_indexes, repeats[i], 0)[:threshold[i]]
            sample_indexes.append(new_indexes)

        sample_indexes = np.concatenate(sample_indexes)
        np.random.shuffle(sample_indexes)

        return sample_indexes

    def set_overfit_mode(self):
        self.overfit = True

    def batch_generator(self):
        indexes = None
        if self._need_sample:
            indexes = self._sample()
            self.epoch_size = len(indexes) // self.batch_size

        if self.show_statistics:
            self.show_statistics = False
            self.dataset.show_statistics(indexes)

        for i in range(self.epoch_size):
            idx = i if not self.overfit else 0
            self._current_idx = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            index = indexes[self._current_idx:end] if indexes is not None else slice(self._current_idx, end)
            self._current_batch = self.dataset[index]
            yield self._current_batch

    def show_batch(self, first_batch: bool = False, counter: int = 0):
        x = int(math.sqrt(len(self._current_batch[0])))
        fig_size = (x, math.ceil(len(self._current_batch[0]) / x))

        fig = plt.figure(figsize=(16, 9))
        for i, img in enumerate(self._current_batch[0]):
            fig.add_subplot(*fig_size, i + 1)
            plt.imshow(img)

        if first_batch:
            path = os.path.join('batch_figures',
                                type(self.dataset).__name__,
                                'first_batch')
            figname = str(counter) + '_' + ''.join(type(e).__name__ + '_' for e in self.dataset.transform.transforms) \
                if self.dataset.transform else 'None'
        else:
            path = os.path.join('batch_figures',
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

    # def show_statistics(self, idx):
    #
    #     sampled_dataset = self.dataset[idx]
    #
    #     count = np.bincount(sampled_dataset.labels)
    #
    #     print('\t---DATASET STAT---')
    #     for i, cls in enumerate(sampled_dataset.classes):
    #         print(f"{i + 1} class: '{cls}':\t {count[i]}")
    #
    #     print(f"overall:\t\t {len(sampled_dataset.labels)}")
# REGISTRY = Registry('dataloaders')
#
#
# class SampleType(Enum):
#     DEFAULT = 0  # равномерно из всего датасета
#     UPSAMPLE = 1  # равномерно из каждого класса, увеличивая размер каждого класса до максимального
#     DOWNSAMPLE = 2  # равномерно из каждого класса, уменьшая размер каждого класса до минимального
#     PROBABILITIES = 3  # случайно из каждого класса в зависимости от указанных вероятностей
#
#
# def _shuffle_dataset(dataset):
#     indexes = np.arange(len(dataset))
#     indexes = np.random.permutation(indexes)
#     dataset.data, dataset.labels = dataset.data[indexes], dataset.labels[indexes]
#     return dataset
#
#
# @REGISTRY.register_module
# class DataLoader(object):
#     def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False,
#                  sample_type: SampleType = SampleType.UPSAMPLE, epoch_size=None, probabilities=None,
#                  overfit: bool = False, drop_last: bool = False):
#         self.dataset = dataset if not shuffle else _shuffle_dataset(dataset)
#         self.batch_size = batch_size
#         self.epoch_size = epoch_size if epoch_size else len(self.dataset) // self.batch_size
#         self._current_batch = None
#         self._current_idx = None
#         self.overfit = overfit
#         self.sample_type = sample_type
#         self.probabilities = probabilities
#         self.drop_last = drop_last
#         self.shuffle = shuffle
#
#         # if sample_type != SampleType.DEFAULT and shuffle:
#         #     raise ValueError('sampler option is mutually exclusive with shuffle')
#         # elif sample_type != SampleType.DEFAULT:
#         #     self._sample(sample_type, probabilities)
#
#     def _sample(self, dataset, probabilities=None):
#         dataset = _shuffle_dataset(dataset)
#         count = np.bincount(dataset.labels)
#         threshold, repeats = np.ones_like(dataset.classes, dtype=int), np.ones_like(dataset.classes,
#                                                                                          dtype=int)
#         new_data = []
#         new_labels = []
#
#         if self.sample_type == SampleType.UPSAMPLE:
#             threshold *= np.max(count)
#         elif self.sample_type == SampleType.DOWNSAMPLE:
#             threshold *= np.min(count)
#         elif self.sample_type == SampleType.PROBABILITIES and probabilities is not None:
#             if len(probabilities) != len(dataset.classes):
#                 raise ValueError('probabilities len not equal number of classes')
#             threshold = np.ceil(probabilities * count).astype(np.int32)
#         else:
#             raise ValueError('SampleType.PROBABILITIES is chosen, but probabilities is None')
#
#         repeats = np.ceil(threshold / count).astype(np.int32)
#         for i, _ in enumerate(dataset.classes):
#             class_data = dataset.data[dataset.labels == i]
#             class_data = np.repeat(class_data, repeats[i], 0)[:threshold[i]]
#             new_data.append(class_data)
#             new_labels.append(np.ones(threshold[i]) * i)
#
#         dataset.data = np.concatenate(new_data)
#         dataset.labels = np.concatenate(new_labels).astype(np.int32)
#         dataset = _shuffle_dataset(dataset)
#         return dataset
#
#     def set_overfit_mode(self):
#         self.overfit = True
#
#     def batch_generator(self):
#         permanent_dataset = self.dataset
#
#         if self.sample_type != SampleType.DEFAULT and self.shuffle:
#             raise ValueError('sampler option is mutually exclusive with shuffle')
#         elif self.sample_type != SampleType.DEFAULT:
#             permanent_dataset = self._sample(permanent_dataset, self.probabilities)
#         self.epoch_size = len(permanent_dataset) // self.batch_size
#         for i in range(self.epoch_size):
#             # todo: overfit
#             idx = i if not self.overfit else 0
#
#
#
#             # if len(permanent_dataset) > self.batch_size:
#             #     self._current_batch = random.sample(permanent_dataset[0, len(permanent_dataset)], self.batch_size)
#             #     permanent_dataset = [data for data in permanent_dataset if data not in self._current_batch]
#             #     yield self._current_batch
#             # elif not self.drop_last:
#             #     yield permanent_dataset
#                 # self._current_batch = np.concatenate(permanent_dataset, random.sample(self.dataset, self.batch_size - len(permanent_dataset)))
#
#
#                 # self._current_batch = random.sample(self.dataset, self.batch_size)
#
#             self._current_idx = idx * self.batch_size
#             end = (idx + 1) * self.batch_size
#             self._current_batch = permanent_dataset[self._current_idx:end]
#             yield self._current_batch
#
#     def show_batch(self, first_batch: bool = False, counter: int = 0):
#         x = int(math.sqrt(len(self._current_batch[0])))
#         fig_size = (x, math.ceil(len(self._current_batch[0]) / x))
#
#         fig = plt.figure(figsize=(16, 9))
#         for i, img in enumerate(self._current_batch[0]):
#             fig.add_subplot(*fig_size, i + 1)
#             plt.imshow(img)
#
#         if first_batch:
#             path = os.path.join(ROOT_DIR, 'batch_figures',
#                                 type(self.dataset).__name__,
#                                 'first_batch')
#             figname = str(counter) + '_' + ''.join(type(e).__name__ + '_' for e in self.dataset.transform.transforms) \
#                 if self.dataset.transform else 'None'
#         else:
#             path = os.path.join(ROOT_DIR, 'batch_figures',
#                                 type(self.dataset).__name__,
#                                 'train' if self.dataset.is_train else 'test',
#                                 ''.join(type(e).__name__ + '_' for e in self.dataset.transform.transforms)
#                                 if self.dataset.transform else 'None')
#             figname = f'{self._current_idx}_batch.png'
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         with open(os.path.join(path, 'transforms.txt'), "w+") as file:
#             file.write(''.join(str(e) + '\n' for e in self.dataset.transform.transforms)
#                        if self.dataset.transform else 'None')
#         file.close()
#
#         plt.savefig(os.path.join(path, figname))
#         plt.close(fig)
