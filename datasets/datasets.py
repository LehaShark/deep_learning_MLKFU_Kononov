import pickle
from urllib import request
import os
from configs.mnist_config import base_url

from datasets.base import Dataset, DataPath, un_gz
import numpy as np
from transforms.registry import Registry

REGISTRY = Registry('datasets')


@REGISTRY.register_module
class MNIST(Dataset):
    def __init__(self, path, is_train, transform, target_transform):
        self.path = path
        self.data_name = f"{'train' if is_train else 't10k'}-images-idx3-ubyte.gz"
        self.label_name = f"{'train' if is_train else 't10k'}-labels-idx1-ubyte.gz"
        self.data_path = DataPath(
            data_path=os.path.join(self.path, self.data_name[:-3]),
            label_path=os.path.join(self.path, self.label_name[:-3]))

        super().__init__(is_train, transform, target_transform, np.arange(10))

    def _read_data(self):
        for key, path in self.data_path.items():
            if not os.path.exists(path):
                self._download_dataset()
                break

        data = []
        for key, path in self.data_path.items():
            with open(path, "rb") as file:
                file_data_count = int.from_bytes(file.read(4), 'big')
                file_data_count = int.from_bytes(file.read(4), 'big')
                shape = (file_data_count, int.from_bytes(file.read(4), 'big'), int.from_bytes(file.read(4), 'big')) \
                    if key == 'input' else file_data_count
                file_data = file.read()
                data.append(np.frombuffer(file_data, dtype=np.uint8).reshape(shape))
        file.close()
        return data[0].astype(np.float32), data[1].astype(np.int32)

    def _download_dataset(self):
        # dpath = self.data_path.items()[1]
        os.makedirs(self.path, exist_ok=True)
        for key, path in self.data_path.items():
            if not os.path.exists(os.path.join(self.path, self.data_name)) or os.path.exists(os.path.join(self.path, self.label_name)):
                print("Downloading " + key + "...")
                request.urlretrieve(base_url + self.data_name, self.path + self.data_name)
                request.urlretrieve(base_url + self.label_name, self.path + self.label_name)
                print("Download complete.")
        self._save_mnist()

    def _save_mnist(self):

        un_gz(self.path + self.data_name)
        un_gz(self.path + self.label_name)
        print("Save complete.")


@REGISTRY.register_module
class CIFAR10(Dataset):
    def __init__(self, path, is_train, transform, target_transform):
        with open(os.path.join(path, 'batches.meta'), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            classes = data['label_names']
        file.close()
        self.path = path
        self._train_list = ('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5')
        self._test_list = ('test_batch',)

        super().__init__(is_train, transform, target_transform, classes)

    def _read_data(self) -> tuple:
        data = []
        labels = []

        for file_name in (self._train_list if self.is_train else self._test_list):
            with open(os.path.join(self.path, file_name), 'rb') as file:
                entry = pickle.load(file, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    labels.extend(entry['labels'])
                else:
                    labels.extend(entry['fine_labels'])
        file.close()

        return np.squeeze(data).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(np.float32), np.array(labels)
