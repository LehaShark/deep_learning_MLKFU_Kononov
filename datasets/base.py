from urllib import request
import gzip
import numpy as np
import os
from configs.mnist_config import base_url

def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    os.remove(file_name)


class DataPath(dict):
    def __init__(self, data_path: str, label_path: str):
        super(DataPath, self).__init__(input=data_path, label=label_path)


class Dataset(object):
    def __init__(self, is_train: bool, transform, target_transform, classes):
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.data, self.labels = self._read_data()

    def _read_data(self) -> tuple:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        input, target = self.data[index], self.labels[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def one_hot_labels(self, label):
        if hasattr(label, '__len__'):
            one_hot = np.zeros((len(label), len(self.classes)), dtype=np.int32)
            label = np.arange(len(label)), np.array(label).reshape(-1)
        else:
            one_hot = np.zeros(len(self.classes), dtype=np.int32)

        one_hot[label] = 1
        return one_hot

    def show_statistics(self):
        count = np.bincount(self.labels)

        print('\t---DATASET STAT---')
        for i, cls in enumerate(self.classes):
            print(f"{i + 1} class: '{cls}':\t {count[i]}")

        print(f"overall:\t\t {len(self.labels)}")


# class MNIST(Dataset):
    # def __init__(self, path, is_train, transform, target_transform, classes):
    #     self.path = path
    #     self.data_name = f"{'train' if is_train else 't10k'}-images-idx3-ubyte.gz"
    #     self.label_name = f"{'train' if is_train else 't10k'}-labels-idx1-ubyte.gz"
    #     self.data_path = DataPath(
    #         data_path=os.path.join(self.path, self.data_name[:-3]),
    #         label_path=os.path.join(self.path, self.label_name[:-3]))
    #
    #     super().__init__(is_train, transform, target_transform, classes)
    #
    # def _read_data(self):
    #     for key, path in self.data_path.items():
    #         if not os.path.exists(path):
    #             self._download_dataset()
    #             break
    #
    #     data = []
    #     for key, path in self.data_path.items():
    #         with open(path, "rb") as file:
    #             magic_number = int.from_bytes(file.read(4), 'big')
    #             file_data_count = int.from_bytes(file.read(4), 'big')
    #             shape = (file_data_count, int.from_bytes(file.read(4), 'big'), int.from_bytes(file.read(4), 'big')) \
    #                 if key == 'input' else file_data_count
    #             file_data = file.read()
    #             data.append(np.frombuffer(file_data, dtype=np.uint8).reshape(shape))
    #     file.close()
    #     return data[0].astype(np.float32), data[1].astype(np.int32)
    #
    # def _download_dataset(self):
    #     # dpath = self.data_path.items()[1]
    #     os.makedirs(self.path, exist_ok=True)
    #     for key, path in self.data_path.items():
    #         if not os.path.exists(os.path.join(self.path, self.data_name)) or os.path.exists(os.path.join(self.path, self.label_name)):
    #             print("Downloading " + key + "...")
    #             request.urlretrieve(base_url + self.data_name, self.path + self.data_name)
    #             request.urlretrieve(base_url + self.label_name, self.path + self.label_name)
    #             print("Download complete.")
    #     # for name in self.mnist_config.filename:
    #     #     if not os.path.exists(os.path.join(self.data_path, name[0])):
    #     #         print("Downloading " + name[1] + "...")
    #     #         request.urlretrieve(self.mnist_config.base_url + name[1], self.data_path + name[1])
    #     #         print("Download complete.")
    #     self._save_mnist()
    #
    # def _save_mnist(self):
    #
    #     un_gz(self.path + self.data_name)
    #     un_gz(self.path + self.label_name)
    #     # mnist = {}
    #     # # for key, path in self.data_path.items():
    #     # filename = [self.data_name, self.label_name]
    #     # for name in filename:
    #     #     with gzip.open(self.path + name, 'rb') as f:
    #     #         mnist[name] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    #     # # for name in filename[-2:]:
    #     # #     with gzip.open(self.data_path + name[1], 'rb') as f:
    #     # #         mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
    #     # with open(os.path.join(self.path, FILENAME), 'wb') as f:
    #     #     pickle.dump(mnist, f)
    #     print("Save complete.")