import gzip
import os
import numpy as np

def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    os.remove(file_name)

class DataPath(dict):
    def __init__(self, data_path: str, label_path: str):
        super(DataPath, self).__init__(input=data_path, target=label_path)


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

        input_, target = self.data[index], self.labels[index]

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_, target

    def one_hot_labels(self, label):
        if hasattr(label, '__len__'):
            one_hot = np.zeros((len(label), len(self.classes)), dtype=np.int32)
            label = np.arange(len(label)), np.array(label).reshape(-1)
        else:
            one_hot = np.zeros(len(self.classes), dtype=np.int32)

        one_hot[label] = 1
        return one_hot

    def show_statistics(self, idx):

        labels = self.labels[idx] if idx is not None else self.labels

        count = np.bincount(labels)

        print('\t---DATASET STAT---')
        for i, cls in enumerate(self.classes):
            print(f"{i + 1} class: '{cls}':\t {count[i]}")

        print(f"overall:\t\t {len(labels)}")
