# from torch.utils.data import Sampler, BatchSampler
from torch.utils.data import DataLoader
from typing import Union, Iterable, Generic, TypeVar, List, Sized, Sequence

import numpy as np

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler[int]):

    def __init__(self, data_source: Union[Sized, int]) -> None:
        self.data_len = len(data_source) if hasattr(data_source, '__len__') else data_source

    def __iter__(self):
        return iter(range(self.data_len))

    def __len__(self) -> int:
        return self.data_len


class WeightedRandomSampler(Sampler[int]):

    def __init__(self, weights: Sequence[float], num_samples: int):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        self.weights = np.asaray(weights)
        self.num_samples = num_samples

    def __iter__(self):
        rand_tensor = np.random.multinomial(self.num_samples, self.weights)
        yield from iter(rand_tensor)

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
