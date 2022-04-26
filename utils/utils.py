import time

import numpy as np
from torch import nn


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


@timer
def get_indexes():
    tensor = np.arange(64*2*14*14).reshape((64, 2, 14, 14))
    # tensor = np.arange(1*2*3*3).reshape((1, 2, 3, 3))

    kernel_size = 3
    stride = 1
    _, d, h, w = tensor.shape

    i_kernels = (w - kernel_size)//stride + 1
    j_kernels = (h - kernel_size)//stride + 1

    k_base = np.arange(d)
    i_base = np.tile(np.arange(kernel_size), i_kernels).reshape(-1, kernel_size)
    j_base = np.tile(np.arange(kernel_size), j_kernels).reshape(-1, kernel_size)

    i_base += stride * np.arange(i_kernels).reshape(-1, 1)
    j_base += stride * np.arange(j_kernels).reshape(-1, 1)

    k = k_base.repeat(kernel_size**2)
    i = np.tile(np.repeat(i_base, kernel_size, axis=1), d) #.reshape(-1, d * kernel_size**2)
    j = np.tile(np.repeat(j_base, kernel_size, axis=0), d)


    k = np.tile(k, i_kernels*j_kernels)
    i = np.repeat(i, i_kernels, axis=0).reshape(-1)
    j = np.tile(j.reshape(-1), j_kernels)

    res = tensor[..., k, i, j].reshape(-1, d*kernel_size**2).T
    # output = np.dot(resh)

    return i, j


if __name__ == "__main__":
    get_indexes()
