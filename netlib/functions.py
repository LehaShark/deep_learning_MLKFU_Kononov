import numpy as np

def log_softmax(x):
    return np.log(softmax(x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_x[e_x == 0] = np.random.normal(1e-30, 1e-32, e_x[e_x == 0].shape)
    res = (e_x.T / e_x.sum(axis=-1)).T
    return res


def nll_loss(predictions, targets):
    # idx or one_hot
    if predictions.shape == targets.shape:
        log_preds = targets * predictions
    else:
        indexes = targets[-1] if len(predictions.shape) == 1 \
            else (np.arange(predictions.shape[0]), targets)
        log_preds = predictions[indexes]

    return -sum(log_preds) / len(predictions)


def cross_entropy(predictions, targets):
    return nll_loss(np.log(softmax(predictions)), targets)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)
