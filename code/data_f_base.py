import torch
import numpy as np
from tqdm import tqdm

from utils import *


class DataLoader:
    def __init__(self, hyper_params, x, delta, prop=None, action=None, labeled=None):
        self.x = x
        self.delta = delta
        self.prop = prop
        self.action = action
        self.labeled = labeled
        self.bsz = hyper_params["batch_size"]
        self.hyper_params = hyper_params

    def __len__(self):
        return len(self.x)

    def iter(self, eval=False):
        x_batch, y_batch, action, delta, all_delta, prop, all_prop, labeled = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        data_done = 0
        for ind in tqdm(range(len(self.x))):
            x_batch.append(np.repeat(self.x[ind].reshape(1, 28, 28), repeats=3, axis=0))
            y_batch.append(np.argmax(self.delta[ind]))

            # Pick already chosen action
            choice = self.action[ind]
            action.append(choice)

            delta.append(self.delta[ind][choice])

            if self.prop[ind][choice] < 0.001:
                prop.append(0.001)
            else:  # Overflow issues, Sanity check
                prop.append(self.prop[ind][choice])

            all_delta.append(self.delta[ind])
            all_prop.append(self.prop[ind])
            if self.labeled is not None:
                labeled.append(self.labeled[ind])
            data_done += 1

            if len(x_batch) == self.bsz:
                if eval == False:
                    if self.labeled is None:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        )
                    else:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), torch.tensor(
                            labeled
                        ).float()
                else:
                    if self.labeled in None:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), all_prop, all_delta
                    else:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), all_prop, all_delta, torch.tensor(
                            labeled
                        ).float()

                x_batch, y_batch, action, delta, all_delta, prop, all_prop, labeled = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )


def readfile(path, hyper_params):
    if "feature_size" in hyper_params:
        feature_size = hyper_params["feature_size"]
    else:
        feature_size = 784
    x, delta, prop, action = [], [], [], []

    data = np.load(path + ".npy")

    for line in data:
        x.append(line[:feature_size])
        delta.append(line[feature_size : (feature_size + 10)])
        prop.append(line[(feature_size + 10) : (feature_size + 20)])
        action.append(int(line[-1]))

    return np.array(x), np.array(delta), np.array(prop), np.array(action)


def readfile_unlabeled(path, hyper_params):
    if "feature_size" in hyper_params:
        feature_size = hyper_params["feature_size"]
    else:
        feature_size = 784
    x, delta, prop, action, labeled = [], [], [], [], []

    data = np.load(path + ".npy")
    for line in data:
        x.append(line[:feature_size])
        delta.append(line[feature_size : (feature_size + 10)])
        prop.append(line[(feature_size + 10) : (feature_size + 20)])
        action.append(int(line[-2]))
        labeled.append(int(line[-1]))

    return (
        np.array(x),
        np.array(delta),
        np.array(prop),
        np.array(action),
        np.array(labeled),
    )


def load_data(hyper_params, labeled=True):
    store_folder = hyper_params.dataset
    path = f"../data/{store_folder}/"

    labeled_train = None
    labeled_val = None
    if labeled:
        x_train, delta_train, prop_train, action_train = readfile(
            path + "train", hyper_params
        )
        x_val, delta_val, prop_val, action_val = readfile(path + "val", hyper_params)
    else:
        (
            x_train,
            delta_train,
            prop_train,
            action_train,
            labeled_train,
        ) = readfile_unlabeled(path + "train", hyper_params)
        x_val, delta_val, prop_val, action_val, labeled_val = readfile_unlabeled(
            path + "val", hyper_params
        )
    x_test, delta_test, prop_test, action_test = readfile(path + "test", hyper_params)

    # Shuffle train set
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    delta_train = delta_train[indices]
    prop_train = prop_train[indices]
    action_train = action_train[indices]
    if not labeled:
        labeled_train = labeled_train[indices]

    trainloader = DataLoader(
        hyper_params,
        x_train,
        delta_train,
        prop_train,
        action_train,
        labeled_train if not labeled else None,
    )
    testloader = DataLoader(hyper_params, x_test, delta_test, prop_test, action_test)
    valloader = DataLoader(
        hyper_params, x_val, delta_val, prop_val, action_val, labeled_val
    )

    return trainloader, testloader, valloader
