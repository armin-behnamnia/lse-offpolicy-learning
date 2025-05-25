import torch
import json
import pickle
import numpy as np
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR100, CIFAR10

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()


def image2flatten(imgs, dataset_info, linear): #linear should be renamed to is_raw_image
    if linear:
        return imgs
    if dataset_info["data_shape"][0] == 3:
        return imgs.reshape(len(imgs), -1)
    elif dataset_info["data_shape"][0] == 1:
        return imgs[:, 0].reshape(len(imgs), -1)
    else:
        raise ValueError(f"data shape {dataset_info['data_shape']} not valid.")


def flatten2image(imgs, dataset_info):
    if dataset_info["data_shape"][0] == 3:
        return imgs.reshape(-1, *dataset_info["data_shape"])
    elif dataset_info["data_shape"][0] == 1:
        return np.repeat(
            imgs.reshape(-1, *dataset_info["data_shape"]), repeats=3, axis=1
        )
    else:
        raise ValueError(f"data shape {dataset_info['data_shape']} not valid.")


dataset_mapper = {
    "fmnist": {
        "class": FashionMNIST,
        "args": {},
        "num_classes": 10,
        "data_shape": (1, 28, 28),
        "sizes": {"train": 50_000, "test": 10_000, "val": 10_000},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [2, 2, 2, 2],
    },
    "glass": {
        "class": None,
        "args": {},
        "num_classes": 6,
        "data_shape": (9, ),
        "sizes": {"train": 148, "test": 46, "val": 20},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "letter": {
        "class": None,
        "args": {},
        "num_classes": 26,
        "data_shape": (16, ),
        "sizes": {"train": 13989, "test": 4011, "val": 2000},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "optdigits": {
        "class": None,
        "args": {},
        "num_classes": 10,
        "data_shape": (64, ),
        "sizes": {"train": 3928, "test": 1129, "val": 563},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "wiki": {
        "class": None,
        "args": {},
        "num_classes": 30938,
        "data_shape": (101938, ),
        "sizes": {"train": 12025, "test": 6616, "val": 2121},
    },
    "yeast": {
        "class": None,
        "args": {},
        "num_classes": 10,
        "data_shape": (8, ),
        "sizes": {"train": 1035, "test": 303, "val": 146},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "cifar": {
        "class": CIFAR10,
        "args": {},
        "num_classes": 10,
        "data_shape": (3, 32, 32),
        "sizes": {"train": 45_000, "test": 10_000, "val": 5_000},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [2, 2, 2, 2],
    },
    "cifar100": {
        "class": CIFAR100,
        "args": {},
        "num_classes": 100,
        "data_shape": (3, 32, 32),
        "sizes": {"train": 45_000, "test": 10_000, "val": 5_000},
        "log_model_layers": [2, 2, 2, 2],
        "optimal_model_layers": [2, 2, 2, 2],
        "normalize": True,
    },
    "emnist": {
        "class": EMNIST,
        "args": {"split": "mnist"},
        "num_classes": 10,
        "data_shape": (1, 28, 28),
        "sizes": {"train": 50_000, "test": 10_000, "val": 10_000},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "kuairec": {
        "data_shape": (1555, 30, 128, 64),
        "sizes": {"train": 35880, "test": 7055, "val": 7055},
        "rec": True,
    },
    "coat": {
        "data_shape": (14, 33, 64, 32),
        "sizes": {"train": 2900, "test": 870, "val": 290},
        "rec": True,
    },
    "wikic": {
        "class": None,
        "args": {},
        "num_classes": 30938,
        "data_shape": (1000, ),
        "sizes": {"train": 12025, "test": 6616, "val": 2121},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
    "rct": {
        "class": None,
        "args": {},
        "num_classes": 5,
        "data_shape": (300, ),
        "sizes": {"train": 2211861, "test": 29493, "val": 28932},
        "log_model_layers": [1, 1, 1, 1],
        "optimal_model_layers": [1, 1, 1, 1],
    },
}


def create_tensors(
    x,
    delta,
    prop=None,
    action=None,
    labeled=None,
    device="cuda:0",
    hyper_params=None,
    use_nue_value=True,
    flip=None,
    pareto_noise=None,
    discrete_reward_values=None,
    discrete_flip_mask=None,
    reward_estimator_values=None,
):
    assert hyper_params is not None
    x = torch.tensor(x).to(device)
    delta = torch.tensor(delta).to(device)
    prop = torch.tensor(prop).to(device)
    action = torch.tensor(action).to(device)
    if flip is not None:
        flip = torch.tensor(flip).to(device)
    if pareto_noise is not None:
        pareto_noise = torch.tensor(pareto_noise).to(device)
    if discrete_reward_values is not None:
        discrete_reward_values = torch.tensor(discrete_reward_values).to(device)
    if discrete_flip_mask is not None:
        discrete_flip_mask = torch.tensor(discrete_flip_mask).to(device)
    if reward_estimator_values is not None:
        reward_estimator_values = torch.tensor(reward_estimator_values).to(device)

    if labeled is not None:
        labeled = torch.tensor(labeled).to(device)
    dataset = hyper_params["dataset"]
    if len(dataset["data_shape"]) > 1:
        c, h, w = dataset["data_shape"]
        if "raw_image" not in hyper_params or not hyper_params["raw_image"]:
            new_x = x.reshape(-1, c, h, w)
            if c == 1:
                new_x = np.repeat(new_x, repeats=3, axis=1)
        else:
            new_x = x
    else:
        new_x = x

    print(new_x.shape)
    if hyper_params["use_n_hot"] is not None:
        new_y = delta
    else:
        new_y = torch.argmax(delta, dim=-1).long()

    new_delta = delta[torch.arange(len(delta)), action]
    if flip is not None:
        new_delta = (new_delta + flip) % 2
    
    if pareto_noise is not None:
        if hyper_params["reward_lomax_noise"] is not None:
            print("Lomax Noise Done!")
            new_delta = new_delta + new_delta * pareto_noise # just ones
        else:
            #new_delta = new_delta + pareto_noise
            #new_delta = new_delta + (1 - new_delta) * pareto_noise * 1000
            print("New one")
            new_delta = new_delta + pareto_noise * 1000.0 - (new_delta * pareto_noise)

    if discrete_reward_values is not None:
        print("We will have discrete reward values...")
        if discrete_flip_mask is not None:
            print("We will have discrete reward flip...")
            #discrete_flip_value = hyper_params["discrete_flip"][1]
            #print(discrete_flip_value)

            final_delta = new_delta * discrete_reward_values
            #new_delta = final_delta + discrete_flip_mask * (discrete_flip_value - final_delta)
            discrete_reward_len = len(hyper_params["discrete_reward"])
            reward_values = hyper_params["discrete_reward"][discrete_reward_len//2:]
            reward_values.append(0.0)
            reward_values = np.array(reward_values)
            uniform_flipped_reward_values = np.random.choice(a=reward_values, size=final_delta.shape, replace=True)
            final_uniform_flipped_reward_values = torch.from_numpy(uniform_flipped_reward_values) - final_delta
            new_delta = final_delta + discrete_flip_mask * final_uniform_flipped_reward_values
            
        else:
            new_delta = new_delta * discrete_reward_values

    new_prop = prop[torch.arange(len(prop)), action]
    if use_nue_value == True:
        new_prop = torch.maximum(new_prop, torch.tensor(0.001).to(device))

    # all_delta.append(self.delta[ind])
    # all_prop.append(self.prop[ind])
    if labeled is not None:
        new_labeled = labeled
    if labeled is not None:
        if reward_estimator_values is not None:
            return (
                new_x.float(),
                new_y,
                action.long(),
                new_delta.float(),
                new_prop.float(),
                labeled.float(),
                reward_estimator_values.float(),
            )
        else:
            return (
                new_x.float(),
                new_y,
                action.long(),
                new_delta.float(),
                new_prop.float(),
                labeled.float(),
                labeled.float()
            ) # just fill the gap
    else:
        if reward_estimator_values is not None:
            return (
                new_x.float(),
                new_y,
                action.long(),
                new_delta.float(),
                new_prop.float(),
                reward_estimator_values.float(),
            )
        else:
            return (
                new_x.float(),
                new_y,
                action.long(),
                new_delta.float(),
                new_prop.float(),
            )


if is_cuda_available:
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_obj_json(obj, name):
    with open(name + ".json", "w") as f:
        json.dump(obj, f)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def load_obj_json(name):
    with open(name + ".json", "r") as f:
        return json.load(f)


def file_write(log_file, s, dont_print=False):
    if dont_print == False:
        print(s)
    f = open(log_file, "a")
    f.write(s + "\n")
    f.close()


def clear_log_file(log_file):
    f = open(log_file, "w")
    f.write("")
    f.close()


def pretty_print(h):
    print("{")
    for key in h:
        print(" " * 4 + str(key) + ": " + h[key])
    print("}\n")


def plot_len_vs_ndcg(len_to_ndcg_at_100_map):
    lens = list(len_to_ndcg_at_100_map.keys())
    lens.sort()
    X, Y = [], []

    for le in lens:
        X.append(le)
        ans = 0.0
        for i in len_to_ndcg_at_100_map[le]:
            ans += float(i)
        ans = ans / float(len(len_to_ndcg_at_100_map[le]))
        Y.append(ans * 100.0)

    # Smoothening
    Y_mine = []
    prev_5 = []
    for i in Y:
        prev_5.append(i)
        if len(prev_5) > 5:
            del prev_5[0]

        temp = 0.0
        for j in prev_5:
            temp += float(j)
        temp = float(temp) / float(len(prev_5))
        Y_mine.append(temp)

    plt.figure(figsize=(12, 5))
    plt.plot(X, Y_mine, label="SVAE")
    plt.xlabel("Number of items in the fold-out set")
    plt.ylabel("Average NDCG@100")
    plt.title(hyper_params["project_name"])
    if not os.path.isdir("saved_plots/"):
        os.mkdir("saved_plots/")
    plt.savefig("saved_plots/seq_len_vs_ndcg_" + hyper_params["project_name"] + ".pdf")

    leg = plt.legend(loc="best", ncol=2)

    plt.show()
