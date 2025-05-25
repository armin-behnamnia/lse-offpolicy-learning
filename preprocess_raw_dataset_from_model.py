import numpy as np
import pickle
from tqdm import tqdm

avg_num_zeros = 0.0
from code.model_h0 import ModelCifar
from functools import partial
from code.hyper_params import load_hyper_params
import argparse
import torch
import os
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR100, CIFAR10
from code.utils import dataset_mapper

np.random.seed(2023)
torch.manual_seed(2023)


def one_hot(arr, num_classes):
    new = []
    for i in tqdm(range(len(arr))):
        temp = np.zeros(num_classes)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


# def get_probs(label):
#     probs = np.power(np.add(list(range(10)), 1), 0.5)
#     probs[label] = probs[label] * 10

#     probs = probs / probs.sum()

#     return probs


# def get_biased_probs(label):
#     probs = np.power(np.add(list(range(10)), 1), 0.5)
#     probs[label] = probs[label] * 1.5

#     probs = probs / probs.sum()

#     return probs

# def get_wrong_probs(label):
#     probs = np.ones(10)
#     probs[(label + 1) % 10] *= 10

#     probs = probs / probs.sum()

#     return probs


def get_uniform_probs(num_classes):
    probs = np.ones(num_classes)

    probs = probs / probs.sum()

    return probs


def get_model_probs(model, image):
    with torch.no_grad():
        image = torch.tensor(
            np.repeat(image.reshape(1, 1, 28, 28), repeats=3, axis=1)
        ).float()
        probs = (
            torch.softmax(model(image.to(device)), dim=-1)
            .cpu()
            .numpy()
            .squeeze(0)
            .astype(np.float32)
        )
    return probs


# def get_model_probs_we(image, model, eps):
#     with torch.no_grad():
#         image = torch.tensor(
#             np.repeat(image.reshape(1, 1, 28, 28), repeats=3, axis=1)
#         ).float()
#         probs = (
#             torch.softmax(model(image.to(device)), dim=-1)
#             .cpu()
#             .numpy()
#             .squeeze(0)
#             .astype(np.float32)
#         )
#     probs[probs < eps] = eps
#     probs /= np.sum(probs)
#     return probs


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", required=True, help="Path to experiment config file."
)
parser.add_argument("-d", "--device", required=True, help="Device", type=str)
parser.add_argument("--tau", type=float, required=True, default=1.0)
parser.add_argument("--ul", type=float, required=True, default=0)
parser.add_argument("--reward_model", type=str, required=False)
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--train_ratio", type=float, help="What ratio of train dataset will be used for training", required=False)

args = parser.parse_args()
hyper_params = load_hyper_params(args.config, create_dir=False)
ul_ratio = None
proportion = 1.0
labeled_proportion = 1.0
tau = args.tau
eps = 0

ul_ratio = args.ul
# dataset = 'biased'
dataset = None

if ul_ratio is not None:
    labeled_proportion = 1 / (ul_ratio + 1)

print("Labelled proportion =", labeled_proportion)

if ul_ratio is None:
    exit()
if proportion < 1.0:
    exit()
if dataset is not None:
    exit()
if eps > 0:
    exit()

dataset = args.dataset
hyper_params["dataset"] = dataset_mapper[dataset]
store_folder = dataset
if args.reward_model:
    store_folder += "_ps"

store_folder += "/"
# store_folder += f"{tau}"
# store_folder += f"_{int(ul_ratio)}"
store_folder += "base"
print(store_folder)
device = args.device

reward_model = None
if args.reward_model:
    reward_model = ModelCifar(hyper_params)
    reward_model.load_state_dict(torch.load(args.reward_model))
    reward_model.to(device)
    reward_model.eval()
    rewards_fn = partial(get_model_probs, model=reward_model)

model = ModelCifar(hyper_params)
model.to(device)
model.eval()

os.makedirs(f"data/{store_folder}", exist_ok=True)


x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []


train_dataset = dataset_mapper[dataset]["class"](
    root=f"data/dataset/",
    train=True,
    **(dataset_mapper[dataset]["args"]),
    download=True,
)
test_dataset = dataset_mapper[dataset]["class"](
    root=f"data/dataset/",
    train=False,
    **(dataset_mapper[dataset]["args"]),
    download=True,
)
N, M = len(train_dataset), len(test_dataset)
print("Len Train =", N)
print("Len Test =", M)

# Train
for i in range(N):
    image, label = train_dataset[i]
    image = np.array(image)
    if dataset in ["cifar", "cifar100"]:
        image = image.transpose(2, 0, 1)
    image = (image.reshape(np.prod(list(dataset_mapper[dataset]["data_shape"]))))
    x_train.append(image)
    y_train.append(label)
x_train = np.stack(x_train)
y_train = np.stack(y_train)

# Test
for i in range(M):
    image, label = test_dataset[i]
    image = np.array(image)
    if dataset in ["cifar", "cifar100"]:
        image = image.transpose(2, 0, 1)
    image = (image.reshape(np.prod(list(dataset_mapper[dataset]["data_shape"]))))
    x_test.append(image)
    y_test.append(label)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

# Normalize X data
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0

# One hot the rewards
y_train = one_hot(y_train, dataset_mapper[dataset]["num_classes"])
y_test = one_hot(y_test, dataset_mapper[dataset]["num_classes"])

# Shuffle the dataset once
indices = np.arange(len(x_train))
np.random.shuffle(indices)
assert len(x_train) == len(y_train)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)
N = len(x_train)
n = int(N * labeled_proportion)
# Start creating bandit-dataset
print("x_train, x_val shape = ", x_train.shape, x_test.shape)
for num_sample in [1]:  # [1, 2, 3, 4, 5]:
    print("Pre-processing for num sample = " + str(num_sample))

    final_x, final_y, final_actions, final_prop, final_labeled = [], [], [], [], []

    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0
    neg_cost_count = 0

    for epoch in range(num_sample):
        for point_num in tqdm(range(x_train.shape[0])):
            image = x_train[point_num]
            label = np.argmax(y_train[point_num])

            probs = get_uniform_probs(
                num_classes=dataset_mapper[dataset]["num_classes"]
            )
            u = probs.astype(np.float64)
            actionvec = np.random.multinomial(1, u / np.sum(u))
            action = np.argmax(actionvec)

            final_x.append(image)
            final_actions.append([action])
            final_prop.append(probs)
            if labeled_proportion < 1.0 and not reward_model:
                if point_num < n:
                    final_labeled.append(np.array([1.0]))
                else:
                    if label == action:
                        neg_cost_count += 1
                    final_labeled.append(np.array([0.0]))
            else:
                final_labeled.append(np.array([1.0]))

            if reward_model and labeled_proportion < 1.0 and point_num >= n:
                final_y.append(rewards_fn(image))
            else:
                final_y.append(y_train[point_num])

            expected_reward += float(int(action == label))
            total += 1.0
            # Printing the first prob. dist.
            # if point_num == 0: print("Prob Distr. for 0th sample:\n", [ round(i, 3) for i in list(probs) ])

    avg_num_zeros /= float(x_train.shape[0])
    avg_num_zeros = round(avg_num_zeros, 4)
    print(
        "Num sample = "
        + str(num_sample)
        + "; Acc = "
        + str(100.0 * expected_reward / total)
    )
    print("Neg reward proportion = " + str(neg_cost_count / total))
    print()

    # Save as CSV
    # if labeled_proportion < 1.0:
    final_normal = np.concatenate(
        (final_x, final_y, final_prop, final_actions, final_labeled), axis=1
    )
    print("final normal = ", final_normal.shape)
    # else:
    #     final_normal = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)

    N = len(final_normal)
    idx = list(range(N))
    idx = np.random.permutation(idx)
    print("Meta train size = ", dataset_mapper[dataset]["sizes"]["train"])
    print("Meta val size = ", dataset_mapper[dataset]["sizes"]["val"])
    
    train_data_count = dataset_mapper[dataset]["sizes"]["train"]
    val_data_count = dataset_mapper[dataset]["sizes"]["val"]
    if args.train_ratio is not None:
        train_data_count = int(train_data_count * args.train_ratio)
        val_data_count = int(val_data_count * args.train_ratio)
    train = final_normal[idx[: train_data_count]]
    val = final_normal[idx[dataset_mapper[dataset]["sizes"]["train"] : dataset_mapper[dataset]["sizes"]["train"] + val_data_count]]
    print("train, val shape = ", train.shape, val.shape, "Train Count =", train_data_count, "Val Count", val_data_count)
    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0
    
    test_prop, test_actions = [], []
    xs = []
    for i, label in tqdm(enumerate(y_test)):
        label = np.argmax(label)
        image = x_test[i]
        probs = get_uniform_probs(num_classes=dataset_mapper[dataset]["num_classes"])
        xs.append(image)
        u = probs.astype(np.float64)
        test_prop.append(probs)
        actionvec = np.random.multinomial(1, u / np.sum(u))
        action = np.argmax(actionvec)
        test_actions.append([action])

        expected_reward += float(int(action == label))
        total += 1.0

    print("Acc = " + str(100.0 * expected_reward / total))
    print()

    test = np.concatenate((xs, y_test, test_prop, test_actions), axis=1)  # Garbage
    filename = f"data/{store_folder}/bandit_data_"
    filename += "sampled_" + str(num_sample)
    print("file name = ", filename)
    
    if args.train_ratio is not None:
        save_obj(train, filename + "_train_" + str(args.train_ratio))
        save_obj(test, filename + "_test_" + str(args.train_ratio))
        save_obj(val, filename + "_val_" + str(args.train_ratio))
    else:
        save_obj(train, filename + "_train")
        save_obj(test, filename + "_test")
        save_obj(val, filename + "_val")        
