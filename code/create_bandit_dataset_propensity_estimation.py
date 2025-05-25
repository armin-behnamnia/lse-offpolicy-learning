import numpy as np
import pickle
from tqdm import tqdm

avg_num_zeros = 0.0
from model_h0 import ModelCifar
from functools import partial
from hyper_params import load_hyper_params
import argparse
import torch
import os
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torch.nn.functional as F
from utils import dataset_mapper, image2flatten
from data import load_data_fast
from utils import file_write
from noise import NoiseGenerator
import random

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
    import pickle

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


# def get_uniform_probs(label):
#     probs = np.ones(10)

#     probs = probs / probs.sum()

#     return probs


# def get_model_probs(model, image):
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
#     return probs


def get_model_probs_from_cfm(label, cfm_probs):
    return cfm_probs[label]

def get_model_probs_we(image, model, eps, device, tau=1.0):
    with torch.no_grad():
        # if dataset == "fmnist":
        #     image = torch.tensor(
        #         image.reshape(-1, 3, 28, 28).repeat(1, 3, 1, 1)
        #     ).float()
        # elif dataset == "cifar":
        #     image = torch.tensor(image.reshape(-1, 3, 32, 32)).float()
        # else:
        #     raise ValueError(f"Dataset {dataset} not valid.")
        probs = (
            torch.softmax(model(image.to(device)) * tau, dim=-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    probs[probs < eps] = eps
    probs /= np.sum(probs, axis=-1, keepdims=True) #why when we have softmax before it
    return probs


def calculate_biased_probs(probs):
    for i in range(len(probs)):
        maximum_index = np.argmax(probs[i])
        swap_candidates = [ind for ind in range(len(probs[i])) if ind != maximum_index]
        np.random.shuffle(swap_candidates)
        swap_index = swap_candidates[0]
        
        temp = probs[i][maximum_index]
        probs[i][maximum_index] = probs[i][swap_index]
        probs[i][swap_index] = temp
    return probs


def initial_cfm_logging_policy(path, num_classes):
    cfm_probs = np.zeros((num_classes, num_classes))
    with open(path, "r") as cfm_file:
        line = cfm_file.readline()
        cur_class_ind = 0
        while line:
            correct_pred_prob = float(line.rstrip())
            cfm_probs[cur_class_ind] = np.ones((1, num_classes)) * ((1 - correct_pred_prob) / (num_classes - 1))
            cfm_probs[cur_class_ind][cur_class_ind] = correct_pred_prob
            line = cfm_file.readline()
            cur_class_ind += 1
    return cfm_probs



parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", required=True, help="Path to experiment config file."
)
parser.add_argument("-d", "--device", required=True, type=str, help="Device")
parser.add_argument(
    "-l",
    "--linear",
    action="store_true",
    help="If used, the logging policy is a linear model",
)
parser.add_argument(
    "--tau",
    type=float,
    required=True,
    help="Softmax inverse temperature for training the logging policy.",
)
parser.add_argument(
    "--ul",
    type=float,
    required=True,
    help="The ratio of missing-reward to known-reward samples.",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset to generate the bandit dataset from. Can be either fmnist or cifar.",
)
parser.add_argument(
    "--raw_image",
    action="store_true",
    help="If used, raw flatten image is given to the model instead of pretrained features.",
)
parser.add_argument(
    "--feature_size",
    required=False,
    type=int,
    help="If used, given feature size is supposed for the context.",
)
parser.add_argument(
    "--add_unlabeled",
    required=False,
    type=int,
)

parser.add_argument("--train_ratio", type=float, help="What ratio of train dataset will be used for training", required=False)
parser.add_argument("--uniform_noise_alpha", type=float, help="Do we have uniform noise on probs?", required=False)
parser.add_argument("--gaussian_noise_alpha", type=float, help="Do we have gaussian noise on probs?", required=False)
parser.add_argument("--gamma_noise_beta", type=float, help="If used, we will have gamma noise on probs", required=False)
parser.add_argument("--unbalance", type=float, nargs=2, help="If used, we will have unbalance dataset", required=False)
parser.add_argument("--data_repeat", type=int, help="If used, we will repeat our data records", required=False)
parser.add_argument("--reward_flip", type=float, help="If used, we will have binary reward flip in our data records", required=False)
parser.add_argument(
    "--biased_log_policy",
    action="store_true",
    default=None,
    help="If used, biased logging policy will be used.",
)
parser.add_argument("--logging_policy_cm", type=str, help="Logging policy confusion matrix path", required=False)
parser.add_argument("--gaussian_imbalance", type=float, help="We use Gaussian distribution to imbalance our dataset", required=False)

args = parser.parse_args()

linear = args.linear
device = args.device
hyper_params = load_hyper_params(args.config)
hyper_params["tau"] = args.tau
hyper_params["raw_image"] = args.raw_image
hyper_params["train_ratio"] = args.train_ratio
hyper_params["uniform_noise_alpha"] = args.uniform_noise_alpha
hyper_params["gaussian_noise_alpha"] = args.gaussian_noise_alpha
hyper_params["gamma_noise_beta"] = args.gamma_noise_beta
hyper_params["biased_log_policy"] = args.biased_log_policy
hyper_params["unbalance"] = args.unbalance # number of bad classes, unbalance prob
hyper_params["data_repeat"] = args.data_repeat
hyper_params["logging_policy_cm"] = args.logging_policy_cm
hyper_params["gaussian_imbalance"] = args.gaussian_imbalance
hyper_params["propensity_estimation"] = True

noise_generator = NoiseGenerator(hyper_params=hyper_params)

ul_ratio = None
labeled_proportion = 1.0
tau = args.tau
eps = 0
dataset = args.dataset
ul_ratio = args.ul
full_dataset = hyper_params["dataset"]
hyper_params["dataset_name_string"] = dataset
hyper_params["dataset"] = dataset_mapper[dataset]
hyper_params["dataset"]["name"] = full_dataset
dataset_info = {}
feature_size = args.feature_size
# dataset = 'biased'
if feature_size is not None:
    hyper_params["feature_size"] = feature_size
else:
    hyper_params["feature_size"] = np.prod(hyper_params["dataset"]["data_shape"])

if ul_ratio is not None:
    labeled_proportion = 1 / (ul_ratio + 1)
print("Hyperparams: ", hyper_params)
print("Labeled proportion =", labeled_proportion)

if ul_ratio is None:
    exit()
if eps > 0:
    exit()

print("Is Linear?", linear)

if args.biased_log_policy:
    print("Logging policy is biased...")

if args.reward_flip:
    print("We will have reward flip...")


model = nn.Linear(
    hyper_params["feature_size"], hyper_params["dataset"]["num_classes"]
)
model_path = f"models/{dataset}/propensity_estimator_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_1.0_tau{args.tau}.pth"

print(model_path)

print("Model Path: ", model_path)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
probs_fn = partial(get_model_probs_we, model=model, eps=eps, device=device, tau=tau)
stored_feature = None

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []

train_reader, test_reader, val_reader = load_data_fast(
    hyper_params, device=device, labeled=False, create_dataset=True
)
loaders = {"train": train_reader, "val": val_reader, "test": test_reader}
data = dict()

propensity_mse = {'train': 0.0, 'val': 0.0, 'test':0.0}

# Start creating bandit-dataset
for num_sample in [hyper_params["num_sample"]]:  # [1, 2, 3, 4, 5]:
    print("Pre-processing for num sample = " + str(num_sample))

    final_x, final_y, final_actions, final_prop, final_labeled, final_flip = [], [], [], [], [], []

    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0
    neg_cost_count = 0
    data = {}

    for mode in ["train", "val"]:
        print(f"####### MODE {mode}:")
        final_x, final_y, final_actions, final_prop, final_labeled, final_flip = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        avg_num_zeros = 0.0
        expected_reward = 0.0
        total = 0.0
        neg_cost_count = 0
        for epoch in range(num_sample):
            s = 0
            N = len(loaders[mode].dataset)
            n = int(N * labeled_proportion)
            print(N, n)
            for x, y, action, delta, prop, _ in loaders[mode]:
                x, y, action, delta, prop = (
                    x.to(device),
                    y.to(device),
                    action.to(device),
                    delta.to(device),
                    prop.to(device),
                )
                image = x
                label = y.cpu().numpy()
                prev_props = prop.cpu().numpy()
                cor_actions = action.cpu().numpy()

                probs = probs_fn(image)
                u = probs.astype(np.float64)
                # print("################## X SHAPE:", u.shape)

                actions = []
                labeled = []
                for i in range(len(u)):
                    actions.append([cor_actions[i]])
                    propensity_mse[mode] += ((u[i][cor_actions[i]] - prev_props[i]) ** 2)
                    if s < n:
                        labeled.append([1.0])
                    else:
                        if label[i] == actions[i]:
                            neg_cost_count += 1
                        if mode == 'train' and args.add_unlabeled is not None and s < args.add_unlabeled + n:
                            labeled.append([2.0])
                        else:
                            labeled.append([0.0])
                    s += 1
                labeled = np.array(labeled)
                actions = np.array(actions)

                u = image2flatten(
                    image.cpu().numpy(),
                    hyper_params["dataset"],
                    hyper_params["raw_image"],
                )

                final_x.append(u)
                final_actions.append(actions)
                final_prop.append(noise_generator.apply_noise(probs))
                final_labeled.append(labeled)
                # print(final_prop[0].shape)

                final_y.append(
                    F.one_hot(y, num_classes=hyper_params["dataset"]["num_classes"])
                    .cpu()
                    .numpy()
                )
                
                expected_reward += (actions[:, 0] == label).sum()
                total += len(label)

        avg_num_zeros /= float(N)
        avg_num_zeros = round(avg_num_zeros, 4)
        print(
            "Num sample = "
            + str(num_sample)
            + "; Acc = "
            + str(100.0 * expected_reward / total)
        )
        dataset_info[mode] = str(100.0 * expected_reward / total) #storing logging policy accuracy
        
        print(
            "Neg reward proportion = "
            + str(
                neg_cost_count / total / (1 - labeled_proportion)
                if labeled_proportion < 1.0
                else 0
            )
        )
        print()

        # Save as CSV
        # if labeled_proportion < 1.0:
        final_x = np.concatenate(final_x, axis=0)
        final_y = np.concatenate(final_y, axis=0)
        final_prop = np.concatenate(final_prop, axis=0)
        final_actions = np.concatenate(final_actions, axis=0)
        final_labeled = np.concatenate(final_labeled, axis=0)
        print(
            final_x.shape,
            final_y.shape,
            final_prop.shape,
            final_actions.shape,
            final_labeled.shape,
        )
        
        if args.reward_flip is not None:
            final_flip = np.concatenate(final_flip, axis=0)
            print("Final Flip shape:", final_flip.shape)
            data[mode] = np.concatenate(
                (final_x, final_y, final_prop, final_actions, final_labeled, final_flip), axis=1
            )
        else:
            data[mode] = np.concatenate(
                (final_x, final_y, final_prop, final_actions, final_labeled), axis=1
            )

        print(f"################ Number of labeled = {(final_labeled == 1).sum()}")

    for mode in ["test"]:
        print(f"####### MODE {mode}:")
        final_x, final_y, final_actions, final_prop = (
            [],
            [],
            [],
            [],
        )
        avg_num_zeros = 0.0
        expected_reward = 0.0
        total = 0.0
        neg_cost_count = 0
        for epoch in range(num_sample):
            s = 0
            N = len(loaders[mode].dataset)
            n = int(N * labeled_proportion)
            for x, y, action, delta, prop in loaders[mode]:
                x, y, action, delta, prop = (
                    x.to(device),
                    y.to(device),
                    action.to(device),
                    delta.to(device),
                    prop.to(device),
                )
                image = x
                label = y.cpu().numpy()
                prev_props = prop.cpu().numpy()
                cor_actions = action.cpu().numpy()

                probs = probs_fn(image)
                u = probs.astype(np.float64)

                actions = []
                for i in range(len(u)):
                    actions.append([cor_actions[i]])
                    propensity_mse[mode] += ((u[i][cor_actions[i]] - prev_props[i]) ** 2)
                    s += 1
                actions = np.array(actions)

                u = image2flatten(
                    image.cpu().numpy(),
                    hyper_params["dataset"],
                    hyper_params["raw_image"],
                )
                # print("################## X SHAPE:", u.shape)
                final_x.append(u)
                final_actions.append(actions)
                final_prop.append(noise_generator.apply_noise(probs))
                # print(final_prop[0].shape)

                final_y.append(
                    F.one_hot(y, num_classes=hyper_params["dataset"]["num_classes"])
                    .cpu()
                    .numpy()
                )
                # print(actions, label)
                expected_reward += (actions[:, 0] == label).sum()
                total += len(label)

        avg_num_zeros /= float(N)
        avg_num_zeros = round(avg_num_zeros, 4)
        print(
            "Num sample = "
            + str(num_sample)
            + "; Acc = "
            + str(100.0 * expected_reward / total)
        )
        dataset_info[mode] = str(100.0 * expected_reward / total)
        print(
            "Neg reward proportion = "
            + str(
                neg_cost_count / total / (1 - labeled_proportion)
                if labeled_proportion < 1.0
                else 0
            )
        )
        print()
        final_x = np.concatenate(final_x, axis=0)
        final_y = np.concatenate(final_y, axis=0)
        final_prop = np.concatenate(final_prop, axis=0)
        final_actions = np.concatenate(final_actions, axis=0)

        # Save as CSV
        # if labeled_proportion < 1.0:
        data[mode] = np.concatenate(
            (final_x, final_y, final_prop, final_actions), axis=1
        )
    

    store_folder = dataset
    if args.add_unlabeled is not None:
        store_folder += f"_u{args.add_unlabeled}"

    if hyper_params["raw_image"]:
        store_folder += "_raw"
    if linear:
        store_folder += "_linear/"
    else:
        store_folder += "/"

    store_folder += f"{tau}"
    store_folder += f"_{int(ul_ratio)}"

    filename = f"../data/{store_folder}/bandit_data_"
    filename += "sampled_" + str(num_sample) + "_"
    print(f"Saving dataset in {filename}...")

    for mode in ["train", "val", "test"]:
        file_name_end = "_propensity_estimation"
        save_obj(data[mode], filename + mode + file_name_end)
        
        end = "_propensity_estimation"
        mse_error = propensity_mse[mode] / len(data[mode])
        final_info = f"Logging Policy Acc = {dataset_info[mode]}, Propensity_MSE={mse_error}"
        file_write(f"../data/{store_folder}/{mode}_info" + end, final_info)
