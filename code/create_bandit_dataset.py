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
from train_reward_estimator import get_reward_estimator_model_name

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

def get_reward_estimator_model_result(reward_model, x, device):
    with torch.no_grad():
        result = (
            reward_model(x.to(device)).cpu()
        )
    return F.sigmoid(result).numpy().astype(np.float32)

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


def data_ratio_selector(data, ratio):
    select_ratio = int(len(data) * ratio)
    return data[:select_ratio]


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
parser.add_argument("--reward_pareto_noise", type=float, nargs=3, help="If used, we will have pareto noise on our reward. Parameters: 1- noise prop, 2- pareto_a, 3- pareto_m")
parser.add_argument("--reward_lomax_noise", type=float, help="If used, we will have lomax noise on our reward. Parameters: alpha")
parser.add_argument(
    "--biased_log_policy",
    action="store_true",
    default=None,
    help="If used, biased logging policy will be used.",
)
parser.add_argument("--logging_policy_cm", type=str, help="Logging policy confusion matrix path", required=False)
parser.add_argument("--gaussian_imbalance", type=float, help="We use Gaussian distribution to imbalance our dataset", required=False)
parser.add_argument(
    "--discrete_reward", 
    type=float, 
    nargs="*", 
    help="If used, we will have discrete distribution reward with descrete values", 
    required=False)

parser.add_argument("--discrete_flip", type=float, nargs=2, help="If used, we will have discrete flip", required=False)
parser.add_argument("--reward_estimator", action="store_true", default=None, help="If used we will put reward estimation in our bandit dataset.")

args = parser.parse_args()

if args.discrete_reward is not None:
    discrete_reward_len = len(args.discrete_reward)
    assert discrete_reward_len % 2 == 0
    assert abs(sum(args.discrete_reward[0:discrete_reward_len//2]) - 1.0) < 1e-5

if args.discrete_flip is not None:
    assert args.discrete_reward is not None

linear = args.linear
device = args.device
hyper_params = load_hyper_params(args.config)
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
hyper_params["discrete_reward"] = args.discrete_reward
hyper_params["discrete_flip"] = args.discrete_flip
hyper_params["reward_estimator"] = args.reward_estimator

print(hyper_params)

noise_generator = NoiseGenerator(hyper_params=hyper_params)

ul_ratio = None
labeled_proportion = 1.0
tau = args.tau
eps = 0
dataset = args.dataset
ul_ratio = args.ul
hyper_params["ul"] = args.ul
full_dataset = hyper_params["dataset"]
hyper_params["dataset_name_string"] = args.dataset
hyper_params["tau"] = args.tau
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

print(store_folder)

print("Is Linear?", linear)

if args.biased_log_policy:
    print("Logging policy is biased...")

if args.reward_flip:
    print("We will have reward flip...")

if args.reward_pareto_noise:
    print("We will have pareto noise on rewards...")

if args.reward_lomax_noise:
    print("We will have lomax noise on rewards...")

if args.discrete_reward:
    print("We will use discrete reward...")

if args.discrete_flip:
    print("We will use discrete flip on rewards...")

if args.reward_estimator:
    print("We will use reward estimator...")

if args.logging_policy_cm is not None:
    cfm_probs = initial_cfm_logging_policy(args.logging_policy_cm, hyper_params["dataset"]["num_classes"])
    probs_fn = partial(get_model_probs_from_cfm, cfm_probs=cfm_probs)
else:
    if linear:
        model = nn.Linear(
            hyper_params["feature_size"], hyper_params["dataset"]["num_classes"]
        )
        model_path = f"models/{dataset}/log_policy_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_1.0_tau1.0.pth"
    else:
        model = ModelCifar(hyper_params)
        model_path = f"models/{dataset}/log_policy_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_1.0_tau1.0.pth"
    print("Model Path: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    probs_fn = partial(get_model_probs_we, model=model, eps=eps, device=device, tau=tau)
stored_feature = None

reward_estimator_model = nn.Linear(hyper_params["feature_size"], hyper_params["dataset"]["num_classes"])
if args.reward_estimator is not None:
    reward_estimator_model_path = get_reward_estimator_model_name(dataset, linear, tau, args)
    print(reward_estimator_model_path)
    reward_estimator_model.load_state_dict(torch.load(reward_estimator_model_path))
    reward_estimator_model.to(device)

os.makedirs(f"../data/{store_folder}", exist_ok=True)

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []

# train_dataset = FashionMNIST(root="data/fmnist/", train=True)
# test_dataset = FashionMNIST(root="data/fmnist/", train=False)
# N, M = len(train_dataset), len(test_dataset)
# print("Len Train =", N)
# print("Len Test =", M)

# # Train
# for i in range(N):
#     image, label = train_dataset[i]
#     image = np.array(image).reshape(28 * 28)
#     x_train.append(image)
#     y_train.append(label)
# x_train = np.stack(x_train)
# y_train = np.stack(y_train)

# # Test
# for i in range(M):
#     image, label = test_dataset[i]
#     image = np.array(image).reshape(28 * 28)
#     x_test.append(image)
#     y_test.append(label)
# x_test = np.stack(x_test)
# y_test = np.stack(y_test)

# # Normalize X data
# x_train = x_train.astype(float) / 255.0
# x_test = x_test.astype(float) / 255.0

# # One hot the rewards
# y_train = one_hot(y_train)
# y_test = one_hot(y_test)

# # Shuffle the dataset once
# indices = np.arange(len(x_train))
# np.random.shuffle(indices)
# assert len(x_train) == len(y_train)
# x_train = x_train[indices]
# y_train = y_train[indices]
# print(x_train.shape)
# N = len(x_train)
# n = int(N * labeled_proportion)

data_set_info_cnt = {"train":{i:0.0 for i in range(hyper_params["dataset"]["num_classes"])}, 
                        "val":{i:0.0 for i in range(hyper_params["dataset"]["num_classes"])}}

if args.unbalance is not None or args.data_repeat is not None or args.gaussian_imbalance is not None:
    hyper_params["batch_size"] = 1

train_reader, test_reader, val_reader = load_data_fast(
    hyper_params, device=device, labeled=False, create_dataset=True
)


loaders = {"train": train_reader, "val": val_reader, "test": test_reader}
data = dict()

hyper_params["reward_pareto_noise"] = args.reward_pareto_noise
hyper_params["reward_lomax_noise"] = args.reward_lomax_noise

if args.gaussian_imbalance is not None:
    train_gaussian_mean = len(train_reader) / hyper_params["dataset"]["num_classes"]
    val_cnt = len(val_reader) / hyper_params["dataset"]["num_classes"]

    action_cnt = {"train":{i:0.0 for i in range(hyper_params["dataset"]["num_classes"])}, 
                 "val":{i:0.0 for i in range(hyper_params["dataset"]["num_classes"])}}

    action_gaussian_f = dict()
    for mode in ["train", "val"]:
        for x, y, action, delta, prop, _ in loaders[mode]:
            cur_actions = action.numpy()
            for action in cur_actions:
                action_cnt[mode][action] += 1
    
    print(action_cnt["train"])
    print(action_cnt["val"])

    action_gaussian_f["train"] = np.abs(np.random.normal(loc=train_gaussian_mean, scale=hyper_params["gaussian_imbalance"], 
                            size=hyper_params["dataset"]["num_classes"]).astype(int)).astype(float)
    action_gaussian_f["val"] = np.ones(hyper_params["dataset"]["num_classes"]) * val_cnt

    print(action_gaussian_f["train"])
    print(action_gaussian_f["val"])

    for cur_action in range(hyper_params["dataset"]["num_classes"]):
        for mode in ["train", "val"]:
            action_gaussian_f[mode][cur_action] = int((action_gaussian_f[mode][cur_action] / action_cnt[mode][cur_action]) * 100)

    print(action_gaussian_f["train"])
    print(action_gaussian_f["val"])

# Start creating bandit-dataset
for num_sample in [hyper_params["num_sample"]]:  # [1, 2, 3, 4, 5]:
    print("Pre-processing for num sample = " + str(num_sample))

    final_x, final_y, final_actions, final_prop, final_labeled, final_flip, final_reward_noise, final_discrete_reward, final_discrete_flip_mask, final_reward_estimate_values = [], [], [], [], [], [], [], [], [], []

    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0
    neg_cost_count = 0
    data = {}
    for mode in ["train", "val"]:
        print(f"####### MODE {mode}:")
        final_x, final_y, final_actions, final_prop, final_labeled, final_flip, final_reward_noise, final_discrete_reward, final_discrete_flip_mask, final_reward_estimate_values = (
            [],
            [],
            [],
            [],
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
            for x, y, action, delta, prop, _, nei in loaders[mode]:
                x, y, action, delta, prop = (
                    x.to(device),
                    y.to(device),
                    action.to(device),
                    delta.to(device),
                    prop.to(device),
                )
                image = x
                label = y.cpu().numpy()
                cur_actions = action.cpu().numpy()

                if args.unbalance is not None:
                    bad_class_range = args.unbalance[0]
                    omit_prop = args.unbalance[1]
                    if label[0] < bad_class_range:
                        if random.random() < omit_prop: #omit this class sample
                            continue
                if args.logging_policy_cm is not None:
                    probs = probs_fn(label)
                else:
                    probs = probs_fn(image)
                u = probs.astype(np.float64)
                if args.biased_log_policy:
                    u = calculate_biased_probs(u)
                    
                actions = []
                labeled = []
                for i in range(len(u)):
                    actionvec = np.random.multinomial(1, u[i] / np.sum(u[i]))
                    # print(actionvec)
                    # print(actionvec)
                    act = np.argmax(actionvec)
                    actions.append([act])
                    if s < n:
                        labeled.append([1.0])
                    else:
                        if label[i] == act:
                            neg_cost_count += 1
                        if mode == 'train' and args.add_unlabeled is not None and s < args.add_unlabeled + n:
                            labeled.append([2.0])
                        else:
                            labeled.append([0.0])
                    s += 1
                # print(np.argmax(probs, axis=-1))
                # print(np.argmax(probs, axis=-1))
                actions = np.array(actions)
                labeled = np.array(labeled)
                u = image2flatten(
                    image.cpu().numpy(),
                    hyper_params["dataset"],
                    hyper_params["raw_image"],
                )
                # print("################## X SHAPE:", u.shape)

                if args.gaussian_imbalance is not None:
                    data_action = cur_actions[0]
                    actions = np.array([[data_action]])
                    action_f = action_gaussian_f[mode][data_action]

                    MIN_PERC = 50.0
                    MAX_PERC = 1000.0
                    # if args.dataset == "letter":
                    #     MIN_PERC = 40

                    action_f = min(MAX_PERC, action_f)
                    action_f = max(MIN_PERC, action_f)
                    if action_f < 100:
                        if random.random() > float(action_f)/100:
                            continue
                        fix_repeat_value = 1
                        var_repeat_value = 0
                    else:
                        fix_repeat_value = int(action_f // 100)
                        var_repeat_value = float(action_f % 100) / 100

                    for repeat_ind in range(fix_repeat_value + 1):
                        if repeat_ind == fix_repeat_value and random.random() > var_repeat_value:
                            break

                        final_x.append(u)
                        final_actions.append(actions)
                        data_set_info_cnt[mode][data_action] += 1
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
                else:
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

                    if args.reward_estimator is not None:
                        #add all x and a pair rewards
                        all_estimated_rewards = get_reward_estimator_model_result(reward_estimator_model, torch.from_numpy(u), device)
                        final_reward_estimate_values.append(all_estimated_rewards)

                    if args.discrete_reward is not None:
                        discrete_reward_len = len(args.discrete_reward)
                        probs = args.discrete_reward[0:discrete_reward_len//2]
                        values = args.discrete_reward[discrete_reward_len//2:]

                        discrete_reward_array = np.random.choice(a=values, size=(len(u), 1), p=probs, replace=True)
                        final_discrete_reward.append(discrete_reward_array)

                        if args.discrete_flip is not None:
                            discrete_flip_prop = args.discrete_flip[0]
                            discrete_flip_value = args.discrete_flip[1]
                            discrete_flip_mask = (np.random.random((len(u), 1)) < discrete_flip_prop) * 1.0
                            final_discrete_flip_mask.append(discrete_flip_mask)

                    flip_array = None
                    if args.reward_flip is not None:
                        flip_array = (np.random.random((len(u), 1)) < args.reward_flip) * 1
                        final_flip.append(flip_array)
                    
                    reward_noise_array = None
                    if args.reward_pareto_noise is not None:
                        noise_prop = hyper_params["reward_pareto_noise"][0]
                        pareto_a = hyper_params["reward_pareto_noise"][1]
                        pareto_m = hyper_params["reward_pareto_noise"][2]

                        is_noisy = (np.random.random((len(u), 1)) < noise_prop) * 1
                        # pareto_noise = (np.random.pareto(pareto_a, (len(u), 1)) + 1) * pareto_m
                        pareto_noise = np.ones((len(u), 1))

                        reward_noise_array = is_noisy * pareto_noise
                        final_reward_noise.append(reward_noise_array)
                    elif args.reward_lomax_noise is not None:
                        lomax_alpha = hyper_params["reward_lomax_noise"]
                        reward_noise_array = np.random.pareto(lomax_alpha, (len(u), 1))
                        final_reward_noise.append(reward_noise_array)

                
                    expected_reward += (actions[:, 0] == label).sum()
                    total += len(label)
                
                if args.data_repeat is not None:
                    for _ in range(args.data_repeat - 1):
                        if args.unbalance is not None and label[0] < args.unbalance[0]:
                            continue
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
                        if args.reward_flip is not None:
                            final_flip.append(flip_array)
                        if args.reward_pareto_noise is not None or args.reward_lomax_noise is not None:
                            final_reward_noise.append(reward_noise_array)

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
        if args.train_ratio is not None:
            final_x = data_ratio_selector(final_x, args.train_ratio)
            final_y = data_ratio_selector(final_y, args.train_ratio)
            final_prop = data_ratio_selector(final_prop, args.train_ratio)
            final_actions = data_ratio_selector(final_actions, args.train_ratio)
            final_labeled = data_ratio_selector(final_labeled, args.train_ratio)

        print(
            final_x.shape,
            final_y.shape,
            final_prop.shape,
            final_actions.shape,
            final_labeled.shape,
        )
        
        if args.reward_flip is not None:
            final_flip = np.concatenate(final_flip, axis=0)
            if args.train_ratio is not None:
                final_flip = data_ratio_selector(final_flip, args.train_ratio)
            print("Final Flip shape:", final_flip.shape)
            data[mode] = np.concatenate(
                (final_x, final_y, final_prop, final_actions, final_labeled, final_flip), axis=1
            )
        elif args.discrete_reward is not None:
            final_discrete_reward = np.concatenate(final_discrete_reward, axis=0)
            if args.train_ratio is not None:
                final_discrete_reward = data_ratio_selector(final_discrete_reward, args.train_ratio)
            print("Final discrete reward shape:", final_discrete_reward.shape)
            if args.discrete_flip is not None:
                final_discrete_flip_mask = np.concatenate(final_discrete_flip_mask, axis=0)
                if args.train_ratio is not None:
                    final_discrete_flip_mask = data_ratio_selector(final_discrete_flip_mask, args.train_ratio)
                print("Final discrete flip mask shape:", final_discrete_flip_mask.shape)
                data[mode] = np.concatenate(
                    (final_x, final_y, final_prop, final_actions, final_labeled, final_discrete_reward, final_discrete_flip_mask), axis=1
                )
            else:
                data[mode] = np.concatenate(
                    (final_x, final_y, final_prop, final_actions, final_labeled, final_discrete_reward), axis=1
                )
        elif args.reward_pareto_noise is not None or args.reward_lomax_noise is not None:
            final_reward_noise = np.concatenate(final_reward_noise, axis=0)
            if args.train_ratio is not None:
                final_reward_noise = data_ratio_selector(final_reward_noise, args.train_ratio)
            print("Final Reward Noise Shape:", final_reward_noise.shape)
            data[mode] = np.concatenate(
                (final_x, final_y, final_prop, final_actions, final_labeled, final_reward_noise), axis=1
            )
        else:
            data[mode] = np.concatenate(
                (final_x, final_y, final_prop, final_actions, final_labeled), axis=1
            )
        
        if args.reward_estimator is not None:
            final_reward_estimate_values = np.concatenate(final_reward_estimate_values, axis=0)
            if args.train_ratio is not None:
                final_reward_estimate_values = data_ratio_selector(final_reward_estimate_values, args.train_ratio)
            data[mode] = np.concatenate((data[mode], final_reward_estimate_values), axis=1)

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

                if args.logging_policy_cm is not None:
                    probs = probs_fn(label)
                else:
                    probs = probs_fn(image)

                u = probs.astype(np.float64)
                if args.biased_log_policy:
                    u = calculate_biased_probs(u)
                    
                actions = []
                for i in range(len(u)):
                    actionvec = np.random.multinomial(1, u[i] / np.sum(u[i]))
                    act = np.argmax(actionvec)
                    actions.append([act])
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
    
    filename = f"../data/{store_folder}/bandit_data_"
    filename += "sampled_" + str(num_sample) + "_"
    print(f"Saving dataset in {filename}...")

    if args.gaussian_imbalance is not None:
        train_action_info = data_set_info_cnt["train"]
        val_action_info = data_set_info_cnt["val"]

        dataset_info["train"] += f"\n Action Info: {train_action_info}"
        dataset_info["val"] += f"\n Action Info: {val_action_info}"

    print(data["train"].shape)

    for mode in ["train", "val", "test"]:
        file_name_end = ""
        if args.train_ratio is not None:
            file_name_end += f"_{args.train_ratio}"
        if args.unbalance is not None:
            file_name_end += f"_(C={args.unbalance[0]},P={args.unbalance[1]})"
        if args.data_repeat is not None:
            file_name_end += f"_Rep={args.data_repeat}"
        if args.uniform_noise_alpha is not None:
            file_name_end += f"_UniformNoise{args.uniform_noise_alpha}"
        if args.gaussian_noise_alpha is not None:
            file_name_end += f"_GaussianNoise{args.gaussian_noise_alpha}"
        if args.gamma_noise_beta is not None:
            file_name_end += f"_GammaNoise{args.gamma_noise_beta}"
        if args.biased_log_policy:
            file_name_end += "_BiasedLoggingPolicy"
        if args.reward_flip is not None:
            file_name_end += f"_RewardFlip{args.reward_flip}"
        if args.reward_pareto_noise is not None:
            file_name_end += f"_RewardParetoNoise_{str(args.reward_pareto_noise)}_"
        if args.reward_lomax_noise is not None:
            file_name_end += f"_RewardLomaxNoise_{str(args.reward_lomax_noise)}"
        if args.logging_policy_cm is not None:
            splitted_cfm_path = args.logging_policy_cm.split("/")
            file_name_end += f"CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
        if args.gaussian_imbalance is not None:
            file_name_end += f"_ImbalanceGaussian_{args.gaussian_imbalance}"
        if args.discrete_reward is not None:
            file_name_end += f"_DiscreteReward_{args.discrete_reward}"
        if args.discrete_flip is not None:
            file_name_end += f"_DiscreteFlip_{args.discrete_flip}"
        if args.reward_estimator is not None:
            file_name_end += "_RewardEstimate"

        save_obj(data[mode], filename + mode + file_name_end)
        end = ""
        if args.train_ratio is not None:
            end += "_TrainRatio" + str(args.train_ratio)
        if args.unbalance is not None:
            end += "_Unbalanced(" + str(args.unbalance[0]) + "," + str(args.unbalance[1]) + ")"
        if args.data_repeat is not None:
            end += "_Rep=" + str(args.data_repeat)
        if args.uniform_noise_alpha is not None:
            end += "_UniformNoise" + str(args.uniform_noise_alpha)
        if args.gaussian_noise_alpha is not None:
            end += "_GaussianNoise" + str(args.gaussian_noise_alpha)
        if args.gamma_noise_beta is not None:
            end += f"_GammaNoise{args.gamma_noise_beta}"
        if args.biased_log_policy:
            end += "_BiasedLoggingPolicy"
        if args.reward_flip is not None:
            end += f"_RewardFlip{args.reward_flip}"
        if args.reward_pareto_noise is not None:
            end += f"_RewardParetoNoise_{str(args.reward_pareto_noise)}"
        if args.reward_lomax_noise is not None:
            end += f"_RewardLomaxNoise_{str(args.reward_lomax_noise)}"
        if args.logging_policy_cm is not None:
            splitted_cfm_path = args.logging_policy_cm.split("/")
            end += f"CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
        if args.gaussian_imbalance is not None:
            end += f"_ImbalanceGaussian_{args.gaussian_imbalance}"
        if args.discrete_reward is not None:
            end += f"_DiscreteReward_{args.discrete_reward}"
        if args.discrete_flip is not None:
            end += f"_DiscreteFlip_{args.discrete_flip}"
        if args.reward_estimator is not None:
            end += "_RewardEstimate"

        file_write(f"../data/{store_folder}/{mode}_info" + end, f"Logging Policy Acc = {dataset_info[mode]}")