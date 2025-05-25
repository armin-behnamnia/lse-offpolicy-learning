import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import time
from tensorboardX import SummaryWriter

writer = None

from model_h0 import ModelCifar
from eval import evaluate
from loss import CustomLoss, KLLoss, KLLossRev, SupKLLoss
from utils import *
from hyper_params import load_hyper_params
import argparse
import yaml
import numpy as np
import copy
from torchvision.datasets import FashionMNIST, CIFAR10
from tqdm import tqdm
from functools import partial
import os


def one_hot(arr):
    new = []
    for i in tqdm(range(len(arr))):
        temp = np.zeros(10)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def train(model, criterion, optimizer, scheduler, reader, hyper_params, device):
    model.train()

    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0])
    correct, total = LongTensor([0]), 0.0
    avg_correct = FloatTensor([0.0]).to(device)
    control_variate = FloatTensor([0.0])
    ips = FloatTensor([0.0])
    main_loss = FloatTensor([0.0])
    tau = hyper_params["tau"] if "tau" in hyper_params else 1.0
    print("------------> Training with temperature =", tau)

    for x, y, action, delta, prop, _ in reader:
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()

        x, y, action, delta, prop = (
            x.to(device),
            y.to(device),
            action.to(device),
            delta.to(device),
            prop.to(device),
        )
        # Forward pass
        output = model(x)
        output = F.softmax(output / tau, dim=1)

        if hyper_params.experiment.feedback == "supervised":
            loss = criterion(output, y)
        elif hyper_params.experiment.feedback == "bandit":
            loss = criterion(output, action, delta, prop)
        elif hyper_params.experiment.feedback is None:
            loss = torch.tensor(0).float().to(x.device)
        else:
            raise ValueError(
                f"Feedback type {hyper_params.experiment.feedback} is not valid."
            )
        main_loss += loss.item()
        if hyper_params.experiment.regularizers:
            if "KL" in hyper_params.experiment.regularizers:
                loss += (
                    KLLoss(output, action, prop)
                    * hyper_params.experiment.regularizers.KL
                )
            if "KL2" in hyper_params.experiment.regularizers:
                loss += (
                    KLLossRev(output, action, prop)
                    * hyper_params.experiment.regularizers.KL2
                )
            if "SupKL" in hyper_params.experiment.regularizers:
                loss += (
                    SupKLLoss(
                        output,
                        action,
                        delta,
                        prop,
                        hyper_params.experiment.regularizers.eps,
                    )
                    * hyper_params.experiment.regularizers.SupKL
                )
        loss.backward()
        optimizer.step()

        # Log to tensorboard
        writer.add_scalar("train loss", loss.item(), total_batches)

        # Metrics evaluation
        total_loss += loss.item()
        control_variate += torch.mean(
            output[range(action.size(0)), action] / prop
        ).item()
        ips += torch.mean((delta * output[range(action.size(0)), action]) / prop).item()
        predicted = torch.argmax(output, dim=1)
        # print(predicted, y)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        avg_correct += output[range(action.size(0)), y].sum().item()
        total_batches += 1.0
    scheduler.step()

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    metrics["AvgAcc"] = round(100.0 * float(avg_correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total_batches, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)

    return metrics


def main(
    config_path,
    device="cuda:0",
    return_model=False,
    proportion=1.0,
    tau=1.0,
    dataset="cifar",
):
    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = load_hyper_params(config_path, proportion=proportion)
    hyper_params["feature_size"] = 3 * 32 * 32 if dataset == "cifar" else 1 * 28 * 28
    print(hyper_params)
    print(f"Training with {proportion} of the data")
    if hyper_params.experiment.regularizers:
        if "KL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer KL added: {hyper_params.experiment.regularizers.KL}"
            )
        if "KL2" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Reverse KL added: {hyper_params.experiment.regularizers.KL2}"
            )
        if "SupKL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Supervised KL added: {hyper_params.experiment.regularizers.SupKL}"
            )

    # Initialize a tensorboard writer
    global writer
    path = hyper_params["tensorboard_path"]
    writer = SummaryWriter(path)
    hyper_params["tau"] = tau

    # Train It..
    train_reader, test_reader, val_reader = load_data(hyper_params, labeled=False)
    file_write(
        hyper_params["log_file"],
        "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n",
    )
    file_write(hyper_params["log_file"], "Data reading complete!")
    file_write(
        hyper_params["log_file"],
        "Number of train batches: {:4d}".format(len(train_reader)),
    )
    file_write(
        hyper_params["log_file"],
        "Number of test batches: {:4d}".format(len(test_reader)),
    )

    if hyper_params.experiment.feedback == "supervised":
        print("Supervised Training.")
        criterion = nn.CrossEntropyLoss()
    elif hyper_params.experiment.feedback == "bandit":
        print("Bandit Training")
        criterion = CustomLoss(hyper_params)
    elif hyper_params.experiment.feedback is None:
        criterion = None
    else:
        raise ValueError(
            f"Feedback type {hyper_params.experiment.feedback} is not valid."
        )

    try:
        best_model_dict = None
        best_metrics_total = []
        for exp in range(hyper_params.experiment.n_exp):
            model = ModelCifar(hyper_params)
            model.to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hyper_params["lr"],
                momentum=0.9,
                weight_decay=hyper_params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=25, gamma=0.5, verbose=True
            )
            file_write(
                hyper_params["log_file"], "\nModel Built!\nStarting Training...\n"
            )
            file_write(
                hyper_params["log_file"],
                f"################################ MODEL ITERATION {exp + 1}:\n--------------------------------",
            )
            best_acc = 0
            best_metrics = None
            for epoch in range(1, hyper_params["epochs"] + 1):
                epoch_start_time = time.time()

                # Training for one epoch
                metrics = train(
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    train_reader,
                    hyper_params,
                    device=device,
                )

                string = ""
                for m in metrics:
                    string += " | " + m + " = " + str(metrics[m])
                string += " (TRAIN)"

                for metric in metrics:
                    writer.add_scalar(
                        f"Train_metrics/exp_{exp}/" + metric, metrics[metric], epoch - 1
                    )

                # Calulating the metrics on the validation set
                metrics = evaluate(
                    model,
                    criterion,
                    val_reader,
                    hyper_params,
                    labeled=False,
                    device=device,
                )
                string2 = ""
                for m in metrics:
                    string2 += " | " + m + " = " + str(metrics[m])
                string2 += " (VAL)"

                for metric in metrics:
                    writer.add_scalar(
                        f"Validation_metrics/exp_{exp}/" + metric,
                        metrics[metric],
                        epoch - 1,
                    )

                ss = "-" * 89
                ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                    epoch, (time.time() - epoch_start_time)
                )
                ss += string
                ss += "\n"
                ss += "-" * 89
                ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                    epoch, (time.time() - epoch_start_time)
                )
                ss += string2
                ss += "\n"
                ss += "-" * 89

                if metrics["Acc"] > best_acc:
                    best_acc = metrics["Acc"]
                    best_model_dict = copy.deepcopy(model.state_dict())
                    metrics = evaluate(
                        model, criterion, test_reader, hyper_params, device=device
                    )
                    string3 = ""
                    for m in metrics:
                        string3 += " | " + m + " = " + str(metrics[m])
                    string3 += " (TEST)"

                    ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                        epoch, (time.time() - epoch_start_time)
                    )
                    ss += string3
                    ss += "\n"
                    ss += "-" * 89

                    for metric in metrics:
                        writer.add_scalar(
                            f"Test_metrics/exp_{exp}/" + metric,
                            metrics[metric],
                            epoch - 1,
                        )
                    best_metrics = metrics

                file_write(hyper_params["log_file"], ss)
            best_metrics_total.append(best_metrics)

    except KeyboardInterrupt:
        print("Exiting from training early")

    writer.close()

    model_summary = {k: [] for k in best_metrics_total[0].keys()}
    for metric in best_metrics_total:
        for k, v in metric.items():
            model_summary[k].append(v)
    for k, v in model_summary.items():
        model_summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    file_write(hyper_params["summary_file"], yaml.dump(model_summary))

    if return_model == True:
        return best_model_dict
    return best_metrics_total


def get_model_probs_we(image, model, eps, dataset, device):
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
            torch.softmax(model(image.to(device)), dim=-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    probs[probs < eps] = eps
    probs /= np.sum(probs, axis=-1, keepdims=True)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs


stored_feature = None


def forward_hook(module, module_in, module_out):
    global stored_feature
    stored_feature = module_in[0].detach().cpu().squeeze().numpy()
    # print("forward hook done.")


def create_feature_data(feature_model, dataset, device, hyper_params):
    print(f"#### CREAT FEATURE DATA FROM DATASET {dataset}...")
    global stored_feature

    os.makedirs(f"../data/features/{dataset}", exist_ok=True)
    if not hyper_params["raw_image"]:
        print("NOT RAW IMAGE")
        _ = feature_model.resnet.linear.register_forward_hook(forward_hook)
    if not hyper_params["raw_image"]:
        print("NOT RAW IMAGE")
        _ = feature_model.resnet.linear.register_forward_hook(forward_hook)

    # x_train, x_test = [], []
    # y_train, y_test = [], []
    probs_fn = partial(
        get_model_probs_we, model=feature_model, eps=0, dataset=dataset, device=device
    )

    train_reader, test_reader, val_reader = load_data(hyper_params, labeled=False)

    loaders = {"train": train_reader, "val": val_reader, "test": test_reader}
    data = dict()
    for mode in ["train", "val"]:
        final_x, final_y, final_actions, final_prop = [], [], [], []

        avg_num_zeros = 0.0
        expected_reward = 0.0
        total = 0.0
        neg_cost_count = 0
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

            probs = probs_fn(image)
            # print(probs.shape)
            if hyper_params["raw_image"]:
                image = image.reshape(hyper_params["batch_size"], -1).cpu().numpy()
            else:
                image = stored_feature
            u = probs.astype(np.float64)
            actions = []
            for i in range(len(u)):
                actionvec = np.random.multinomial(1, u[i] / np.sum(u[i]))
                act = np.argmax(actionvec)
                actions.append([act])
            actions = np.array(actions)
            final_x.append(image)
            final_actions.append(actions)
            final_prop.append(probs)
            # print(final_prop[0].shape)
            final_y.append(F.one_hot(y, num_classes=10).cpu().numpy())
            # print(actions, label)
            expected_reward += (actions[:, 0] == label).sum()
            total += len(label)

        avg_num_zeros /= total
        avg_num_zeros = round(avg_num_zeros, 4)
        print("Num sample = 1" + "; Acc = " + str(100.0 * expected_reward / total))
        print("Neg reward proportion = " + str(neg_cost_count / total))
        print()
        final_x = np.concatenate(final_x, axis=0)
        final_y = np.concatenate(final_y, axis=0)
        final_prop = np.concatenate(final_prop, axis=0)
        final_actions = np.concatenate(final_actions, axis=0)
        # print(final_x.shape, final_y.shape, final_prop.shape, final_actions.shape)
        # Save as CSV
        # if labeled_proportion < 1.0:
        data[mode] = np.concatenate(
            (final_x, final_y, final_prop, final_actions), axis=1
        )

    for mode in ["test"]:
        final_x, final_y, final_actions, final_prop = [], [], [], []

        avg_num_zeros = 0.0
        expected_reward = 0.0
        total = 0.0
        neg_cost_count = 0
        for x, y, action, delta, prop in loaders[mode].iter():
            x, y, action, delta, prop = (
                x.to(device),
                y.to(device),
                action.to(device),
                delta.to(device),
                prop.to(device),
            )
            image = x
            label = y.cpu().numpy()

            probs = probs_fn(image)
            if hyper_params["raw_image"]:
                image = image.reshape(hyper_params["batch_size"], -1).cpu().numpy()
            else:
                image = stored_feature
            if hyper_params["raw_image"]:
                image = image.reshape(hyper_params["batch_size"], -1).cpu().numpy()
            else:
                image = stored_feature
            u = probs.astype(np.float64)
            actions = []
            for i in range(len(u)):
                actionvec = np.random.multinomial(1, u[i] / np.sum(u[i]))
                act = np.argmax(actionvec)
                actions.append([act])
            actions = np.array(actions)
            final_x.append(image)
            final_actions.append(actions)
            final_prop.append(probs)
            # print(final_prop[0].shape)
            final_y.append(F.one_hot(y, num_classes=10).cpu().numpy())
            # print(actions, label)
            expected_reward += (actions[:, 0] == label).sum()
            total += len(label)

        avg_num_zeros /= total
        avg_num_zeros = round(avg_num_zeros, 4)
        print("Acc = " + str(100.0 * expected_reward / total))
        print()
        final_x = np.concatenate(final_x, axis=0)
        final_y = np.concatenate(final_y, axis=0)
        final_prop = np.concatenate(final_prop, axis=0)
        final_actions = np.concatenate(final_actions, axis=0)
        # print(final_x.shape, final_y.shape, final_prop.shape, final_actions.shape)
        # Save as CSV
        # if labeled_proportion < 1.0:
        data[mode] = np.concatenate(
            (final_x, final_y, final_prop, final_actions), axis=1
        )
    # if dataset == "fmnist":
    #     train_dataset = FashionMNIST(
    #         root=f"../data/{dataset}/", train=True, download=True
    #     )
    #     test_dataset = FashionMNIST(
    #         root=f"../data/{dataset}/", train=False, download=True
    #     )
    # else:
    #     train_dataset = CIFAR10(root=f"../data/{dataset}/", train=True, download=True)
    #     test_dataset = CIFAR10(root=f"../data/{dataset}/", train=False, download=True)
    # N, M = len(train_dataset), len(test_dataset)
    # print("Len Train =", N)
    # print("Len Test =", M)

    # probs_fn = partial(
    #     get_model_probs_we, model=feature_model, eps=0, dataset=dataset, device=device
    # )
    # feature_size = 3072 if dataset == "cifar" else 784
    # # Train
    # for i in range(N):
    #     image, label = train_dataset[i]
    #     image = np.array(image).reshape(feature_size)
    #     x_train.append(image)
    #     y_train.append(label)
    # x_train = np.stack(x_train)
    # y_train = np.stack(y_train)

    # # Test
    # for i in range(M):
    #     image, label = test_dataset[i]
    #     image = np.array(image).reshape(feature_size)
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

    # Shuffle the dataset once
    # indices = np.arange(len(x_train))
    # np.random.shuffle(indices)
    # assert len(x_train) == len(y_train)
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    # print(x_train.shape)
    # N = len(x_train)

    # for point_num in tqdm(range(x_train.shape[0])):
    #     image = x_train[point_num]
    #     label = np.argmax(y_train[point_num])

    #     probs = probs_fn(image)
    #     image = stored_feature
    #     u = probs.astype(np.float64)
    #     actionvec = np.random.multinomial(1, u / np.sum(u))
    #     action = np.argmax(actionvec)
    #     print(action, label)

    #     final_x.append(image)
    #     final_actions.append([action])
    #     final_prop.append(probs)

    #     final_y.append(y_train[point_num])

    #     expected_reward += float(int(action == label))
    #     total += 1.0
    #     # Printing the first prob. dist.
    #     # if point_num == 0: print("Prob Distr. for 0th sample:\n", [ round(i, 3) for i in list(probs) ])

    # avg_num_zeros /= float(x_train.shape[0])
    # avg_num_zeros = round(avg_num_zeros, 4)
    # print("Num sample = 1" + "; Acc = " + str(100.0 * expected_reward / total))
    # print("Neg reward proportion = " + str(neg_cost_count / total))
    # print()

    # # Save as CSV
    # # if labeled_proportion < 1.0:
    # final_normal = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)
    # # else:
    # #     final_normal = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)
    # if dataset == "fmnist":
    #     train_size = 50000
    # elif dataset == "cifar":
    #     train_size = 45000
    # else:
    #     raise ValueError(f"Dataset {dataset} not valid.")

    # N = len(final_normal)
    # idx = list(range(N))
    # idx = np.random.permutation(idx)
    # train = final_normal[idx[:train_size]]
    # val = final_normal[idx[train_size:]]

    # avg_num_zeros = 0.0
    # expected_reward = 0.0
    # total = 0.0

    # test_prop, test_actions = [], []
    # xs = []
    # for i, label in tqdm(enumerate(y_test)):
    #     label = np.argmax(label)
    #     image = x_test[i]
    #     probs = probs_fn(image)
    #     xs.append(stored_feature)
    #     u = probs.astype(np.float64)
    #     test_prop.append(probs)
    #     actionvec = np.random.multinomial(1, u / np.sum(u))
    #     action = np.argmax(actionvec)
    #     test_actions.append([action])

    #     expected_reward += float(int(action == label))
    #     total += 1.0

    # print("Acc = " + str(100.0 * expected_reward / total))
    # print()

    # test = np.concatenate((xs, y_test, test_prop, test_actions), axis=1)  # Garbage

    filename = f"../data/features{'_raw' if hyper_params['raw_image'] else ''}/{dataset}/bandit_data_"
    filename += "sampled_1_"

    # for mode in ["train", "test", "val"]:
    #     save_obj(data[mode], filename + mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to experiment config file."
    )
    parser.add_argument("-d", "--device", required=True, type=str, help="Device")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--raw_image", action="store_true")
    parser.add_argument("--raw_image", action="store_true")
    args = parser.parse_args()
    hyper_params = load_hyper_params(args.config, proportion=1.0)
    hyper_params["raw_image"] = args.raw_image
    hyper_params["raw_image"] = args.raw_image
    dataset = args.dataset
    if dataset == "fmnist":
        from data_fmnist import load_data

        hyper_params["feature_size"] = 784
    else:
        from data import load_data

        hyper_params["feature_size"] = 3072

    device = args.device

    # print(f"#### TRAIN FEATURE EXTRACTOR MODEL...")
    # model_dict = main(
    #     args.config,
    #     device=args.device,
    #     return_model=True,
    #     proportion=1.0,
    #     tau=1.0,
    #     dataset=dataset,
    # )
    # torch.save(model_dict, f"models/main/{dataset}.pth")
    model_dict = torch.load(f"models/main/{dataset}.pth")

    feature_model = ModelCifar(hyper_params)
    feature_model.load_state_dict(model_dict)
    feature_model.to(device)

    create_feature_data(
        feature_model=feature_model,
        dataset=dataset,
        device=device,
        hyper_params=hyper_params,
    )
