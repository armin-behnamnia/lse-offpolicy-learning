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
from utils import dataset_mapper
from data import load_data_fast
import os
from tqdm import tqdm

STOP_THRESHOLD = 20

class CustomLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLinearModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def train_one_epoch(model, criterion, optimizer, scheduler, reader, hyper_params, device):
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
    # print(model)
    for x, y, action, delta, prop, _, nei in tqdm(reader):
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
        if hyper_params["use_n_hot"] is not None:
            output = F.sigmoid(output)
        else:
            output = F.softmax(output / tau, dim=1) #this soft is not necessary because cross entropy has it

        if hyper_params.experiment.feedback == "supervised":
            if hyper_params["propensity_estimation"] is not None:
                loss = criterion(output, action)
            else:
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
        total_loss += loss.item() #with regulizer
        control_variate += torch.mean(
            output[range(action.size(0)), action] / prop
        ).item()
        ips += torch.mean((delta * output[range(action.size(0)), action]) / prop).item()
        predicted = torch.argmax(output, dim=1)
        # print(predicted, y)
        total += y.size(0)
        if hyper_params["propensity_estimation"] is not None:
            correct += (predicted == action).sum().item()
        elif hyper_params["use_n_hot"] is not None:
            correct += (predicted == torch.argmax(y, dim=1)).sum().item()
        else:
            correct += (predicted == y).sum().item()
        if hyper_params["use_n_hot"] is None:
            avg_correct += output[range(action.size(0)), y].sum().item()
        total_batches += 1.0
    scheduler.step()

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    if hyper_params["use_n_hot"] is None:
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
    linear=False,
    raw_image=False,
    feature_size=None,
    train_ratio=None,
    propensity_estimation=None,
    use_npz=None,
    use_n_hot=None
):
    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = load_hyper_params(config_path, proportion=proportion, train_ratio=train_ratio, 
                                     propensity_estimation=propensity_estimation, use_npz=use_npz, use_n_hot=use_n_hot)
    hyper_params["raw_image"] = raw_image
    print(f"Training with {proportion} of the data")
    if train_ratio is not None:
        print(f"Dataset train ratio = {train_ratio}")
    if propensity_estimation is not None:
        print("We will use logging policy as ground truth...")
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
    global STOP_THRESHOLD
    print(f"Feature Size = {feature_size}")
    path = hyper_params["tensorboard_path"]
    writer = SummaryWriter(path)
    raw_image = hyper_params["raw_image"]
    hyper_params["tau"] = tau
    hyper_params["train_ratio"] = train_ratio
    hyper_params["propensity_estimation"] = propensity_estimation
    full_dataset = hyper_params["dataset"]
    hyper_params["dataset"] = dataset_mapper[dataset]
    hyper_params["dataset"]["name"] = full_dataset
    hyper_params["dataset_name_string"] = dataset
    if feature_size is not None:
        hyper_params["feature_size"] = feature_size
    else:
        hyper_params["feature_size"] = np.prod(hyper_params["dataset"]["data_shape"])
    # if linear and not hyper_params["raw_image"]:
    #     hyper_params["feature_size"] = 512
    # if linear:
    #     if raw_image:
    #         hyper_params["feature_size"] = np.prod(dataset_mapper[dataset]["sizes"])
    # elif dataset == "cifar":
    #     hyper_params["feature_size"] = 3072
    # else:
    #     hyper_params["feature_size"] = 784
    # print(hyper_params)
    # Train It..

    #the device input has not been used in this function
    train_reader, test_reader, val_reader = load_data_fast(
        hyper_params, device=device, labeled=False, create_dataset=True
    )

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
        best_metrics_total = []
        for exp in range(hyper_params.experiment.n_exp):
            not_improved = 0
            if "wiki" in hyper_params["dataset_name_string"]:
                print("Custom Linear Model")
                model = CustomLinearModel(hyper_params["feature_size"], hyper_params["dataset"]["num_classes"])
            elif linear:
                model = nn.Linear(
                    hyper_params["feature_size"], hyper_params["dataset"]["num_classes"]
                )
            else:
                model = ModelCifar(hyper_params)
            model.to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hyper_params["lr"],
                momentum=0.9,
                weight_decay=hyper_params["weight_decay"],
            )
            if "lr_sch" in hyper_params:
                if hyper_params["lr_sch"] == "OneCycle":
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=hyper_params["lr"],
                        epochs=hyper_params["epochs"],
                        steps_per_epoch=len(train_reader),
                    )
                elif hyper_params["lr_sch"] == "CosineAnnealingLR":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=hyper_params["epochs"], verbose=True
                    )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.9, verbose=True
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
            best_model_dict = None
            best_loss = 10000000000.0

            for epoch in range(1, hyper_params["epochs"] + 1):
                epoch_start_time = time.time()

                # Training for one epoch
                metrics = train_one_epoch(
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
                    device=device,
                    labeled=False,
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

                if (hyper_params["use_n_hot"] is None and metrics["Acc"] > best_acc) or (hyper_params["use_n_hot"] is not None and metrics["loss"] < best_loss): #best_loss
                    not_improved = 0
                    best_acc = metrics["Acc"]
                    best_model_dict = copy.deepcopy(model.state_dict())
                    metrics = evaluate(
                        model,
                        criterion,
                        test_reader,
                        hyper_params,
                        device=device,
                        labeled=True,
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
                else:
                    not_improved += 1

                file_write(hyper_params["log_file"], ss)

                if not_improved >= STOP_THRESHOLD:
                    print("STOP THRESHOLD PASSED.")
                    break

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to experiment config file."
    )
    parser.add_argument(
        "-p",
        "--proportion",
        required=False,
        default=1.0,
        type=float,
        help="Proportion of data to be trained with",
    )
    parser.add_argument("-d", "--device", required=True, type=str, help="Device")
    parser.add_argument(
        "-t", "--tau", required=True, type=float, help="Softmax temperature"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset to train the logging policy on. Can be either fmnist or cifar.",
    )
    parser.add_argument(
        "--linear",
        required=False,
        action="store_true",
        help="If used, linear logging policy is trained.",
    )
    parser.add_argument(
        "--raw_image",
        required=False,
        action="store_true",
        help="If used, raw flatten image is given to the model instead of pretrained features.",
    )
    parser.add_argument(
        "--feature_size",
        required=False,
        type=int,
        help="If used, given feature size is supposed for the context.",
    )
    parser.add_argument("--train_ratio", type=float, help="What ratio of train dataset will be used for training", required=False)
    parser.add_argument(
        "--propensity_estimation",
        action="store_true",
        default=None,
        help="If used, We will use logging policy action as ground truth",
    )
    parser.add_argument(
        "--use_npz",
        action="store_true",
        default=None,
        help="If used, We will load and store data as npz files",
    )
    parser.add_argument(
        "--use_n_hot",
        action="store_true",
        default=None,
        help="If used, we will use y as n hot vector not int",
    )

    args = parser.parse_args()
    proportion = args.proportion
    tau = args.tau
    dataset = args.dataset
    linear = args.linear
    
    model_dict = main(
        args.config,
        device=args.device,
        return_model=True,
        proportion=proportion,
        tau=tau,
        dataset=dataset,
        linear=linear,
        raw_image=args.raw_image,
        feature_size=args.feature_size,
        train_ratio=args.train_ratio,
        propensity_estimation=args.propensity_estimation,
        use_npz=args.use_npz,
        use_n_hot=args.use_n_hot
    )
    
    os.makedirs(f"models/{dataset}/", exist_ok=True)
    if args.propensity_estimation is not None:
        torch.save(
            model_dict,
            f"models/{dataset}/propensity_estimator_v2_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_{proportion}_tau{tau}.pth",
        )        
    elif args.train_ratio is not None:
        torch.save(
            model_dict,
            f"models/{dataset}/log_policy_v2_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_{proportion}_tau{tau}_TrainRatio{args.train_ratio}.pth",
        )
    else:
        torch.save(
            model_dict,
            f"models/{dataset}/log_policy_v2_{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_{proportion}_tau{tau}.pth",
        )        
