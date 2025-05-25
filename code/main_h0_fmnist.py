import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import time
from tensorboardX import SummaryWriter

writer = None

from model_h0 import ModelCifar
from data_f_base import load_data
from eval import evaluate
from loss import CustomLoss, KLLoss, KLLossRev, SupKLLoss
from utils import *
from hyper_params import load_hyper_params
import argparse
import yaml
import numpy as np


def train(model, criterion, optimizer, scheduler, reader, hyper_params, device):
    model.train()

    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0])
    correct, total = LongTensor([0]), 0.0
    control_variate = FloatTensor([0.0])
    ips = FloatTensor([0.0])
    main_loss = FloatTensor([0.0])
    tau = hyper_params["tau"] if "tau" in hyper_params else 1.0
    print("------------> Training with temperature =", tau)

    for x, y, action, delta, prop in reader.iter():
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
        total_batches += 1.0
    scheduler.step()

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total_batches, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)

    return metrics


def main(config_path, device="cuda:0", return_model=False, proportion=1.0, tau=1.0):
    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = load_hyper_params(config_path, proportion=proportion)
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
    train_reader, test_reader, val_reader = load_data(hyper_params)
    print(len(train_reader), len(test_reader), len(val_reader))

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
                    model, criterion, val_reader, hyper_params, device=device
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
        return model
    return best_metrics_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to experiment config file."
    )
    parser.add_argument(
        "-p",
        "--proportion",
        required=True,
        type=float,
        help="Proportion of data to be trained with",
    )
    parser.add_argument("-d", "--device", required=True, type=str, help="Device")
    parser.add_argument(
        "-t", "--tau", required=True, type=float, help="Softmax temparature"
    )
    args = parser.parse_args()
    proportion = args.proportion
    tau = args.tau
    model = main(
        args.config,
        device=args.device,
        return_model=True,
        proportion=proportion,
        tau=tau,
    )
    torch.save(model.state_dict(), f"models/fmnist/h0_{proportion}_tau{tau}.pth")
