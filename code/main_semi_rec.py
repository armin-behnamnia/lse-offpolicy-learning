import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import time
from tensorboardX import SummaryWriter

writer = None

from model import ModelRec
from eval_rec import evaluate
from loss import KLLossRec, KLLossRevRec, LSE_LossRec, PowerMeanLossRec, ExponentialSmoothingLossRec, AlphaRenyiLossRec, CustomLossRec
from utils import *
from hyper_params import load_hyper_params
import argparse
import yaml
import numpy as np
import optuna
from functools import partial
import os
from copy import deepcopy
from optuna.trial import TrialState
from torch.utils.data import DataLoader, TensorDataset
from utils import create_tensors
from tqdm import tqdm
from data_rec import load_data_fast

torch.set_printoptions(8)


def add_subpath(path, id):
    return path + "/" + str(id)


def dirname(path):
    return "/".join(path.split("/")[:-1])


STOP_THRESHOLD = 10


def group_softmax(output, source, batch_idx):
    index = batch_idx
    max_values = torch.scatter_reduce(
        input=torch.ones(len(output)).to(output.device) * -100000,
        src=source,
        index=index,
        reduce="amax",
        dim=0,
    ).detach()
    # print("MAX_VALUES=", max_values.min().item(), max_values.max().item())

    source = torch.exp(source - max_values[batch_idx])
    output = torch.exp(output - max_values)
    den = torch.scatter_reduce(
        input=torch.zeros(len(output)).to(output.device),
        src=source,
        index=index,
        reduce="sum",
        dim=0,
    )
    # print(den.shape, output.shape)
    # print("DEN < OUT=", (den < output).sum())
    softmax_values = (output + 1e-8) / (den + 1e-8)
    return softmax_values


def normalized_sigmoid(output, source, batch_idx):
    index = batch_idx
    return torch.sigmoid(output) / torch.scatter_reduce(
        input=torch.zeros(len(output)).to(output.device),
        src=torch.sigmoid(source),
        index=index,
        reduce="sum",
        dim=0,
    )


def train(trial, model, criterion, optimizer, scheduler, reader, hyper_params, device):
    ignore_unlabeled = hyper_params["ignore_unlabeled"]
    # add_unlabeled = hyper_params["add_unlabeled"]
    # print("--------> Added unlabeled data =", add_unlabeled)
    model.train()

    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0])
    correct, total = 0, 0.0
    control_variate = FloatTensor([0.0])
    # avg_correct = FloatTensor([0.0])
    ips = FloatTensor([0.0])
    main_loss = FloatTensor([0.0])
    N = len(reader.dataset)

    item_matrix = torch.tensor(reader.dataset.item_matrix).float().to(device)
    for (
        x,
        user_ids,
        action,
        prop,
        reward,
        labeled,
        all_rewards,
        all_actions,
        batch_idx,
    ) in tqdm(reader):
        # Empty the gradients
        optimizer.zero_grad()

        (
            x,
            user_ids,
            action,
            prop,
            reward,
            labeled,
            all_rewards,
            all_actions,
            batch_idx,
        ) = (
            x.to(device),
            user_ids.to(device),
            action.to(device),
            prop.to(device),
            reward.to(device),
            labeled.to(device),
            all_rewards.to(device),
            all_actions.to(device),
            batch_idx.to(device),
        )
        # Forward pass
        # print(batch_idx)
        # print(x.shape, item_matrix[action].shape)
        # print(x[batch_idx].shape, item_matrix[all_actions].shape)
        # output = model(x, item_matrix[action])
        # print(output.max().item(), output.min().item())
        # print(torch.sigmoid(output).max(), torch.sigmoid(output).min(), output.dtype)
        # print(prop.max(), prop.min(), prop.dtype)
        source = model(x[batch_idx], item_matrix[all_actions])
        if torch.isnan(source).any():
            break
        # print(action)
        # print(all_actions)
        nonzeros = (all_actions == action[batch_idx]).nonzero()[:, 0]
        output = source[nonzeros]
        # print("SHAPES=", source.shape, output.shape, nonzeros.shape)
        # print(output.shape, action.shape)
        assert len(output) == len(action)
        # print("Output=", output)
        # print("IDX=", batch_idx)
        # print("Action=", action)
        # print("Reward=", reward)
        # print("Source=", source)
        # d = output[batch_idx]
        # for i in range(len(source)):
        #     print(d[i].item(), source[i].item())
        output = group_softmax(output, source, batch_idx)
        # print("Output=", output)
        # print("_--------------------------------------_")
        # print(output.max(), output.min(), output.dtype)

        # print(output.shape, output)
        output_labeled = output[labeled == 1]
        action_labeled = action[labeled == 1]
        reward_labeled = reward[labeled == 1]
        prop_labeled = prop[labeled == 1]
        su = (labeled == 1).sum()
        # print("SU SIZE=", su, output_labeled)
        if su > 0:
            if hyper_params.experiment.feedback == "supervised":
                raise ValueError(
                    "Supervised training not implemented for Rec datasets."
                )
            elif hyper_params.experiment.feedback == "bandit":
                # if hyper_params.as_reward:
                #     loss = criterion(output_labeled, action_labeled, reward_labeled)
                # else:
                loss = criterion(output_labeled, reward_labeled, prop_labeled)
            elif hyper_params.experiment.feedback is None:
                loss = torch.tensor(0).float().to(x.device)
            else:
                raise ValueError(
                    f"Feedback type {hyper_params.experiment.feedback} is not valid."
                )
        else:
            loss = torch.tensor(0).float().to(x.device)
        # print(delta.mean().item(), y.cpu().numpy(), action.cpu().numpy(), prop.mean().item())
        # print("IPS Loss value =", loss.item())
        main_loss += loss.item()
        # print("LOSS0=", loss)
        reg_output = output[labeled > 0] if ignore_unlabeled else output
        reg_action = action[labeled > 0] if ignore_unlabeled else action
        reg_prop = prop[labeled > 0] if ignore_unlabeled else prop
        # print("len data = ", len(reg_output))
        if hyper_params.experiment.regularizers:
            if len(reg_output) > 0:
                if "KL" in hyper_params.experiment.regularizers:
                    loss += (
                        KLLossRec(
                            reg_output,
                            reg_action,
                            reg_prop,
                        )
                        * hyper_params.experiment.regularizers.KL
                    )
                if "KL2" in hyper_params.experiment.regularizers:
                    loss += (
                        KLLossRevRec(
                            reg_output,
                            reg_action,
                            reg_prop,
                        )
                        * hyper_params.experiment.regularizers.KL2
                    )
            if "AlphaRenyi" in hyper_params.experiment.regularizers:
                loss += AlphaRenyiLossRec(output=reg_output, action=reg_action, prop=reg_prop, hyper_params=hyper_params)
            if "SupKL" in hyper_params.experiment.regularizers:
                raise ValueError("SupKL not implemented for Rec datasets.")
                # if su > 0:
                #     loss += (
                #         SupKLLoss(
                #             output_labeled,
                #             action_labeled,
                #             delta_labeled,
                #             prop_labeled,
                #             hyper_params.experiment.regularizers.eps,
                #             action_size=hyper_params["dataset"]["num_classes"],
                #         )
                #         * hyper_params.experiment.regularizers.SupKL
                #     )
        # print("IPS+REG Loss value =", loss.item(), "\n\n")
        # print(loss.requires_grad)
        # print("LOSS1=", loss)
        if loss.requires_grad:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # for param in model.parameters():
            #     print("GRAD=", param.grad.max().item(), param.grad.min().item())
            optimizer.step()
            if "lr_sch" in hyper_params and hyper_params["lr_sch"] == "OneCycle":
                scheduler.step()
        # else:
        #     print("No grad to optimize!!!")

        # Log to tensorboard
        writer.add_scalar("train loss", loss.item(), total_batches)

        # Metrics evaluation
        total_loss += loss.item()
        control_variate += torch.mean(output / prop).item()
        ips += torch.mean((reward * output) / prop).item()
        # predicted = torch.argmax(output, dim=1)
        # print(predicted, y)
        total += output.size(0)
        correct += (reward == 1).sum().item()
        total_batches += 1.0
        # exit()
    if "lr_sch" not in hyper_params or hyper_params["lr_sch"] != "OneCycle":
        scheduler.step()

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    # metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total_batches, 4)
    metrics["IPS"] = round(float(ips) / total_batches, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)

    return metrics


def prepare_hyperparams(trial, hyper_params):
    if trial is None:
        trial_id = 1000
    else:
        trial_id = trial._trial_id
    hyper_params["tensorboard_path"] = add_subpath(
        hyper_params["tensorboard_path"], trial_id
    )
    hyper_params["output_path"] = add_subpath(hyper_params["output_path"], trial_id)
    hyper_params["log_file"] = add_subpath(hyper_params["log_file"], trial_id)
    hyper_params["summary_file"] = add_subpath(hyper_params["summary_file"], trial_id)
    print(dirname(hyper_params["summary_file"]), hyper_params["summary_file"])
    os.makedirs(dirname(hyper_params["tensorboard_path"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["log_file"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["summary_file"]), exist_ok=True)
    if hyper_params["save_model"]:
        os.makedirs(dirname(hyper_params["output_path"]), exist_ok=True)
    if hyper_params.experiment.regularizers:
        if "AlphaRenyi" in hyper_params.experiment.regularizers:
            print("--> Regularizer Alpha-Renyi added...")
            if trial is not None:
                hyper_params["ar_beta"] = trial.suggest_float(
                    "AR_beta",
                    hyper_params["ar_beta"][0],
                    hyper_params["ar_beta"][1],
                    log=True
                )
        if "KL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer KL added: {hyper_params.experiment.regularizers.KL}"
            )
            if trial is not None:
                hyper_params.experiment.regularizers.KL = trial.suggest_float(
                    "KL_coef",
                    hyper_params.experiment.regularizers.KL[0],
                    hyper_params.experiment.regularizers.KL[1],
                    log=True,
                )
        if "KL2" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Reverse KL added: {hyper_params.experiment.regularizers.KL2}"
            )
            if trial is not None:
                hyper_params.experiment.regularizers.KL2 = trial.suggest_float(
                    "KL2_coef",
                    hyper_params.experiment.regularizers.KL2[0],
                    hyper_params.experiment.regularizers.KL2[1],
                    log=True,
                )
        if "SupKL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Supervised KL added: {hyper_params.experiment.regularizers.SupKL}"
            )
            if trial is not None:
                hyper_params.experiment.regularizers.SupKL = trial.suggest_float(
                    "SupKL_coef",
                    hyper_params.experiment.regularizers.SupKL[0],
                    hyper_params.experiment.regularizers.SupKL[1],
                    log=True,
                )
    if (trial is not None) and (hyper_params["disable_weight_decay"] is None):
        hyper_params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            hyper_params["weight_decay"][0],
            hyper_params["weight_decay"][1],
            log=True,
        )
    print(hyper_params)
    return hyper_params

def get_experiment_optimizer(model, hyper_params):
    if hyper_params["disable_weight_decay"] is not None: #we will not use weight decay
        return torch.optim.SGD(model.parameters(), lr=hyper_params["lr"], momentum=0.9) 
    return torch.optim.SGD(model.parameters(), lr=hyper_params["lr"], momentum=0.9, weight_decay=hyper_params["weight_decay"])   


def main(    
    trial,
    hyper_params,
    device="cuda:0",
    return_model=False,
    as_reward=False,
    reward_model=None,):
    STOP_THRESHOLD = 30
    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = deepcopy(hyper_params)
    hyper_params = prepare_hyperparams(trial, hyper_params)
    print(dirname(hyper_params["summary_file"]), hyper_params["summary_file"])
    os.makedirs(dirname(hyper_params["tensorboard_path"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["log_file"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["summary_file"]), exist_ok=True)
    if hyper_params["save_model"]:
        os.makedirs(dirname(hyper_params["output_path"]), exist_ok=True)

    print(hyper_params)

    # Initialize a tensorboard writer
    global writer
    path = hyper_params["tensorboard_path"]
    writer = SummaryWriter(path)

    # Train It..
    train_reader, test_reader, val_reader = load_data_fast(hyper_params)

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
        if as_reward:
            raise ValueError("Reward model not implemented for recommender datasets.")
            # print("Reward Training")
            # criterion = RewardLoss(hyper_params)
        else:
            print("Bandit Training")
            if "lse" in hyper_params.experiment.name:
                print("LSE Loss")
                criterion = LSE_LossRec(hyper_params)
            elif "powermean" in hyper_params.experiment.name:
                print("Power Mean Loss")
                criterion = PowerMeanLossRec(hyper_params)
            elif "exponential_smoothing" in hyper_params.experiment.name:
                print("Exponential Smoothing Loss")
                criterion = ExponentialSmoothingLossRec(hyper_params)
            else:
                print("IPS Loss")
                criterion = CustomLossRec(hyper_params)
    elif hyper_params.experiment.feedback is None:
        criterion = None
    else:
        raise ValueError(
            f"Feedback type {hyper_params.experiment.feedback} is not valid."
        )
        
    try:
        best_metrics_total = []
        for exp in range(hyper_params.experiment.n_exp):
            model = ModelRec(*(hyper_params["dataset"]["data_shape"])).to(device)
            print(model)
            optimizer = get_experiment_optimizer(model, hyper_params)
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
            not_improved = 0
            for epoch in range(1, hyper_params["epochs"] + 1):
                epoch_start_time = time.time()

                # Training for one epoch
                metrics = train(
                    trial,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    train_reader,
                    hyper_params,
                    device,
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
                    device,
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
                val_metrics = metrics
                if metrics["SNIPS"] > best_acc:
                    not_improved = 0
                    best_acc = metrics["SNIPS"]

                    metrics = evaluate(
                        model,
                        criterion,
                        test_reader,
                        hyper_params,
                        device,
                    )
                    if hyper_params["save_model"]:
                        torch.save(model.state_dict(), hyper_params["output_path"])
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

                # trial.report(val_metrics["Acc"], epoch)

                # Handle pruning based on the intermediate value.
                # if trial.should_prune():
                #     best_metrics_total.append(best_metrics)
                #     raise optuna.exceptions.TrialPruned()

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
    # trial.report(model_summary["Acc"]["mean"], )

    return model_summary["SNIPS"]["mean"]

def fill_lse_params(hyper_params, args):
    if args.lse_lambda is not None:
        hyper_params["lse_lamda"] = args.lse_lambda
        if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
            hyper_params["ar_alpha"] = 1 / (1 + args.lse_lambda)

    if "lse" in hyper_params.experiment.name:
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "_LSE#Lambda_", "_LSE#Lambda_" + str(hyper_params["lse_lamda"])
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "_LSE#Lambda_", "_LSE#Lambda_" + str(hyper_params["lse_lamda"])
        )
        hyper_params["log_file"] = hyper_params["log_file"].replace(
            "_LSE#Lambda_", "_LSE#Lambda_" + str(hyper_params["lse_lamda"])
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "_LSE#Lambda_", "_LSE#Lambda_" + str(hyper_params["lse_lamda"])
        )

def fill_power_mean_params(hyper_params, args):
    if args.power_mean_lambda is not None:
        hyper_params["power_mean_lamda"] = args.power_mean_lambda

    if "powermean" in hyper_params.experiment.name:
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "_PowerMean#Lambda_", "_PowerMean#Lambda_" + str(hyper_params["power_mean_lamda"])
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "_PowerMean#Lambda_", "_PowerMean#Lambda_" + str(hyper_params["power_mean_lamda"])
        )
        hyper_params["log_file"] = hyper_params["log_file"].replace(
            "_PowerMean#Lambda_", "_PowerMean#Lambda_" + str(hyper_params["power_mean_lamda"])
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "_PowerMean#Lambda_", "_PowerMean#Lambda_" + str(hyper_params["power_mean_lamda"])
        ) 
        
def fill_exponential_smoothing_params(hyper_params, args):
    if args.exs_alpha is not None:
        hyper_params["exs_alpha"] = args.exs_alpha

    if "exponential_smoothing" in hyper_params.experiment.name:
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "_ExponentialSmoothing#Alpha_", "_ExponentialSmoothing#Alpha_" + str(hyper_params["exs_alpha"])
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "_ExponentialSmoothing#Alpha_", "_ExponentialSmoothing#Alpha_" + str(hyper_params["exs_alpha"])
        )
        hyper_params["log_file"] = hyper_params["log_file"].replace(
            "_ExponentialSmoothing#Alpha_", "_ExponentialSmoothing#Alpha_" + str(hyper_params["exs_alpha"])
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "_ExponentialSmoothing#Alpha_", "_ExponentialSmoothing#Alpha_" + str(hyper_params["exs_alpha"])
        )   

def fill_hyperparams(hyper_params, args):
    hyper_params["raw_image"] = args.raw_image
    hyper_params["linear"] = args.linear
    hyper_params["add_unlabeled"] = args.add_unlabeled
    hyper_params["save_model"] = args.save_model
    hyper_params["model_reward"] = args.model_reward  # args.as_reward
    hyper_params["ignore_unlabeled"] = args.ignore_unlabeled
    full_dataset = hyper_params["dataset"]
    dataset = full_dataset.split("/")[0].split("_")[0]
    print("Dataset =", dataset, full_dataset)
    hyper_params["dataset"] = dataset_mapper[dataset]
    hyper_params["dataset"]["name"] = full_dataset
    if args.feature_size is not None:
        hyper_params["feature_size"] = args.feature_size
    elif 'rec' in hyper_params['dataset'] and hyper_params['dataset']['rec']:
        hyper_params["feature_size"] = hyper_params["dataset"]["data_shape"][0]
    else:
        hyper_params["feature_size"] = np.prod(hyper_params["dataset"]["data_shape"])
    if "${UL}" in hyper_params.dataset["name"]:
        ul_string = None
        tau_string = None
        if hyper_params["linear"] and not args.deeplog:
            ul_string = args.ul
            tau_string = args.tau
        else:
            # if dataset == "cifar":
            #     ul_string = f"_ul{args.ul}" if args.ul != "0" else ""
            #     tau_string = f"_tau{args.tau}" if args.tau != "1.0" else ""
            # else:
            ul_string = args.ul
            tau_string = args.tau
        hyper_params.dataset["name"] = hyper_params.dataset["name"].replace(
            "${UL}", ul_string
        )
        hyper_params.dataset["name"] = hyper_params.dataset["name"].replace(
            "${TAU}", tau_string
        )
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "${UL}", ul_string
        )
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "${TAU}", tau_string
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "${UL}", ul_string
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "${TAU}", tau_string
        )
        hyper_params["log_file"] = hyper_params["log_file"].replace("${UL}", ul_string)
        hyper_params["log_file"] = hyper_params["log_file"].replace(
            "${TAU}", tau_string
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "${UL}", ul_string
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "${TAU}", tau_string
        )
    # if hyper_params["raw_image"] and hyper_params["linear"] and not args.deeplog:
    #     hyper_params["dataset"]["name"] = hyper_params["dataset"]["name"].replace(
    #         dataset, dataset + "_raw"
    #     )
    #     hyper_params["summary_file"] = hyper_params["summary_file"].replace(
    #         dataset, dataset + "_raw"
    #     )
    #     hyper_params["log_file"] = hyper_params["log_file"].replace(
    #         dataset, dataset + "_raw"
    #     )
    #     hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
    #         dataset, dataset + "_raw"
    #     )
    if hyper_params["ignore_unlabeled"] and hyper_params["add_unlabeled"] > 0:
        hyper_params["tensorboard_path"] = hyper_params["tensorboard_path"].replace(
            "_KL", "_u" + str(hyper_params["add_unlabeled"]) + "_KL"
        )
        hyper_params["output_path"] = hyper_params["output_path"].replace(
            "_KL", "_u" + str(hyper_params["add_unlabeled"]) + "_KL"
        )
        hyper_params["log_file"] = hyper_params["log_file"].replace(
            "_KL", "_u" + str(hyper_params["add_unlabeled"]) + "_KL"
        )
        hyper_params["summary_file"] = hyper_params["summary_file"].replace(
            "_KL", "_u" + str(hyper_params["add_unlabeled"]) + "_KL"
        )
    if hyper_params["ignore_unlabeled"] and hyper_params["add_unlabeled"] > 0:
        hyper_params["dataset"]["name"] = hyper_params["dataset"]["name"].replace(
            dataset, dataset + "_u" + str(hyper_params["add_unlabeled"])
        )

    #change the config default for lse, power_mean, and exponential smoothing
    fill_lse_params(hyper_params, args)
    fill_power_mean_params(hyper_params, args)
    fill_exponential_smoothing_params(hyper_params, args)
    

    if args.wd is not None:
        hyper_params["weight_decay"] = [args.wd, args.wd]
    if args.kl is not None:
        if "KL" not in hyper_params.experiment.regularizers:
            raise ValueError("Config does not allow KL regularizer.")
        hyper_params.experiment.regularizers.KL = [args.kl, args.kl]
    if args.kl2 is not None:
        if "KL2" not in hyper_params.experiment.regularizers:
            raise ValueError("Config does not allow KL2 regularizer.")
        hyper_params.experiment.regularizers.KL2 = [args.kl2, args.kl2]

    print("Ignoring unlabeled data?", hyper_params["ignore_unlabeled"])
    print("save model:", hyper_params["save_model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to experiment config file."
    )
    parser.add_argument("-d", "--device", required=True, help="Device", type=str)
    parser.add_argument(
        "--linear",
        required=False,
        action="store_true",
        help="If used, the learned policy is a linear model",
    )
    parser.add_argument(
        "-s",
        "--save_model",
        required=False,
        action="store_true",
        help="If used, the trained model is saved.",
    )
    # parser.add_argument("-r", "--as_reward", required=False, action="store_true", help="If used, the trained model is saved.")
    parser.add_argument(
        "-l",
        "--ignore_unlabeled",
        required=False,
        action="store_true",
        help="If used, missing-reward instances are completely ignored.",
    )
    parser.add_argument(
        "--tau",
        required=False,
        type=str,
        help="Softmax temperature for the logging policy.",
    )
    parser.add_argument(
        "--add_unlabeled",
        required=False,
        type=int,
        default=0,
        help="The number of missing-reward instances to be added to the known-reward instances. Only use when ignore_unlabeled is true.",
    )
    parser.add_argument(
        "--ul",
        required=False,
        type=str,
        help="The ratio of missing-reward to known-reward samples.",
    )
    parser.add_argument(
        "--raw_image",
        action="store_true",
        help="If used, raw flatten image is given to the model instead of pretrained features.",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        help="If used, given feature size is supposed for the context.",
    )
    parser.add_argument(
        "--deeplog",
        action="store_true",
        help="If used, dataset generated by deep logging policy is used.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        help="If used, weight decay is manually overwritten.",
    )
    parser.add_argument(
        "--kl",
        type=float,
        help="If used, KL coefficient is manually overwritten.",
    )
    parser.add_argument(
        "--kl2",
        type=float,
        help="If used, KL2 coefficient is manually overwritten.",
    )
    parser.add_argument(
        "--model_reward",
        action="store_true",
        help="If used, reward is estimated and the fully-supervised setting is used.",
    )
    parser.add_argument(
        "--lse_lambda",
        type=float,
        help="If used, LSE_lambda coefficient is manually overwritten.",
    )
    parser.add_argument(
        "--power_mean_lambda",
        type=float,
        help="If used, power mean lambda coefficient is manually overwritten.",
    )
    parser.add_argument(
        "--exs_alpha",
        type=float,
        help="If used, exponential smoothing alpha coefficient is manually overwritten.",
    )
    parser.add_argument("--train_ratio", type=float, help="What ratio of train dataset will be used for training", required=False)
    parser.add_argument("--uniform_noise_alpha", type=float, help="Do we have uniform noise on probs?", required=False)
    parser.add_argument("--gaussian_noise_alpha", type=float, help="Do we have Gaussian noise on probs?", required=False)
    parser.add_argument("--gamma_noise_beta", type=float, help="If used we will have gamma noise on probs", required=False)
    parser.add_argument("--unbalance", type=float, nargs=2, help="If used, we will have unbalance dataset", required=False)
    parser.add_argument("--data_repeat", type=int, help="If used, we will repeat our data records", required=False)
    parser.add_argument("--reward_flip", type=float, help="If used, we will have binary reward flip in our data records", required=False)
    parser.add_argument(
        "--biased_log_policy",
        action="store_true",
        default=None,
        help="If used, biased logging policy will be used.",
    )
    parser.add_argument(
        "--disable_weight_decay",
        action="store_true",
        default=None,
        help="If used, biased logging policy will be used.",
    )#Add assert that config.num_trials must be one when we dont have weight decay

    args = parser.parse_args()
    
    hyper_params = load_hyper_params(args.config, train_ratio=args.train_ratio, uniform_noise_alpha=args.uniform_noise_alpha,
                                     gaussian_noise_alpha=args.gaussian_noise_alpha, gamma_noise_beta=args.gamma_noise_beta, biased_log_policy=args.biased_log_policy, 
                                     disable_weight_decay=args.disable_weight_decay, unbalance=args.unbalance, data_repeat=args.data_repeat,
                                     reward_flip=args.reward_flip)
    fill_hyperparams(hyper_params, args)

    if "lse" in hyper_params.experiment.name:
        print("The LSE lambda parameter is:", hyper_params["lse_lamda"])
        
    if "powermean" in hyper_params.experiment.name:
        print("The Power mean lambda parameter is:", hyper_params["power_mean_lamda"])

    if "exponential_smoothing" in hyper_params.experiment.name:
        print("The Exponential Smoothing alpha parameter is:", hyper_params["exs_alpha"])
    
    if args.unbalance is not None:
        print(f"We will use unbalance dataset {args.unbalance}...")
    
    if args.data_repeat is not None:
        print(f"We will use data repeat dataset= {args.data_repeat}...")
    
    if args.uniform_noise_alpha is not None:
        print(f"We will use Uniform {args.uniform_noise_alpha} noise dataset...")
    
    if args.gamma_noise_beta is not None:
        print(f"We will use gamma noist with beta = {args.gamma_noise_beta} dataset...")
        
    if args.biased_log_policy is not None:
        print("We will use biased logging policy...")
        
    if args.reward_flip is not None:
        print("We will use reward flipping...")

    if args.disable_weight_decay is not None:
        print("We will not use weight decay in this experiment...")

    if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
        print("We are using alpha renyi regularizer...")

    # train main model
    print(
        "############################################################################# TRAINING MAIN MODEL"
    )
    result_path = hyper_params["summary_file"] + "/final.txt"

    print(result_path)
    print(hyper_params["summary_file"])
    print("Dataset =", hyper_params["dataset"])
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(
            main, hyper_params=hyper_params, device=args.device, return_model=False
        ),
        n_trials=hyper_params.experiment.n_trials,
    )
    # best_metrics = main(args.config, device=args.device)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    s = ""
    print("Study statistics: ")
    s += "Study statistics: \n"
    print("  Number of finished trials: ", len(study.trials))
    s += "  Number of finished trials: " + str(len(study.trials)) + "\n"
    print("  Number of pruned trials: ", len(pruned_trials))
    s += "  Number of pruned trials: " + str(len(pruned_trials)) + "\n"
    print("  Number of complete trials: ", len(complete_trials))
    s += "  Number of complete trials: " + str(len(complete_trials)) + "\n"

    print("Best trial:")
    s += "Best trial:\n"
    trial = study.best_trial

    print("  Value: ", trial.value)
    s += "  Value: " + str(trial.value) + "\n"
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        s += "    {}: {}".format(key, value) + "\n"
    file_write(result_path, s)
