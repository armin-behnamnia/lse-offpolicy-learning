import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loss import KLLossRec, KLLossRevRec, AlphaRenyiLossRec
from tqdm import tqdm
from utils import *


def normalized_sigmoid(output, source, batch_idx):
    index = batch_idx
    return torch.sigmoid(output) / torch.scatter_reduce(
        input=torch.zeros(len(output)).to(output.device),
        src=torch.sigmoid(source),
        index=index,
        reduce="sum",
        dim=0,
    )


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


def evaluate(model, criterion, reader, hyper_params, device):
    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0]).to(device)
    main_loss = FloatTensor([0.0]).to(device)
    correct, total = 0, 0.0
    avg_correct = FloatTensor([0.0]).to(device)
    control_variate = FloatTensor([0.0]).to(device)
    ips = FloatTensor([0.0]).to(device)
    tau = hyper_params["tau"] if "tau" in hyper_params else 1.0

    model.eval()

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
        with torch.no_grad():
            source = model(x[batch_idx], item_matrix[all_actions])
            if torch.isnan(source).any():
                exit()
            nonzeros = (all_actions == action[batch_idx]).nonzero()[:, 0]
            output = source[nonzeros]
            assert len(output) == len(action)
            output = group_softmax(output, source, batch_idx)
            if hyper_params.experiment.feedback == "supervised":
                raise ValueError(
                    "Supervised training not implemented for Rec datasets."
                )
            elif hyper_params.experiment.feedback == "bandit":
                # if hyper_params.as_reward:
                #     loss = criterion(output_labeled, action_labeled, reward_labeled)
                # else:
                loss = criterion(output, reward, prop)
            elif hyper_params.experiment.feedback is None:
                loss = torch.tensor(0).float().to(x.device)
            else:
                raise ValueError(
                    f"Feedback type {hyper_params.experiment.feedback} is not valid."
                )
        main_loss += loss.item()
        total_loss += loss.item()
        if hyper_params.experiment.regularizers:
            if "KL" in hyper_params.experiment.regularizers:
                loss += (
                    KLLossRec(
                        output,
                        action,
                        prop,
                    )
                    * hyper_params.experiment.regularizers.KL
                )
            if "KL2" in hyper_params.experiment.regularizers:
                loss += (
                    KLLossRevRec(
                        output,
                        action,
                        prop,
                    )
                    * hyper_params.experiment.regularizers.KL2
                )
            if "AlphaRenyi" in hyper_params.experiment.regularizers:
                loss += AlphaRenyiLossRec(output=output, action=action, prop=prop, hyper_params=hyper_params)
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

        # Metrics evaluation
        total_loss += loss.item()
        control_variate += torch.mean(output / prop).item()
        ips += torch.mean((reward * output) / prop).item()
        # predicted = torch.argmax(output, dim=1)
        # print(predicted, y)
        total += output.size(0)
        correct += (reward == 1).sum().item()
        total_batches += 1.0

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    # metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total_batches, 4)
    metrics["IPS"] = round(float(ips) / total_batches, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)

    return metrics
