import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import argparse
from hyper_params import load_hyper_params
import os
from data import load_data_fast
import datetime as dt
from loss import RewardEstimatorMSELoss
import time
from tqdm import tqdm
from eval import evaluate_regression_model
import copy
import yaml

STOP_THRESHOLD = 20

def get_reward_estimator_model_name(dataset, linear, tau, args):
    model_name = f"models/{dataset}/reward_estimator_"

    if args.gamma_noise_beta is not None:
        model_name += f"gamma_noise_{args.gamma_noise_beta}_"
    if args.gaussian_imbalance is not None:
        model_name += f"gaussian_imbalance_{args.gaussian_imbalance}_"
    if args.reward_pareto_noise is not None:
        model_name += f"reward_pareto_noise_{args.reward_pareto_noise}_"
    if args.reward_lomax_noise is not None:
        model_name += f"reward_lomax_noise_{args.reward_lomax_noise}_"
    if args.discrete_reward is not None:
        model_name += f"_discrete_reward_{args.discrete_reward}"
    if args.discrete_flip is not None:
        model_name += f"_discrete_flip_{args.discrete_flip}"
    
    model_name += f"base_"

    model_name += f"{'deep' if not linear else 'linear'}{'_raw' if args.raw_image else ''}_tau{tau}.pth"

    return model_name


def fill_hyperparams(hyper_params, args):
    hyper_params["raw_image"] = args.raw_image
    hyper_params["linear"] = args.linear
    hyper_params["save_model"] = args.save_model
    hyper_params["ignore_unlabeled"] = args.ignore_unlabeled
    full_dataset = hyper_params["dataset"]
    dataset = full_dataset.split("/")[0].split("_")[0]
    print("Dataset =", dataset, full_dataset)
    hyper_params["dataset"] = dataset_mapper[dataset]
    hyper_params["dataset"]["name"] = full_dataset
    if args.feature_size is not None:
        hyper_params["feature_size"] = args.feature_size
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

    if args.wd is not None:
        hyper_params["weight_decay"] = [args.wd, args.wd]

    print("Ignoring unlabeled data?", hyper_params["ignore_unlabeled"])
    print("save model:", hyper_params["save_model"])


def train_one_epoch(model, criterion, optimizer, scheduler, reader, hyper_params, device):
    model.train()

    metrics = {}
    total_batches = 0.0
    main_loss = FloatTensor([0.0])
    tau = hyper_params["tau"] if "tau" in hyper_params else 1.0
    
    print("------------> Training with temperature =", tau)
    # print(model)
    for x, y, action, delta, prop, _, reward_estimates in tqdm(reader):
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
        output = F.sigmoid(output)
        
        loss = criterion(output, delta, action)

        main_loss += loss.item()

        loss.backward()
        optimizer.step()

        total_batches += 1.0

    scheduler.step()

    metrics["MSE"] = float(main_loss) / total_batches

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
    hyper_params=None,
):

    global STOP_THRESHOLD
    print(f"Feature Size = {feature_size}")

    hyper_params["tau"] = tau

    # Train

    train_reader, test_reader, val_reader = load_data_fast(
        hyper_params, device=device, labeled=False
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

    criterion = RewardEstimatorMSELoss(hyper_params=hyper_params)

    try:
        best_metrics_total = []
        for exp in range(hyper_params.experiment.n_exp):
            not_improved = 0

            #regression model for reward estimation
            model = nn.Linear(
                hyper_params["feature_size"], hyper_params["dataset"]["num_classes"]
            )

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
            best_mse = 10000000000.0
            best_metrics = None
            best_model_dict = None
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

                # Calulating the metrics on the validation set
                metrics = evaluate_regression_model(
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

                if metrics["MSE"] < best_mse:
                    not_improved = 0
                    best_mse = metrics["MSE"]
                    best_model_dict = copy.deepcopy(model.state_dict())
                    metrics = evaluate_regression_model(
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
        "--uniform_noise_alpha",
        type=float,
        help="Do we have uniform noise on probs?",
        required=False,
    )
    parser.add_argument(
        "--gaussian_noise_alpha",
        type=float,
        help="Do we have Gaussian noise on probs?",
        required=False,
    )
    parser.add_argument(
        "--gamma_noise_beta",
        type=float,
        help="If used we will have gamma noise on probs",
        required=False,
    )
    parser.add_argument(
        "--gaussian_imbalance",
        type=float,
        help="If used, we will have gaussian imbalance dataset",
        required=False,
    )
    parser.add_argument(
        "--reward_flip",
        type=float,
        help="If used, we will have binary reward flip in our data records",
        required=False,
    )
    parser.add_argument(
        "--reward_pareto_noise",
        type=float,
        nargs=3,
        help="If used, we will have pareto noise on our reward. Parameters: 1- noise prop, 2- pareto_a, 3- pareto_m",
        required=False
    )
    parser.add_argument(
        "--reward_lomax_noise",
        type=float,
        help="If used, we will have lomax noise on our reward. Parameters: alpha",
        required=False
    )
    parser.add_argument(
        "--biased_log_policy",
        action="store_true",
        default=None,
        help="If used, biased logging policy will be used.",
    )

    parser.add_argument(
        "--propensity_estimation",
        action="store_true",
        default=None,
        help="If used, we will use estiamted propensity scores",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="If used, we will change batch size",
        required=False,
    )

    parser.add_argument(
            "--disable_weight_decay",
            action="store_true",
            default=None,
            help="If used, biased logging policy will be used.",
    ) 

    parser.add_argument(
        "--discrete_reward", 
        type=float, 
        nargs="*", 
        help="If used, we will have descrete distribution reward with descrete values", 
        required=False)
    
    parser.add_argument(
        "--discrete_flip",
        type=float,
        nargs=2,
        help="If used, we will have discrete flip",
        required=False)    

    args = parser.parse_args()

    hyper_params = load_hyper_params(
            args.config,
            uniform_noise_alpha=args.uniform_noise_alpha,
            gaussian_noise_alpha=args.gaussian_noise_alpha,
            gamma_noise_beta=args.gamma_noise_beta,
            biased_log_policy=args.biased_log_policy,
            gaussian_imbalance=args.gaussian_imbalance,
            reward_flip=args.reward_flip,
            reward_pareto_noise=args.reward_pareto_noise,
            reward_lomax_noise=args.reward_lomax_noise,
            propensity_estimation = args.propensity_estimation,
            batch_size=args.batch_size,
            disable_weight_decay=args.disable_weight_decay,
            discrete_reward=args.discrete_reward,
            discrete_flip=args.discrete_flip
        )
    fill_hyperparams(hyper_params, args)

    args = parser.parse_args()
    tau = args.tau
    dataset = hyper_params["target_dataset"]
    linear = args.linear
    hyper_params["reward_estimator_training"] = True

    print(hyper_params)

    model_dict = main(
        args.config,
        device=args.device,
        return_model=True,
        tau=tau,
        dataset=dataset,
        linear=linear,
        raw_image=args.raw_image,
        feature_size=args.feature_size,
        hyper_params=hyper_params
    )
        
    os.makedirs(f"models/{dataset}/", exist_ok=True)

    model_name = get_reward_estimator_model_name(dataset, linear, tau, args)

    torch.save(
        model_dict,
        model_name,
    )  