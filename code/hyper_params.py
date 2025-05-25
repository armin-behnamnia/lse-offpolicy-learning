import yaml
from easydict import EasyDict as edict
import os


def load_config(path):
    with open(path, "r", encoding="utf8") as f:
        return edict(yaml.safe_load(f))

def fill_alpha_renyi_parameters(hyper_params):
    if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
        hyper_params["ar_type"] = hyper_params.experiment.regularizers.AlphaRenyi.type
        hyper_params["ar_beta"] = hyper_params.experiment.regularizers.AlphaRenyi.beta
    return hyper_params

def load_hyper_params(config_path, proportion=None, as_reward=False, create_dir=True, train_ratio=None, uniform_noise_alpha=None,
                      gaussian_noise_alpha=None, gamma_noise_beta=None, biased_log_policy=None, disable_weight_decay=None,
                      unbalance=None, gaussian_imbalance=None, data_repeat=None, reward_flip=None, reward_pareto_noise=None, 
                      reward_lomax_noise=None, logging_policy_cm=None, propensity_estimation=None, ips_c=None, ips_trunc=None, ops_lambda=None, switch_t=None, switch_no_dr=None, ls_lambda=None, 
                      batch_size=None, adaptive_lambda=None, discrete_reward=None, discrete_flip=None, args=None, use_npz=None, use_n_hot=None, device=None, mrdr=None):
    
    hyper_params = load_config(config_path)
    hyper_params["train_ratio"] = train_ratio
    hyper_params["unbalance"] = unbalance
    hyper_params["gaussian_imbalance"] = gaussian_imbalance
    hyper_params["data_repeat"] = data_repeat
    hyper_params["uniform_noise_alpha"] = uniform_noise_alpha
    hyper_params["gaussian_noise_alpha"] = gaussian_noise_alpha
    hyper_params["biased_log_policy"] = biased_log_policy
    hyper_params["disable_weight_decay"] = disable_weight_decay
    hyper_params["reward_flip"] = reward_flip
    hyper_params["reward_pareto_noise"] = reward_pareto_noise
    hyper_params["reward_lomax_noise"] = reward_lomax_noise
    hyper_params["gamma_noise_beta"] = gamma_noise_beta
    hyper_params["logging_policy_cm"] = logging_policy_cm
    hyper_params["propensity_estimation"] = propensity_estimation
    hyper_params["ips_c"] = ips_c
    hyper_params["ips_trunc"] = ips_trunc
    hyper_params["ops_lambda"] = ops_lambda
    hyper_params["switch_t"] = switch_t
    hyper_params["switch_no_dr"] = switch_no_dr
    hyper_params["ls_lambda"] = ls_lambda
    hyper_params["adaptive_lambda"] = adaptive_lambda
    hyper_params["discrete_reward"] = discrete_reward
    hyper_params["discrete_flip"] = discrete_flip
    hyper_params["use_npz"] = use_npz
    hyper_params["use_n_hot"] = use_n_hot
    hyper_params["device"] = device
    hyper_params["mrdr"] = mrdr

    if "DM" in hyper_params.experiment.name or "DR" in hyper_params.experiment.name:
        hyper_params["reward_estimator"] = True
    else:
        hyper_params["reward_estimator"] = None

    if batch_size is not None:
        hyper_params["batch_size"] = batch_size

    hyper_params = fill_alpha_renyi_parameters(hyper_params)

    if proportion is not None:
        hyper_params["train_limit"] = int(50_000 * proportion)
        hyper_params["experiment"]["name"] += "_p_" + str(proportion)
    if as_reward:
        hyper_params.experiment.name += "_REWARD"
    common_path = hyper_params["dataset"]
    common_path += f"_{hyper_params.experiment.name}_"
    
    if disable_weight_decay is not None: #We will not use weight decay
        common_path += "_NO&&WEIGHT&&DECAY_"
        hyper_params["weight_decay"] = None
    else:
        common_path += "_wd_" + str(hyper_params["weight_decay"])


    common_path += "_lamda_" + str(hyper_params["lamda"])
    
    if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
        common_path += "_AR_BETA_" + str(hyper_params["ar_beta"])

    if train_ratio is not None:
        common_path += "_TrainRatio_" + str(train_ratio)
    if unbalance is not None:
        common_path += "_Unbalanced_(" + str(unbalance[0]) + "," + str(unbalance[1]) + ")"
    if gaussian_imbalance is not None:
        common_path += f"_ImbalanceGaussian_{gaussian_imbalance}_"
    if data_repeat is not None:
        common_path += "_Rep=" + str(data_repeat) + "_"
    if uniform_noise_alpha is not None:
        common_path += "_UniformNoise_Alpha_" + str(uniform_noise_alpha)
    if gaussian_noise_alpha is not None:
        common_path += "_GaussianNoise_Alpha_" + str(gaussian_noise_alpha)
    if gamma_noise_beta is not None:
        common_path += "_GammaNoise_Beta_" + str(gamma_noise_beta)
    if biased_log_policy is not None:
        common_path += "_BiasedLoggingPolicy"
    if reward_flip is not None:
        common_path += f"_RewardFlip={reward_flip}_"
    if reward_pareto_noise is not None:
        common_path += f"_FixedRewardNoise_1000_prop={reward_pareto_noise[0]}_"
    if reward_lomax_noise is not None:
        common_path += f"_LomaxRewardNoise_{str(reward_lomax_noise)}_"
    if discrete_reward is not None:
        common_path += f"_DiscreteReward_{str(discrete_reward)}_"
    if discrete_flip is not None:
        common_path += f"_DiscreteFlip_{str(discrete_flip)}_"
    if logging_policy_cm is not None:
        splitted_cfm_path = logging_policy_cm.split("/")
        common_path += f"_CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
    if propensity_estimation is not None:
        common_path += "_Propensity_Estimator"
    if batch_size is not None:
        common_path += f"_BSZ={batch_size}"


    if "lse" in hyper_params.experiment.name or (args is not None and args.lse_lambda is not None):
        common_path += "_LSE#Lambda_"
    if "powermean" in hyper_params.experiment.name or (args is not None and args.power_mean_lambda is not None):
        common_path += "_PowerMean#Lambda_"
    if "exponential_smoothing" in hyper_params.experiment.name or (args is not None and args.exs_alpha is not None):
        common_path += "_ExponentialSmoothing#Alpha_"
    if ips_c is not None or (args is not None and args.ips_c is not None):
        common_path += "_IPS#C_" + str(ips_c)
    if ips_trunc is not None or (args is not None and args.ips_trunc is not None):
        common_path += "_IPS#TRUNC_" + str(ips_trunc)
    if ls_lambda is not None or (args is not None and args.ls_lambda is not None):
        common_path += "_LOGARITHMIC##SMOOTHING_" + str(ls_lambda)
    if ops_lambda is not None or (args is not None and args.ops_lambda is not None):            
        common_path += "_Optimistic##Shrinkage_"
        if adaptive_lambda is not None:
            common_path += "ADAPTIVE"
        else:
            common_path += str(ops_lambda)
    if switch_t is not None or (args is not None and args.switch_t is not None):
        common_path += "_@@SWITCH##Threshold@@_" + str(switch_t)
    if switch_no_dr is not None or (args is not None and args.switch_no_dr is not None):
        common_path += "_!NO_DR!_"

    if mrdr is not None or (args is not None and args.mrdr is not None):
        common_path += "_&&&&MRDR&&&&_"


    common_path += "_FIXED_AVG_ACC"

    hyper_params["tensorboard_path"] = "tensorboard_stuff/" + common_path
    hyper_params["log_file"] = "saved_logs/" + common_path
    hyper_params["summary_file"] = "accs/" + common_path
    hyper_params["output_path"] = "models/outputs/" + common_path
    if create_dir: #false in prepare raw dataset
        os.makedirs(os.path.dirname(hyper_params["tensorboard_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["log_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["summary_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["output_path"]), exist_ok=True)
    return hyper_params
