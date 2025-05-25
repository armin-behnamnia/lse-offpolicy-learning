import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import *
from scipy.optimize import minimize

# ACTION_SIZE = 10


class RewardLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, output, action, delta):
        return F.binary_cross_entropy(output[range(action.size(0)), action], delta)


class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]
        print("Bandit Net Lambda:", self.lamda)

    def forward(self, output, action, delta, prop, switch_t=None):
        risk = -delta
        w = output[range(action.size(0)), action] / prop
        loss = (risk - self.lamda) * (output[range(action.size(0)), action] / prop)
        if switch_t is not None:
            loss = (w <= switch_t) * loss
        return torch.mean(loss)


class IPS_C_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.c = self.hyper_params["ips_c"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        loss = (risk - self.lamda) * (
            output[range(action.size(0)), action] / (prop + self.c)
        )
        return torch.mean(loss)


class TRUNC_IPS_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.trunc = self.hyper_params["ips_trunc"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        loss = (risk - self.lamda) * (
            torch.min(torch.tensor(self.trunc), output[range(action.size(0)), action] / prop)
        )
        return torch.mean(loss)


class CustomLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta

        loss = (risk - self.lamda) * (output / prop)

        return torch.mean(loss)


class RewardEstimatorMSELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]
    
    def forward(self, output, delta, action):
        estimated_rewards = output[range(action.size(0)), action]
        return torch.mean((estimated_rewards - delta) ** 2)



def SecondMomentLoss(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = (h_scores / (prop + 1e-8)) ** 2
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


class LSE_Loss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lse_lamda = hyper_params["lse_lamda"]
        self.lamda = hyper_params["lamda"]

    def calculate_lambda_value(self, output, action, reward, prop):
        if self.lse_lamda == "ADAPTIVE":
            significance = torch.as_tensor(0.1)
            eps = torch.as_tensor(1.0)
            n = action.size(0)
            e_value = torch.exp(torch.as_tensor(1.0))

            w_theta = (output[range(action.size(0)), action] / prop) * reward
            v = torch.mean(torch.pow(w_theta, 1 + eps))
            first_part = (e_value * (1 + eps)) / eps  
            first_part = first_part * (1 - eps + torch.sqrt(torch.pow(1 - eps, 2) + ((8 * eps) / (3 * e_value * (1 + eps)))))
            first_part = torch.pow(first_part, (2 / (1 + eps)))
            
            second_part = torch.pow(torch.log(1 / significance) / (v * n), 1 / (1 + eps))
            
            data_driven_lambda = torch.clip(first_part * second_part, 0.0, 1.0)

            return data_driven_lambda, v

        else:
            return self.lse_lamda, -1

    def forward(self, output, action, delta, prop, switch_t=None):
        risk = -delta
        w = output[range(action.size(0)), action] / prop
        lse_lambda_value, v_value = self.calculate_lambda_value(output, action, delta, prop)
        loss = (
            lse_lambda_value
            * (risk - self.lamda)
            * (output[range(action.size(0)), action] / prop)
        )
        max_loss = torch.amax(loss, dim=-1, keepdim=True)
        loss = torch.exp(loss - max_loss)

        if switch_t is not None:
            loss = ((w <= switch_t) * loss) + 1e-8

        loss = torch.log(torch.mean(loss)) + max_loss
        
        if self.lse_lamda == "ADAPTIVE":
            return (1 / lse_lambda_value) * loss, lse_lambda_value, v_value
        else:
            return (1 / lse_lambda_value) * loss


class LS_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.ls_lambda = hyper_params["ls_lambda"]
        self.lamda = hyper_params["lamda"]
    
    def forward(self, output, action, delta, prop):
        risk = -delta
        pi = output[range(action.size(0)), action]
        pi_0 = prop

        loss = (1 / self.ls_lambda) * pi * torch.log(1 - ((self.ls_lambda * risk) / (pi_0 + 1e-8)))

        return -torch.mean(loss)


class LSE_LossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lse_lamda = hyper_params["lse_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        loss = self.lse_lamda * (risk - self.lamda) * (output / prop)
        max_loss = torch.amax(loss, dim=-1, keepdim=True)
        loss = torch.exp(loss - max_loss)
        loss = torch.log(torch.mean(loss)) + max_loss
        return (1 / self.lse_lamda) * loss


class PowerMeanLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.power_mean_lamda = hyper_params["power_mean_lamda"]
        self.lamda = hyper_params["lamda"]

    def compute_renyi_divergence(self, output, action, prop):
        p = output[range(action.size(0)), action]
        p2 = p * p
        return torch.mean(p2/prop)

    def calculate_lambda_value(self, output, action, risk, prop):
        if self.power_mean_lamda == "ADAPTIVE":
            n = action.size(0)
            significance = torch.as_tensor(0.1)
            d2_Renyi = self.compute_renyi_divergence(output, action, prop)
            lambda_value = torch.sqrt(torch.log(1 / significance) / (3 * d2_Renyi * n))
            if "gamma_noise_beta" in self.hyper_params.keys():
                return torch.clip(lambda_value, 0.0, 1.0)
            else:
                return lambda_value
        else:
            return self.power_mean_lamda

    def forward(self, output, action, delta, prop):
        risk = -delta
        power_mean_lambda_value = self.calculate_lambda_value(output, action, delta, prop)
        #print("Adaptive Powermean lambda value", power_mean_lambda_value)
        w = output[range(action.size(0)), action] / prop
        power_mean_w = w / (1 - power_mean_lambda_value + (power_mean_lambda_value * w))
        loss = (risk - self.lamda) * power_mean_w
        return torch.mean(loss)


class OPS_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        if self.hyper_params["adaptive_lambda"] is not None:
            self.ops_lambda = "ADAPTIVE"
        else:
            self.ops_lambda = hyper_params["ops_lambda"]
        self.lamda = hyper_params["lamda"]
    
    def calculate_adaptive_lambda(self, risk, w):
        def obj(lambda_):
            shrinkage_weight = (lambda_ * w) / (w ** 2 + lambda_)
            estimated_rewards_ = shrinkage_weight * risk
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean((w - shrinkage_weight) ** 2)) * max(risk)
            return bias ** 2 + variance
        return minimize(obj, x0=np.array([1]), bounds=[(0., np.inf)], method='Powell').x
    
    def forward(self, output, action, delta, prop):
        risk = -delta
        w = output[range(action.size(0)), action] / prop
        w2 = w * w

        lambda_value = 0.0
        if self.ops_lambda == "ADAPTIVE":
            lambda_value = self.calculate_adaptive_lambda(delta.detach().cpu().numpy(), w.detach().cpu().numpy())
            lambda_value = torch.from_numpy(lambda_value).to(self.hyper_params["device"])
        else:
            lambda_value = self.ops_lambda

        #print("#################", lambda_value)
        ops_w = (lambda_value * w) / (w2 + lambda_value)
        loss = (risk - self.lamda) * ops_w
        return torch.mean(loss)

class MRDR_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]
        print("Bandit Net Lambda:", self.lamda)

    def forward(self, output, action, delta, prop, switch_t=None):
        risk = -delta
        pi_t = output[range(action.size(0)), action]
        pi_0 = prop
        loss = (risk - self.lamda) * (((1 - pi_0) / (pi_0 * pi_0)) * pi_t)
        return torch.mean(loss)

class PowerMeanLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.power_mean_lamda = hyper_params["power_mean_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        w = output / prop
        power_mean_w = w / (1 - self.power_mean_lamda + (self.power_mean_lamda * w))
        loss = (risk - self.lamda) * power_mean_w
        return torch.mean(loss)


class ExponentialSmoothingLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.alpha = hyper_params["exs_alpha"]
        self.lamda = self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        w = output[range(action.size(0)), action] / torch.pow(prop, self.alpha)
        loss = (risk - self.lamda) * w
        return torch.mean(loss)


class ExponentialSmoothingLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.alpha = hyper_params["exs_alpha"]
        self.lamda = self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        w = output / torch.pow(prop, self.alpha)
        loss = (risk - self.lamda) * w
        return torch.mean(loss)

class DirectMethodLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
    
    def forward(self, output, action, delta, prop, reward_estimates, switch_t=None):
        if reward_estimates is None:
            risk = -delta
            w = output[range(action.size(0)), action]
            return torch.mean(w)
        risk_estimate = -reward_estimates
        temp = output * risk_estimate
        temp = temp.sum(dim=1, keepdim=True)

        if switch_t is not None:
            w = output[range(action.size(0)), action] / prop
            temp = (w > switch_t) * temp

        return torch.mean(temp)

class DoublyRobustLoss(torch.nn.Module):
    def __init__(self, hyper_params, dm_criterion, secondary_criterion):
        super().__init__()
        self.hyper_params = hyper_params
        self.dm_criterion = dm_criterion
        self.secondary_criterion = secondary_criterion
    
    def forward(self, output, action, delta, prop, reward_estimates):
        if reward_estimates is None:
            risk = -delta
            w = output[range(action.size(0)), action]
            return torch.mean(w) 

        switch_t_param = self.hyper_params["switch_t"]
        switch_no_dr_param = self.hyper_params["switch_no_dr"]

        if switch_t_param is not None:
            if switch_no_dr_param is not None:
                first_part = self.secondary_criterion(output, action, delta, prop, switch_t=switch_t_param)
            else:
                reward_diff = delta - reward_estimates[range(action.size(0)), action]
                first_part = self.secondary_criterion(output, action, reward_diff, prop, switch_t=switch_t_param)
            second_part = self.dm_criterion(output, action, delta, prop, reward_estimates, switch_t=switch_t_param)
            loss = first_part + second_part
            return loss
        else:    
            reward_diff = delta - reward_estimates[range(action.size(0)), action]
            loss = self.dm_criterion(output, action, delta, prop, reward_estimates) + self.secondary_criterion(output, action, reward_diff, prop)
            return loss



# def KLLoss(output, action, prop):
#     h_scores = output[range(action.size(0)), action]
#     return torch.sum(h_scores * torch.log(H_scores / prop + 1e-8))


def KLLoss(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = h_scores * torch.log(h_scores / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRec(output, action, prop):
    action_size = torch.max(action) + 1
    h_scores = output
    all_values = h_scores * torch.log(h_scores / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRev(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = prop * torch.log(prop / (h_scores + 1e-8) + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRevRec(output, action, prop):
    action_size = torch.max(action) + 1
    h_scores = output
    all_values = prop * torch.log(prop / (h_scores + 1e-8) + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def AlphaRenyiLoss(output, action, prop, num_classes, hyper_params):
    alpha = hyper_params["ar_alpha"]
    beta = hyper_params["ar_beta"]
    type = hyper_params["ar_type"]

    if abs(alpha - 1) < 0.001:
        return KLLoss(output=output, action=action, prop=prop, action_size=num_classes)

    if type == 1:
        if alpha > 0 and alpha < 1:
            w = torch.pow(prop, alpha) * torch.pow(
                output[range(action.size(0)), action], 1 - alpha
            )
        else:
            w = torch.pow(prop, alpha) / torch.pow(
                output[range(action.size(0)), action], alpha - 1
            )
    else:
        if alpha > 0 and alpha < 1:
            w = torch.pow(output[range(action.size(0)), action], alpha) * torch.pow(
                prop, 1 - alpha
            )
        else:
            w = torch.pow(output[range(action.size(0)), action], alpha) / torch.pow(
                prop, alpha - 1
            )

    w = torch.reshape(w, (len(w), 1))
    one_hot_y = F.one_hot(action, num_classes=num_classes).float()
    action_sum = torch.mm(one_hot_y.T, w)
    action_count = torch.reshape(one_hot_y.sum(dim=0), (num_classes, 1))
    action_mean = action_sum / (action_count + 1e-8)

    all_action_mean = torch.sum(action_mean[action_count != 0])

    return (beta / (alpha - 1)) * torch.log(all_action_mean)


def AlphaRenyiLossRec(output, action, prop, hyper_params):
    alpha = hyper_params["ar_alpha"]
    beta = hyper_params["ar_beta"]
    type = hyper_params["ar_type"]
    num_classes = torch.max(action) + 1

    if abs(alpha - 1) < 0.001:
        return KLLossRec(output=output, prop=prop, action_size=num_classes)

    if type == 1:
        if alpha > 0 and alpha < 1:
            w = torch.pow(prop, alpha) * torch.pow(output, 1 - alpha)
        else:
            w = torch.pow(prop, alpha) / torch.pow(output, alpha - 1)
    else:
        if alpha > 0 and alpha < 1:
            w = torch.pow(output, alpha) * torch.pow(prop, 1 - alpha)
        else:
            w = torch.pow(output, alpha) / torch.pow(prop, alpha - 1)

    w = torch.reshape(w, (len(w), 1))
    one_hot_y = F.one_hot(action, num_classes=num_classes).float()
    action_sum = torch.mm(one_hot_y.T, w)
    action_count = torch.reshape(one_hot_y.sum(dim=0), (num_classes, 1))
    action_mean = action_sum / (action_count + 1e-8)

    all_action_mean = torch.sum(action_mean[action_count != 0])

    return (beta / (alpha - 1)) * torch.log(all_action_mean)


def SupKLLoss(output, action, delta, prop, eps, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = h_scores * torch.log(h_scores * (delta + eps) / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)
