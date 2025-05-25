import numpy as np
import argparse
import os
import torch
from scipy.stats import norm, lomax
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 18})

def func_sigmoid(x, alpha):
    return -1 * 1 / (1 + np.exp(-(x - alpha)))

def func_exp(x, alpha):
    return -1 * np.exp(alpha * x**2, dtype=np.float64)

def func_lomax(x, alpha):
    return -1 * (x + 1) ** alpha

def lomax_noise(n, alpha):
    return np.random.pareto(alpha, size=n)

def gaussian_noise(n, alpha):
    return -np.abs(np.random.normal(loc=0, scale=alpha, size=n)[:, None])

def lomax_noise(n, alpha):
    return -np.random.pareto(a=alpha, size=n)[:, None]

FORMAT = 'svg'

type_to_name = {
    'lse': 'LSE (ours)',
    'lse_adapt': 'LSE Adaptive (ours)',
    'lse_adapt2': 'LSE $\\lambda=1/\\sqrt{n}$',
    'lse_random': 'LSE Random',
    'es': 'ES',
    'pm': 'PM',
    'pm_adapt': 'PM Adaptive',
    'ix': 'IX',
    'ix_adapt': 'IX Adaptive',
    'ops': 'OPS(Shrinkage)',
    'ls': 'LS-LIN',
    'ls_adapt': 'LS-LIN $\\lambda=1/\\sqrt{n}$',
    'ls_nl': 'LS',
    'ls_nl_adapt': 'LS $\\lambda=1/\\sqrt{n}$',
    'ls_nl_random': 'LS Random',
    'tr': 'TR-IPS(Clipped IPS)',
    'tr_adapt': 'TR-IPS Adaptive',
    'sn': 'SNIPS'
}

def calculate_lambda_value(method, p, q, func_values, eps=1.0):
    significance = 0.1
    n = len(p)
    e_value = np.exp(1.0)
    w_theta = np.abs((p / q) * func_values)
    if method == 'lse_adapt':
        v = np.mean(w_theta ** (1 + eps))
        first_part = (e_value * (1 + eps)) / eps  
        first_part = first_part * (1 - eps + np.sqrt((1 - eps) ** 2 + ((8 * eps) / (3 * e_value * (1 + eps)))))
        first_part = first_part ** (2 / (1 + eps))
        
        second_part = (np.log(1 / significance) / (v * n)) ** (1 / (1 + eps))
        out = second_part * first_part
    elif method.endswith('random'):
        out = np.random.random()
    elif method == 'pm_adapt':
        d2_Renyi = np.mean(p ** 2 / q)
        out = (np.log(1 / significance) / (3 * d2_Renyi * n)) ** 0.5
    elif method == 'ix_adapt':
        out = (np.log(2 / significance) / n) ** 0.5
    elif method in ['ls_adapt', 'tr_adapt', 'lsnl_adapt', 'lse_adapt2']:
        out = 1 / n ** 0.5
    else:
        raise ValueError(f'method {method} not valid')
    return out


class Experiment:
    def __init__(self, n, q_dist_info, p_dist_info, alpha, n_exp, _type, adaptive_lambda):
        self.n = n
        self.Q = q_dist_info
        self.P = p_dist_info
        self.f = func_exp if _type == 'gaussian' else func_lomax
        self.alpha = alpha
        self.n_exps = n_exp
        if args.noisy:
            self.lambdas = {'pm': [0.1, 0.3, 0.5, 0.7, 0.9],
                'pm_adapt': [0.01] * 5,
                'lse': [0.001, 0.01, 0.1, 1.0, 10.0],
                'lse_adapt': [0.01] * 5,
                'lse_random': [0.01] * 5,
                'es': [0.1, 0.3, 0.5, 0.7, 0.9],
                'ix': [0.001, 0.01, 0.1, 1.0, 10.0],
                'ix_adapt': [0.01] * 5,
                'tr': [2.0, 5.0, 10.0, 50.0, 100.0],
                'tr_adapt': [0.01] * 5,
                'sn': [1.0, 1.0, 1.0, 1.0, 1.0],
                'ls': [0.001, 0.01, 0.1, 1.0, 10.0],
                'ls_adapt': [0.01] * 5,
                'ops': [0.001, 0.01, 0.1, 1.0, 10.0],
            }
        else:
            self.lambdas = {'pm': [0.1, 0.3, 0.5, 0.8, 1.0],
                'lse': [0.001, 0.01, 0.1, 1, 5],
                'es': [0.1, 0.3, 0.5, 0.8, 1.0],
                'ix': [0.01, 0.1, 1.0, 10, 100],
                'tr': [2.0, 5.0, 10.0, 50.0, 100.0],
                'sn': [1.0, 1.0, 1.0, 1.0, 1.0],
                'ls': [0.001, 0.01, 0.1, 1, 5],
                'lsnl': [0.001, 0.01, 0.1, 1, 5],
                'ops': [0.001, 0.01, 0.1, 1, 5],
                'pm_adapt': [0.01] * 5,
                'lse_adapt': [0.01] * 5,
                'lse_random': [0.01] * 5,
                'lse_adapt2': [0.01] * 5,
                'ls_adapt': [0.01] * 5,
                'lsnl_adapt': [0.01] * 5,
                'lsnl_random': [0.01] * 5,
                'ix_adapt': [0.01] * 5,
                'tr_adapt': [0.01] * 5,
            }
        self._type = _type

        # adaptive lambda only affects LSE estimator
        self.adaptive_lambda = adaptive_lambda

    def print_results(self, errors, noise_levels):
        out = dict()
        for k, error in errors.items():
            out[k] = []
            for i, noise_level in enumerate(noise_levels):
                error_per_noise = error[:, i]
                print(f"---> {k}")
                mses = np.mean(error_per_noise ** 2, axis=0)
                u = np.argmin(mses)
                mse = mses[u]
                bias = np.mean(error_per_noise, axis=0)[u]
                var = mse - bias ** 2
                print("Noise Level=", noise_level,"Bias:", bias, "Var:", var, "MSE:", mse)
                print("<---")
                out[k].append(mse)
        return out
    
    def get_reward_estimate(self, samples, func_values):
        n = len(samples)
        a = n * np.sum(samples * func_values) - np.sum(samples) * np.sum(func_values)
        b = np.sum(samples ** 2) * np.sum(func_values) - np.sum(samples * func_values) * np.sum(samples)
        D = n * np.sum(samples ** 2) - np.sum(samples) ** 2
        plt.scatter(samples, func_values, s=3, c='red')
        plt.axline((0, b/D), (-b/a, 0))
        plt.savefig('test.png')
        exit()
        return a / D, b / D, (a / D) * samples + b / D
    
    def prepare_samples(self):
        if self._type == 'gaussian':
            samples = np.random.normal(self.Q[0], np.sqrt(self.Q[1]), size=(self.n, 1))
            q = norm.pdf(samples, self.Q[0], np.sqrt(self.Q[1]))
            p = norm.pdf(samples, self.P[0], np.sqrt(self.P[1]))
            epsilon = 1.0
        elif self._type == 'lomax':
            samples = np.random.pareto(a=self.Q[0], size=(self.n, 1))
            q = lomax.pdf(samples, self.Q[0])
            p = lomax.pdf(samples, self.P[0])
            k = self.Q[0] / (self.P[0] - self.alpha)
            # print("K =", k)
            epsilon = min(1, 1 / (2 * abs(k - 1))) if k > 1 else 1 
        else:
            raise ValueError(f'Function type {self._type} is not valid.')
        func_values = self.f(samples, self.alpha)
        if args.dr:
            if self._type != 'lomax':
                raise ValueError("Not Implemented.")
            a, b, r_hat = self.get_reward_estimate(samples, func_values)
            dm_est = b + a / (self.P[0] - 1)
            return q, p, func_values, dm_est, r_hat, epsilon
        return q, p, func_values, epsilon

    def calculate_pm_expected_value(self, pm_lambda, q, p, func_values):
        w =  p / q
        power_mean_w = w / (1 - pm_lambda + (pm_lambda * w))
        return np.mean(func_values * power_mean_w)

    def calculate_es_expected_value(self, es_lambda, q, p, func_values):
        return np.mean(func_values * p / (q ** es_lambda))

    def calculate_ls_expected_value(self, ls_lambda, q, p, func_values):
        return -np.mean((1 / ls_lambda) * p * np.log(1 - ((ls_lambda * func_values) / (q + 1e-8))))

    def calculate_lsnl_expected_value(self, ls_lambda, q, p, func_values):
        return -np.mean((1 / ls_lambda) * np.log(1 - ((ls_lambda * func_values * p) / (q + 1e-8))))

    def calculate_ops_expected_value(self, ops_lambda, q, p, func_values):
        w = p / q
        w2 = w * w
        ops_w = (ops_lambda * w) / (w2 + ops_lambda)
        return np.mean(ops_w * func_values)
    
    def calculate_ix_expected_value(self, ix_lambda, q, p, func_values):
        return np.mean(func_values * p / (q + ix_lambda))

    def calculate_tr_expected_value(self, tr_lambda, q, p, func_values):
        return np.mean(func_values * np.minimum(p / q, tr_lambda))

    def calculate_sn_expected_value(self, tr_lambda, q, p, func_values):
        return np.mean(func_values * (p / q)) / np.mean(p / q)

    def calculate_lse_expected_value(self, lse_lambda, q, p, func_values):
        result = lse_lambda * (func_values) * (p / q)
        result = np.exp(result)
        result = np.log(np.mean(result))
        return ((1 / lse_lambda) * result)
    
    def calculate_mc_expected_value(self, q, p, func_values):
        return np.mean(p / q * func_values)

    def run_experiments(self, method, correct_expected_value, q, p, func_values, epsilon):
        errors = []
        if self.adaptive_lambda and (method.endswith('_adapt') or method.endswith('_random') or method.endswith('_adapt2')):
            _lambda = calculate_lambda_value(method, p, q, func_values, eps=epsilon)
            calculator = getattr(self, f'calculate_{method.split("_")[0]}_expected_value')
            # _lambda = 1 / len(q) ** (1 / (epsilon + 1))
            # print(_lambda)
            # _lambda = 1.0
            expected_value = calculator(_lambda, q, p, func_values)
            errors = [expected_value - correct_expected_value] * len(self.lambdas[method])
        else:
            calculator = getattr(self, f'calculate_{method}_expected_value')
            for _lambda in self.lambdas[method]:
                expected_value = calculator(_lambda, q, p, func_values)
                errors.append(expected_value - correct_expected_value)
        return errors


    def run(self, noise_levels):
        if self._type == 'gaussian':
            correct_expected_value = -(1/(np.sqrt(1 - 2 * self.alpha * self.P[1]))) * np.exp(self.alpha * (self.P[0] ** 2) / (1 - 2 * self.alpha * self.P[1]))
        else:
            correct_expected_value = -self.P[0] / (self.P[0] - self.alpha)
        if args.only_lse:
            errors = {
                'lse': [],
                'lse_adapt': [],
                'lse_adapt2': [],
                'lse_random': []
            }
        elif args.lse_ls:
            errors = {
                'lse': [],
                'lse_adapt': [],
                'lse_adapt2': [],
                'lse_random': [],
                'ls': [],
                'lsnl': [],
                'lsnl_adapt': [],
                'lsnl_random': []        
            }
        elif args.no_lse_ls:
            errors = {
                'pm': [],
                'es': [],
                'tr': [],
                'ix': [],
                'sn': [],
                'ops': [],
                'pm_adapt': [],
                'tr_adapt': [],
                'ix_adapt': [],
                'ls_adapt': [],
            }
        elif args.dr:
            errors = {
            'pm': [],
            'es': [],
            'lse': [],
            'tr': [],
            'ix': [],
            'sn': [],
            # 'ls': [],
            'ops': [],
            'lsnl': [],
            }
        else:
            errors = {
                'pm': [],
                'es': [],
                'lse': [],
                'tr': [],
                'ix': [],
                'sn': [],
                'ls': [],
                'ops': [],
                'lse_adapt': [],
                'lse_random': [],
                'lse_adapt2': [],
                'pm_adapt': [],
                'tr_adapt': [],
                'ix_adapt': [],
                'ls_adapt': [],
                'lsnl': [],
                'lsnl_adapt': [],
                'lsnl_random': []        
            }
        for _ in tqdm(range(self.n_exps), total=self.n_exps):
            if args.dr:
                q, p, func_values, dm, r_hat, epsilon = self.prepare_samples()
                func_values -= r_hat # removing estimated r
                # print("MAX, MIN", np.max(func_values), np.min(func_values))
            else:
                q, p, func_values, epsilon = self.prepare_samples()                
            noise_errors = {k: [] for k in errors.keys()}
            for log_noise_level in noise_levels:
                if args.type == 'gaussian':
                    noise_level = np.exp(log_noise_level)
                else:
                    noise_level = log_noise_level
                if self._type == 'gaussian':
                    func_values_noisy = func_values + gaussian_noise(len(q), noise_level)
                    func_values_noisy = np.minimum(func_values_noisy, 0)
                else:
                    func_values_noisy = func_values + lomax_noise(len(q), noise_level)
                    func_values_noisy = np.minimum(func_values_noisy, 0)

                for k in errors:
                    e = self.run_experiments(k, correct_expected_value, q, p, func_values_noisy, epsilon)
                    if args.dr:
                        e = np.array(e) + dm # adding dm estimation
                    noise_errors[k].append(e)
            for k, v in noise_errors.items():
                errors[k].append(v)
        for k, v in errors.items():
            errors[k] = np.array(v)
        if args.plot:
            for i, noise_level in enumerate(noise_levels):
                print('---> PLOTTING...')
                plt.figure(figsize=(14, 8))
                for k in errors.keys():
                    if k in dist_set1:
                        errors_for_all_lambda = np.array(errors[k][:, i])
                        best_lambda = np.argmin(np.sum(errors_for_all_lambda ** 2, axis=0))
                        y, x = np.histogram(errors_for_all_lambda[:, best_lambda], bins=200)
                        plt.plot(x[:-1], y, label=type_to_name[k], linewidth=2)
                plt.grid()
                plt.legend()
                os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}1', exist_ok=True)
                plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}1/plot_{noise_level}.{FORMAT}', format=FORMAT)
                plt.figure(figsize=(14, 8))
                for k in errors.keys():
                    if k not in dist_set1:
                        errors_for_all_lambda = np.array(errors[k][:, i])
                        best_lambda = np.argmin(np.sum(errors_for_all_lambda ** 2, axis=0))
                        y, x = np.histogram(errors_for_all_lambda[:, best_lambda], bins=200)
                        plt.plot(x[:-1], y, label=type_to_name[k], linewidth=2)
                plt.grid()
                plt.legend()
                os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}2', exist_ok=True)
                plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}2/plot_{noise_level}.{FORMAT}', format=FORMAT)
        return self.print_results(errors, noise_levels=noise_levels)

def plot_results(xaxis, yaxis):
    xlabel, xvalues = xaxis
    ylabel, y_values = yaxis
    print('---> PLOTTING...')
    plt.figure(figsize=(14, 8))
    for k, v in y_values.items():
        if k in set1:
            plt.plot(xvalues, v, label=type_to_name[k], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}', exist_ok=True)
    plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/plot_{"noisy" if args.noisy else "normal"}_bad.{FORMAT}', format=FORMAT)

    plt.figure(figsize=(14, 8))
    for k, v in y_values.items():
        if k not in set1:
            plt.plot(xvalues, v, label=type_to_name[k], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}', exist_ok=True)
    plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/plot_{"noisy" if args.noisy else "normal"}_good.{FORMAT}', format=FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=False, help="Number of Samples")
parser.add_argument("--type", type=str, required=True, help="type of experiment. Either gaussian or lomax.")
parser.add_argument("--n_exp", type=int, required=True)
parser.add_argument("--multi_n", action='store_true', required=False)
parser.add_argument("--q_dist", type=float, nargs=2, required=False, help="Q Distribution mean and variance")
parser.add_argument("--p_dist", type=float, nargs=2, required=False, help="P Distribution mean and variance")
parser.add_argument("--alpha", type=float, required=False, help="Reward function parameter alpha")
parser.add_argument("--adaptive_lambda", action='store_true')
parser.add_argument("--plot", action='store_true')
parser.add_argument("--noisy", action='store_true')
parser.add_argument("--only_lse", action='store_true')
parser.add_argument("--lse_ls", action='store_true')
parser.add_argument("--no_lse_ls", action='store_true')
parser.add_argument("--dr", action="store_true")

args = parser.parse_args()

NUM = 11
if args.type == 'gaussian':
    noise_levels = np.linspace(0, np.log(20), num=NUM)
    set1 = ['pm', 'tr', 'sn', 'ops']
    dist_set1 = ['es', 'pm', 'ops', 'sn', 'tr']
else:
    noise_levels = np.linspace(1.05, 2, num=NUM)
    set1 = ['pm', 'tr', 'sn', 'es', 'ops']
    dist_set1 = ['es', 'pm', 'ops', 'sn', 'tr', 'ix']


if args.multi_n:
    for n in [50, 100, 500, 1000, 5000, 10000, 50_000, 100_000]:
        print(f"n = {n}:")
        print('-' * 50)
        new_exp = Experiment(n, args.q_dist, args.p_dist, args.alpha, args.n_exp, args.type, args.adaptive_lambda)
        new_exp.run()
else:
    if args.type == 'gaussian':
        if args.noisy:
            alpha = 1.4
            print(f"ALPHA = {alpha}:")
            print('-' * 50)
            new_exp = Experiment(args.n, (1.0, 0.25), (0.5, 0.25), alpha, args.n_exp, 'gaussian', args.adaptive_lambda)
            mses = new_exp.run(noise_levels)
            strx = 'Log-Std' if args.type == 'gaussian' else '$\\alpha$'
            if args.plot:
                plot_results(xaxis=(strx, noise_levels), yaxis=('MSE', mses))
        else:
            if args.plot:
                alpha = 1.4
                noise_levels = np.linspace(-100, -100, num=1)
                print(f"ALPHA = {alpha}:")
                print('-' * 50)
                new_exp = Experiment(args.n, (1.0, 0.25), (0.5, 0.25), alpha, args.n_exp, 'gaussian', args.adaptive_lambda)
                mses = new_exp.run(noise_levels)
                strx = 'Log-Std' if args.type == 'gaussian' else '$\\alpha$'
                plot_results(xaxis=(strx, noise_levels), yaxis=('MSE', mses))
            else:
                alphas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
                for alpha in alphas:
                    noise_levels = np.linspace(-100, -100, num=1)
                    print(f"ALPHA = {alpha}:")
                    print('-' * 50)
                    new_exp = Experiment(args.n, (1.0, 0.25), (0.5, 0.25), alpha, args.n_exp, 'gaussian', args.adaptive_lambda)
                    mses = new_exp.run(noise_levels)
                    strx = 'Log-Std' if args.type == 'gaussian' else '$\\alpha$'

    else:
        if args.noisy:
            alpha = 2.0
            p = 2.5, 1.0
            q = 1.5, 1.0
            new_exp = Experiment(args.n, q, p, alpha, args.n_exp, 'lomax', args.adaptive_lambda)
            mses = new_exp.run(noise_levels)
            strx = 'Log-Std' if args.type == 'gaussian' else '$\\alpha$'
            if args.plot:
                plot_results(xaxis=(strx, noise_levels), yaxis=('MSE', mses))
        else:
            if args.plot:
                alpha = 2.0
                p = 2.5, 1.0
                q = 1.5, 1.0
                noise_levels = np.linspace(10000, 10000, num=1)
                set1 = []
                dist_set1 = ['sn', 'ops', 'pm', 'tr', 'es']
                new_exp = Experiment(args.n, q, p, alpha, args.n_exp, 'lomax', args.adaptive_lambda)
                mses = new_exp.run(noise_levels)
                strx = 'Log-Std' if args.type == 'gaussian' else '$\\alpha$'
                plot_results(xaxis=(strx, noise_levels), yaxis=('MSE', mses))
            else:
                noise_levels = np.linspace(10000, 10000, num=1)
                for alpha in [2.0]:#[1.5, 1.0, 2.0]:#
                    p = (p_0, p_1) = (alpha + 0.5, 1)
                    for u in [3]:#[2, 3, 4]:#
                        q = (q_0, q_1) = ((p_0 - alpha) * u, 1)
                        print(f"BETA = {alpha}, ALPHA={p[0]}, ALPHA'={q[0]}")
                        print('-' * 50)
                        set1 = []
                        dist_set1 = ['sn', 'ops', 'pm', 'tr', 'es']
                        new_exp = Experiment(args.n, q, p, alpha, args.n_exp, 'lomax', args.adaptive_lambda)
                        mses = new_exp.run(noise_levels)


