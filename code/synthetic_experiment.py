import numpy as np
import argparse
import os
import torch
from scipy.stats import norm, lomax
from tqdm import tqdm
import matplotlib.pyplot as plt

def func_sigmoid(x, alpha):
    return -1 * 1 / (1 + np.exp(-(x - alpha)))

def func_exp(x, alpha):
    return -1 * np.exp(alpha * x**2, dtype=np.float64)

def func_lomax(x, alpha):
    return -1 * (x + 1) ** alpha

class Experiment:
    def __init__(self, n, q_dist_info, p_dist_info, alpha, n_exp, _type, adaptive_lambda):
        self.n = n
        self.Q = q_dist_info
        self.P = p_dist_info
        self.f = func_exp if _type == 'gaussian' else func_lomax
        self.alpha = alpha
        self.n_exps = n_exp
        self.lambdas = {'pm': [0.1, 0.3, 0.5, 0.8, 1.0],
            'lse': [0.001, 0.01, 0.1, 1, 5],
            'es': [0.1, 0.3, 0.5, 0.8, 1.0],
            'ix': [0.01, 0.1, 1.0, 10, 100],
            'tr': [2.0, 5.0, 10.0, 50.0, 100.0],
            'sn': [1.0, 1.0, 1.0, 1.0, 1.0],
            'ls': [0.001, 0.01, 0.1, 1, 5],
            'ops': [0.001, 0.01, 0.1, 1, 5],
        }
        self._type = _type

        # adaptive lambda only affects LSE estimator
        self.adaptive_lambda = adaptive_lambda

    def print_results(self, errors):
        for k, error in errors.items():
            error = np.array(error)
            print(f"---> {k}")
            mses = np.mean(error ** 2, axis=0)
            u = np.argmin(mses)
            mse = mses[u]
            bias = np.mean(error, axis=0)[u]
            var = mse - bias ** 2
            print("Bias:", bias, "Var:", var, "MSE:", mse)
            print("<---")

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
        # print(samples, func_values)
        # for item in zip(samples, q, p, func_values):
        #     print(item)
        return q, p, func_values, epsilon

    def calculate_pm_expected_value(self, pm_lambda, q, p, func_values):
        w =  p / q
        power_mean_w = w / (1 - pm_lambda + (pm_lambda * w))
        return np.mean(func_values * power_mean_w)

    def calculate_es_expected_value(self, es_lambda, q, p, func_values):
        return np.mean(func_values * p / (q ** es_lambda))

    def calculate_ls_expected_value(self, ls_lambda, q, p, func_values):
        return -np.mean((1 / ls_lambda) * p * np.log(1 - ((ls_lambda * func_values) / (q + 1e-8))))
    
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
        calculator = getattr(self, f'calculate_{method}_expected_value')
        if self.adaptive_lambda and method == 'lse':
            _lambda = 1 / len(q) ** (1 / (epsilon + 1))
            # print(_lambda)
            # _lambda = 1.0
            expected_value = calculator(_lambda, q, p, func_values)
            errors = [expected_value - correct_expected_value] * len(self.lambdas[method])
        else:
            for _lambda in self.lambdas[method]:
                expected_value = calculator(_lambda, q, p, func_values)
                errors.append(expected_value - correct_expected_value)
        return errors

    # def run_lse_experiments(self, correct_expected_value, q, p, func_values):
    #     abs_errors = []
    #     errors = []
    #     for lse_lambda in self.lse_lambdas:
    #         lse_expected_value = self.calculate_lse_expected_value(lse_lambda, q, p, func_values)
    #         abs_errors.append(np.abs(lse_expected_value - correct_expected_value))
    #         errors.append(lse_expected_value - correct_expected_value)
    #         # print(lse_expected_value, correct_expected_value)
    #     return errors, abs_errors

    def run(self):
        if self._type == 'gaussian':
            correct_expected_value = -(1/(np.sqrt(1 - 2 * self.alpha * self.P[1]))) * np.exp(self.alpha * (self.P[0] ** 2) / (1 - 2 * self.alpha * self.P[1]))
        else:
            correct_expected_value = -self.P[0] / (self.P[0] - self.alpha)
        # print(self.P[0], self.P[1], self.alpha, correct_expected_value)
        # pm_errors = []
        # lse_errors = []
        # pm_abs_errors = []
        # lse_abs_errors = []
        errors = {
            # 'pm': [],
            # 'es': [],
            # 'lse': [],
            # 'tr': [],
            # 'ix': [],
            # 'sn': [],
            'ls': [],
            'ops': [],
        }

        for _ in range(self.n_exps):
            q, p, func_values, epsilon = self.prepare_samples()
            for k in errors:
                e = self.run_experiments(k, correct_expected_value, q, p, func_values, epsilon)
                # print(self.calculate_mc_expected_value(q, p, func_values), correct_expected_value)
                errors[k].append(e)
        if args.plot:
            print('---> PLOTTING...')
            plt.figure(figsize=(14, 8))
            for k in errors:
                errors_for_all_lambda = np.array(errors[k])
                best_lambda = np.argmax(np.sum(errors_for_all_lambda * 2, axis=0))
                y, x = np.histogram(errors_for_all_lambda[:, best_lambda], bins=50)
                plt.plot(x[:-1], y, label=k, linewidth=2)
            plt.grid()
            plt.legend()
            plt.savefig(f'plots/s2/plot_{self.n}')
        self.print_results(errors)
        print("----------------------------")

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

args = parser.parse_args()

if args.multi_n:
    for n in [50, 100, 500, 1000, 5000, 10000, 50_000, 100_000]:
        print(f"n = {n}:")
        print('-' * 50)
        new_exp = Experiment(n, args.q_dist, args.p_dist, args.alpha, args.n_exp, args.type, args.adaptive_lambda)
        new_exp.run()
else:
    if args.type == 'gaussian':
        for alpha in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
            print(f"ALPHA = {alpha}:")
            print('-' * 50)
            new_exp = Experiment(args.n, (1.0, 0.25), (0.5, 0.25), alpha, args.n_exp, 'gaussian', args.adaptive_lambda)
            new_exp.run()
    else:
        for alpha in [0.5, 1, 2]:
            p_0, p_1 = alpha + 0.5, 1
            for u in [2, 3, 4]:
                q_0, q_1 = (p_0 - alpha) * u, 1
                print(f"BETA = {alpha}, ALPHA = {p_0}, ALPHA' = {q_0}:")
                print('-' * 50)
                new_exp = Experiment(args.n, (q_0, q_1), (p_0, p_1), alpha, args.n_exp, 'lomax', args.adaptive_lambda)
                new_exp.run()



