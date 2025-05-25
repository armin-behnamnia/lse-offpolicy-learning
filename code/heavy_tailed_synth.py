import numpy as np
from scipy.stats import norm, lomax, invweibull, cauchy, levy, t as tdist, genextreme
from tqdm import tqdm

true_means = {
    "lomax": -4.8746343722717285,
    "genextreme": -0.7410760221997239,
    "t": -3.843860991687368,
    "invweibull": -5.412080741150039
}
class Sampler:
    def lomax_sample(self, n, alpha=1.2):
        return -np.random.pareto(alpha, size=n)

    def log_cauchy_sample(self, n, alpha=None):
        return -np.exp(cauchy.rvs(size=n))

    def invweibull_sample(self, n, alpha=1.2):
        return -invweibull.rvs(c=alpha, size=n)

    def levy_sample(self, n, alpha=None):
        return -levy.rvs(size=n)

    def t_sample(self, n, alpha=1.2):
        return -np.abs(tdist.rvs(size=n, df=alpha))
    
    def genextreme_sample(self, n, alpha=-0.9):
        return -np.abs(genextreme.rvs(size=n, c=alpha))
    
    def sample(self, dist_name: str, n: int, alpha: float = None) -> np.ndarray:
        if alpha is None:
            return getattr(self, dist_name + "_sample")(n)
        else:  
            return getattr(self, dist_name + "_sample")(n, alpha)  
        
class Experiment:
    def __init__(self, n, q_dist_info, p_dist_info, n_exp, _type):
        self.n = n
        self.n_exps = n_exp
        self.Q = q_dist_info
        self.P = p_dist_info
        self.sampler = Sampler()
        self.type = _type
        self.lambdas = {'pm': [0.1, 0.3, 0.5, 0.8, 1.0],
            'lse': [0.001, 0.01, 0.1, 1, 5],
            'es': [0.1, 0.3, 0.5, 0.8, 1.0],
            'ix': [0.01, 0.1, 1.0, 10, 100],
            'tr': [2.0, 5.0, 10.0, 50.0, 100.0],
            'sn': [1.0, 1.0, 1.0, 1.0, 1.0],
            'ls': [0.001, 0.01, 0.1, 1, 5],
            'lsnl': [0.001, 0.01, 0.1, 1, 5],
            'ops': [0.001, 0.01, 0.1, 1, 5],
        }

        # adaptive lambda only affects LSE estimator

    def print_results(self, errors):
        out = []
        for k, error in errors.items():
            print(f"---> {k}")
            mses = np.mean(error ** 2, axis=0)
            u = np.argmin(mses)
            mse = mses[u]
            bias = np.mean(error, axis=0)[u]
            var = mse - bias ** 2
            print("Bias:", bias, "Var:", var, "MSE:", mse)
            print("<---")
            out.append(mse)
        return out
    
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
        calculator = getattr(self, f'calculate_{method}_expected_value')
        for _lambda in self.lambdas[method]:
            expected_value = calculator(_lambda, q, p, func_values)
            errors.append(expected_value - correct_expected_value)
        return errors

    def prepare_samples(self):
        func_values = self.sampler.sample(dist_name=self.type, n=self.n)[:, None]
        samples = np.random.normal(self.Q[0], np.sqrt(self.Q[1]), size=(self.n, 1))
        q = norm.pdf(samples, self.Q[0], np.sqrt(self.Q[1]))
        p = norm.pdf(samples, self.P[0], np.sqrt(self.P[1]))
        epsilon = 1.0
        return q, p, func_values, epsilon

    def run(self):
        errors = {
            'pm': [],
            'es': [],
            'lse': [],
            'tr': [],
            'ix': [],
            'sn': [],
            'ls': [],
            'ops': [],
            'lsnl': [],
        }
        # get an estimate of the true mean
        # n = self.n
        # self.n = 100_000
        # all_values = []
        # for _ in tqdm(range(10), total=10):
        #     values = []
        #     for _ in tqdm(range(10_000), total=10_000):
        #         q, p, func_values, epsilon = self.prepare_samples()
        #         values.append(np.mean(p / q * func_values))
        #     values = np.mean(values)
        #     all_values.append(values)
        # true_mean = np.median(all_values)
        # self.n = n
        true_mean = true_means[self.type]
        print("TRUE MEAN: ", true_mean)
        for _ in tqdm(range(self.n_exps), total=self.n_exps):
            q, p, func_values, epsilon = self.prepare_samples()
            for k in errors:
                e = self.run_experiments(k, true_mean, q, p, func_values, epsilon)
                errors[k].append(e)
        for k, v in errors.items():
            errors[k] = np.array(v)
        # if args.plot:
        #     for i, noise_level in enumerate(noise_levels):
        #         print('---> PLOTTING...')
        #         plt.figure(figsize=(14, 8))
        #         for k in errors.keys():
        #             if k in dist_set1:
        #                 errors_for_all_lambda = np.array(errors[k][:, i])
        #                 best_lambda = np.argmin(np.sum(errors_for_all_lambda ** 2, axis=0))
        #                 y, x = np.histogram(errors_for_all_lambda[:, best_lambda], bins=200)
        #                 plt.plot(x[:-1], y, label=type_to_name[k], linewidth=2)
        #         plt.grid()
        #         plt.legend()
        #         os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}1', exist_ok=True)
        #         plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}1/plot_{noise_level}.{FORMAT}', format=FORMAT)
        #         plt.figure(figsize=(14, 8))
        #         for k in errors.keys():
        #             if k not in dist_set1:
        #                 errors_for_all_lambda = np.array(errors[k][:, i])
        #                 best_lambda = np.argmin(np.sum(errors_for_all_lambda ** 2, axis=0))
        #                 y, x = np.histogram(errors_for_all_lambda[:, best_lambda], bins=200)
        #                 plt.plot(x[:-1], y, label=type_to_name[k], linewidth=2)
        #         plt.grid()
        #         plt.legend()
        #         os.makedirs(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}2', exist_ok=True)
        #         plt.savefig(f'plots/worst{"_adapt" if args.adaptive_lambda else ""}/{args.type}/{"noisy" if args.noisy else "normal"}2/plot_{noise_level}.{FORMAT}', format=FORMAT)
        return self.print_results(errors)

print("Lomax Experiment: ")
new_exp = Experiment(1000, (1.0, 0.25), (0.5, 0.25), 10000, 'lomax')
new_exp.run()

print("GEV Experiment: ")
new_exp = Experiment(1000, (1.0, 0.25), (0.5, 0.25), 10000, 'genextreme')
new_exp.run()

print("T-students Experiment: ")
new_exp = Experiment(1000, (1.0, 0.25), (0.5, 0.25), 10000, 't')
new_exp.run()

print("Frechet Experiment: ")
new_exp = Experiment(1000, (1.0, 0.25), (0.5, 0.25), 10000, 'invweibull')
new_exp.run()


# print("Log Cauchy Experiment: ")
# new_exp = Experiment(1000, (1.0, 0.25), (0.5, 0.25), 1000, 'log_cauchy')
# new_exp.run()

