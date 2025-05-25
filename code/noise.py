from functools import partial
import numpy as np

def uniform_noise_generator(alpha, size):
    return np.random.uniform(-alpha, alpha, size)

def gaussian_noise_generator(alpha, size):
    return np.exp(np.random.normal(loc=0.0, scale=alpha, size=size))

def gamma_noise_generator(alpha, beta, size):
    return 1 / (np.random.gamma(alpha, 1/beta, size) + 0.00001)

class NoiseGenerator:
    def __init__(self, hyper_params):
        self.generator = None
        if hyper_params["uniform_noise_alpha"] is not None:
            self.generator = partial(uniform_noise_generator, alpha=hyper_params["uniform_noise_alpha"])
        if hyper_params["gaussian_noise_alpha"] is not None:
            self.generator = partial(gaussian_noise_generator, alpha=hyper_params["gaussian_noise_alpha"])
        if hyper_params["gamma_noise_beta"] is not None:
            gamma_beta = hyper_params["gamma_noise_beta"]
            self.generator = partial(gamma_noise_generator, alpha=gamma_beta, beta=gamma_beta)
    
    def truncate(self, probs):
        return np.maximum(np.minimum(probs, 1), 0) 
    
    def apply_noise(self, probs):
        if self.generator == None:
            return probs
        elif self.generator.func == uniform_noise_generator:
            return self.truncate(probs + self.generator(size=probs.shape))
        elif self.generator.func == gaussian_noise_generator or self.generator.func == gamma_noise_generator:
            return probs * self.generator(size=probs.shape)