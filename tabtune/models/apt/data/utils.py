import math
import random
import numpy as np
import scipy.stats as stats

def sample_trunc_norm(mu, sigma, low=0, high=1000000):
    return stats.truncnorm((low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]

def sample_trunc_norm_log_scaled(min_mu, max_mu, min_sigma=0.01, max_sigma=1., low=0, high=1000000):
    log_mu = np.random.uniform(math.log(min_mu), math.log(max_mu))
    log_sigma = np.random.uniform(math.log(min_sigma), math.log(max_sigma))

    mu = math.exp(log_mu)
    sigma = math.exp(log_mu)*math.exp(log_sigma)

    return sample_trunc_norm(mu, sigma, low=low, high=high)

def sample_trunc_norm_log_scaled_int(min_mu, max_mu, min_sigma=0.01, max_sigma=1., shift=2, low=0, high=1000000):
    log_mu = np.random.uniform(math.log(min_mu), math.log(max_mu))
    log_sigma = np.random.uniform(math.log(min_sigma), math.log(max_sigma))

    mu = math.exp(log_mu)
    sigma = math.exp(log_mu)*math.exp(log_sigma)

    return round(sample_trunc_norm(mu, sigma, low=low, high=high-shift) + shift)

def sample_trunc_gamma_int(max_k, max_mu, shift=2, high=300):
    log_k = np.random.uniform(0, math.log(max_k))
    mu = np.random.uniform(0, max_mu)

    k = math.exp(log_k)
    theta = mu / k

    return min(round(np.random.gamma(k, theta) + shift), high)

def sample_trunc_beta_min_max(min_p, max_p, min_b=0.1, max_b=5.0):
    b = np.random.uniform(min_b, max_b)
    k = np.random.uniform(min_b, max_b)

    return min_p + (max_p - min_p) * np.random.beta(b, k)

def sample_zero_inflated_uniform(zero_p, max_v):
    if random.random() < zero_p:
        return 0.
    else:
        return np.random.uniform(0, max_v)
