"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    shaped_X = X.reshape((X.shape[0],1,X.shape[1])).repeat(mixture.mu.shape[0],axis=1)
    shaped_mu = mixture.mu.reshape((1,mixture.mu.shape[0],mixture.mu.shape[1])).repeat(X.shape[0],axis=0)
    shaped_var = mixture.var.reshape((1,mixture.var.shape[0],1)).repeat(X.shape[0],axis=0)
    shaped_p = mixture.p.reshape((1,mixture.var.shape[0],1)).repeat(X.shape[0],axis=0)

    shaped_var_extended = shaped_var.repeat(X.shape[1],axis=2)
    log_N_X = -1/2*np.log(2*np.pi*shaped_var_extended)-(shaped_X-shaped_mu)**2*np.reciprocal(2*shaped_var_extended)
    log_N_X_clean = np.where(shaped_X == 0, shaped_X, log_N_X).sum(axis=2,keepdims=True)
    f = np.log(shaped_p) + log_N_X_clean.sum(axis=2,keepdims=True)
    lse = logsumexp(f,axis=1,keepdims=True).repeat(f.shape[1],axis=1)
    log_post = f-lse

    post = np.exp(log_post.reshape((log_post.shape[0],log_post.shape[1])))
    ll = logsumexp(f,axis=1,keepdims=True).sum()

    return post, ll



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    nKd_X = X.reshape((n,1,d)).repeat(K,axis=1)
    nKd_post = post.reshape((n,K,1)).repeat(d, axis=2)

    full_sum = np.where(nKd_X == 0, nKd_X, nKd_X*nKd_post).sum(axis=0)
    weights_sum = np.where(nKd_X == 0, nKd_X, nKd_post).sum(axis=0)

    mu = np.where(weights_sum < 1, mixture.mu, full_sum / weights_sum)

    nKd_mu = mu.reshape((1,K,d)).repeat(n,axis=0)

    diffs = np.where(nKd_X == 0, nKd_X, nKd_X - nKd_mu)
    sq_diffs = (diffs*diffs).sum(axis=2)
    var_not_normalized = (sq_diffs*post).sum(axis=0)
    x_ones = np.ones((n,K,d))
    norm_Cu = np.where(nKd_X == 0, nKd_X, x_ones).sum(axis=2)
    denominator = (post*norm_Cu).sum(axis=0)
    threshold = np.ones((K)) * 0.25
    raw_var = (var_not_normalized / denominator)

    var = np.where(raw_var < 0.25, threshold, raw_var)

    pond = post.sum(axis=0) / post.shape[0]

    return GaussianMixture(mu, var, pond)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    current_likelihood = None
    previous_likelihood = None
    while previous_likelihood == None or previous_likelihood - current_likelihood < current_likelihood * 10**(-6):
        previous_likelihood = current_likelihood
        post, current_likelihood = estep(X,mixture)
        mixture = mstep(X,post, mixture)
    return mixture, post, current_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    shaped_X = X.reshape((X.shape[0],1,X.shape[1])).repeat(mixture.mu.shape[0],axis=1)
    shaped_mu = mixture.mu.reshape((1,mixture.mu.shape[0],mixture.mu.shape[1])).repeat(X.shape[0],axis=0)
    shaped_var = mixture.var.reshape((1,mixture.var.shape[0],1)).repeat(X.shape[0],axis=0)
    shaped_p = mixture.p.reshape((1,mixture.var.shape[0],1)).repeat(X.shape[0],axis=0)

    shaped_var_extended = shaped_var.repeat(X.shape[1],axis=2)
    log_N_X = -1/2*np.log(2*np.pi*shaped_var_extended)-(shaped_X-shaped_mu)**2*np.reciprocal(2*shaped_var_extended)
    log_N_X_clean = np.where(shaped_X == 0, shaped_X, log_N_X).sum(axis=2,keepdims=True)
    f = np.log(shaped_p) + log_N_X_clean.sum(axis=2,keepdims=True)
    lse = logsumexp(f,axis=1,keepdims=True).repeat(f.shape[1],axis=1)
    log_post = f-lse

    post = np.exp(log_post.reshape((log_post.shape[0],log_post.shape[1])))

    n, d = X.shape
    K = mixture.mu.shape[0]

    nKd_post = post.reshape(n,K,1).repeat(d,axis=2)
    nKd_mu = mixture.mu.reshape((1,K,d)).repeat(n,axis=0)

    predictions = (nKd_mu * nKd_post).sum(axis=1)

    return np.where(X == 0, predictions, X)
    
