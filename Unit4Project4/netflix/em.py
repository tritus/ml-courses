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
    shaped_X = X.reshape((X.shape[0],1,X.shape[1])).repeat(post.shape[1],axis=1)
    shaped_post = post.reshape((post.shape[0],post.shape[1],1))

    ponderated_points = shaped_X*shaped_post
    full_sum = ponderated_points.sum(axis=0)
    weights_sum = shaped_post.sum(axis=0)

    mu = full_sum / weights_sum

    shaped_mu = mu.reshape((1,mu.shape[0],mu.shape[1])).repeat(X.shape[0],axis=0)
    diffs = shaped_X - shaped_mu
    diffs_clean = np.where(shaped_X == 0, shaped_X, diffs)
    sq_diffs = (diffs_clean*diffs_clean).sum(axis=2,keepdims=True)
    var_not_normalized = (sq_diffs*shaped_post).sum(axis=0)
    x_ones = np.ones((shaped_X.shape[0], shaped_X.shape[1], shaped_X.shape[2]))
    norm_Cu = np.where(shaped_X == 0, shaped_X, x_ones).sum(axis=2, keepdims=True)
    denominator = (shaped_post*norm_Cu).sum(axis=0)

    import pdb;pdb.set_trace()

    var = (var_not_normalized / denominator).reshape((var_not_normalized.shape[0]))

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
    raise NotImplementedError
