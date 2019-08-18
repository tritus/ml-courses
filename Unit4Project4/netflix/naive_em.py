"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    dimensioned_mu = np.expand_dims(mixture.mu, axis=0)
    mu_for_all_x = np.repeat(dimensioned_mu, X.shape[0], axis=0)

    var_dimensioned = np.expand_dims(mixture.var,axis=0).transpose()
    var_for_all_x = np.repeat(np.expand_dims(var_dimensioned,axis=0),X.shape[0],axis=0)

    p_dimensioned = np.expand_dims(mixture.p,axis=0)
    p_for_all_x = np.repeat(p_dimensioned,X.shape[0],axis=0)

    X_dimensioned = np.expand_dims(X,axis=1)
    X_for_mus = np.repeat(X_dimensioned,mixture.mu.shape[0],axis=1)

    X_minus_mu = X_for_mus - mu_for_all_x
    X_minus_mu_squared = X_minus_mu * X_minus_mu
    norms_sq = np.sum(X_minus_mu_squared,axis=2)
    norms_sq_dimensioned = np.expand_dims(norms_sq,axis=1)
    all_norms_sq = np.transpose(norms_sq_dimensioned, axes=[0, 2, 1])

    exponent = -all_norms_sq / (2 * var_for_all_x)
    probabilities = np.reciprocal(2*np.pi*var_for_all_x)**(X.shape[1]/2) * np.exp(exponent)
    probabilities_reshaped = np.reshape(probabilities, (probabilities.shape[0], probabilities.shape[1]))
    ponderated_probabilities = probabilities_reshaped * p_for_all_x
    probabilities_sum = np.sum(ponderated_probabilities,axis=1)

    log_likelihood = np.sum(np.log(probabilities_sum))
    post = ponderated_probabilities / np.reshape(probabilities_sum, (probabilities_sum.shape[0],1))

    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

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
    sq_diffs = (diffs*diffs).sum(axis=2,keepdims=True)
    var_not_normalized = (sq_diffs*shaped_post).sum(axis=0)

    var = (var_not_normalized / (X.shape[1]*weights_sum)).reshape((var_not_normalized.shape[0]))

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
        mixture = mstep(X,post)
    return mixture, post, current_likelihood
