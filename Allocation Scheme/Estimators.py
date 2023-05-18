"""
@created 05/13/2023 - 6:13 PM
@author Kaiwen Zhou
"""
import numpy as np


def sample_mean_cov(ndarray: 'np.array'):
    """
    Input: ndarray where each column represents the total samples for each invariant

    Sample mean and sample covariance
    """
    #  compute the mean
    mean = np.mean(ndarray, axis=0)
    #  compute the covariance
    cov = (1/ndarray.shape[0])*np.matmul((ndarray-mean).T, (ndarray-mean))
    return mean, cov


def shrinkage_mean_cov(ndarray: 'np.array'):
    """
    Input: ndarray where each column represents the total samples for each invariant

    Sample mean and sample covariance
    """
    # number of observations
    N_obs = ndarray.shape[0]
    # number of invariants
    N_invariants = ndarray.shape[1]

    #  compute the mean
    mean = np.mean(ndarray, axis=0)
    #  compute the covariance
    cov = (1/ndarray.shape[0])*np.matmul((ndarray-mean).T, (ndarray-mean))

    ########### Get shrinkage covariance
    # Compute epsilon
    numerator = 0
    for i in range(N_obs):
        numerator += np.trace((ndarray[i, :]@ndarray[i, :].T - cov)@(ndarray[i, :]@ndarray[i, :].T - cov))/N_obs

    denominator = 0
    for i in range(N_invariants):
        denominator += np.trace((cov-np.eye(N_invariants)*(cov[i, i]/N_invariants))@(cov-np.eye(N_invariants)*(cov[i, i]/N_invariants)))

    epsilon = (1/N_obs) * numerator/denominator

    cov_shrinkage = (1-epsilon)*cov + (epsilon/N_invariants)*np.trace(cov)*np.eye(N_invariants)

    ########### Get shrinkage mean
    e = np.ones(N_invariants)
    b = (e.T@np.linalg.inv(cov_shrinkage)@mean / (e.T@np.linalg.inv(cov_shrinkage)@e))*e
    largest_eigenvalue = np.amax(np.linalg.eigvals(cov_shrinkage))
    gamma = (np.trace(cov_shrinkage)-2*largest_eigenvalue)/((mean-b).T@(mean-b))/N_obs

    mean_shrinkage = (1-gamma)*mean + gamma*b
    return mean_shrinkage, cov_shrinkage