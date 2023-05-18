"""
@created 05/13/2023 - 5:49 PM
@author Kaiwen Zhou
"""
import numpy as np


class Investors_Profile(object):
    def __init__(self, initial_wealth=None, tau_estimate=1, tau_horizon=3):
        self.initial_wealth = initial_wealth
        self.tau_estimate = tau_estimate
        self.tau_horizon = tau_horizon

    @staticmethod
    def M_objective(P_mean: 'ndarray' = None, P_Cov: 'ndarray' = None, obj_type: 'str' = None,
                    **kwargs) -> ('ndarray', 'ndarray'):

        #  To make it compatible with the case where P_mean is in shape (1,n) or (n, 1)
        P_mean = P_mean.squeeze()
        #  Get the number of asset we are dealing with
        num_asset = P_mean.shape[0]

        # Get the current price: P_current & the benchmark allocation: alpha_benchmark
        P_current = None
        alpha_benchmark = None
        for key, value in kwargs.items():
            if key == 'P_current':
                P_current = value
            elif key == 'α_benchmark':
                alpha_benchmark = value

        ########
        # Get the parameters a, B for the linear mapping from P to M
        # #######
        a = None
        B = None

        ####### Absolute Wealth
        if obj_type in ['absolute wealth', 'final wealth']:
            a = np.zeros(num_asset)
            B = np.eye(num_asset)

        ####### Relative Wealth
        if obj_type == 'relative wealth':
            # Getting current price and benchmark allocation from **kwarg
            a = np.zeros(num_asset)
            B = np.eyes(num_asset) - P_current.T @ alpha_benchmark / alpha_benchmark.T @ P_current

        ####### Net Profits
        if obj_type == 'net profits':
            a = -P_current
            B = np.eyes(num_asset)

        #  Find the corresponding mean and covariance for M
        M_mean = a + B @ P_mean
        M_Cov = B @ P_Cov @ B.T
        return M_mean, M_Cov


