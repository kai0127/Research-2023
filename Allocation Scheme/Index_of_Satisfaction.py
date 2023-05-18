"""
@created 05/13/2023 - 10:12 PM
@author Kaiwen Zhou
"""
import numpy as np

class Index_of_Satisfaction(object):
    def __init__(self, optimal_allocation_curve):
        self.optimal_allocation_curve = optimal_allocation_curve
        return

    def certainty_equivalent(self, utility_coef=None, P_mean: 'np.array' = None, P_Cov: 'np.array' = None, N_MC=None, CE_type: 'str'=None):
        np.random.seed(66)
        if CE_type == 'power':
            IoS_optimal_allocation_curve = []
            for alpha_optimal in self.optimal_allocation_curve:
                sum_objective=0
                for P_j in np.exp(np.random.multivariate_normal(P_mean.squeeze(), P_Cov, N_MC)):
                    sum_objective += (P_j@alpha_optimal)**((utility_coef-1)/utility_coef)
                IoS_optimal_allocation_curve.append((sum_objective/N_MC)**(utility_coef/(utility_coef-1)))
            return IoS_optimal_allocation_curve
        if CE_type == 'exponential':
            return

    def VaR(self, P_mean: 'np.array'=None , P_Cov: 'np.array'=None):

        return

    def Exepcted_Shortfall(self, P_mean: 'np.array'=None , P_Cov: 'np.array'=None):

        return
