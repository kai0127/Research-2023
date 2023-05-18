"""
@created 05/13/2023 - 6:10 PM
@author Kaiwen Zhou
"""
import Estimators
import numpy as np


def invariants_distribution(invariants: 'np.array' = None, estimator_type: 'str' = 'sample'):
    if estimator_type == 'sample':
        # estimate sample mean and sample covariance
        mu_X, Sigma_X = Estimators.sample_mean_cov(invariants)
        return mu_X, Sigma_X

    if estimator_type == 'MLE':
        # estimate sample mean and sample covariance
        mu_X, Sigma_X = Estimators.sample_mean_cov(invariants)
        return mu_X, Sigma_X

    if estimator_type == 'shrinkage':
        # estimate sample mean and sample covariance
        mu_X, Sigma_X = Estimators.shrinkage_mean_cov(invariants)
        return mu_X, Sigma_X

class Invariants_Estimation(object):
    def __init__(self, invariants, estimator_type):
        self.invariants = invariants
        self.mu_X, self.Sigma_X = invariants_distribution(invariants=invariants, estimator_type=estimator_type)

    @staticmethod
    def projection_onto_investment_horizon(tau_estimate=1, tau_horizon=3, mu_X=None, Sigma_X=None, Price_current=None, asset_class=None):
        # Number of assets
        N_assets = Price_current.squeeze().shape[0]

        # Project the invariants onto the investment horizon
        mu_X_invest_horizon = (tau_horizon/tau_estimate)*mu_X
        Sigma_X_invest_horizon = (tau_horizon/tau_estimate)*Sigma_X

        # Recover Prices from the invariants at the investment horizon
        gamma = None
        E = None
        if asset_class in ['equity', 'index', 'commodity']:
            # Construct appropriate \gamma and \epsilon for the given type of assets
            gamma = np.log(Price_current)

            E = np.diagflat(np.ones(N_assets))

        # TODO: Plz add gamma and E for bonds and options also.

        # Project the investment horizon invariants to prices of assets at investment horizon
        # Notice: the prices of assets at investment horizon P_{T+τ} ~ logNormal(μ_P_logN, Σ_P_logN)
        mu_P_inv_hor_logN = (gamma + E @ mu_X_invest_horizon).squeeze()
        Sigma_P_inv_hor_logN = E @ Sigma_X_invest_horizon @ E

        # Find the mean and covariance matrix for P_{T+τ}
        mu_P_inv_hor = np.exp(mu_P_inv_hor_logN + 0.5 * Sigma_P_inv_hor_logN.diagonal())

        Sigma_P_inv_hor = np.zeros((N_assets, N_assets))
        for i in range(N_assets):
            for j in range(N_assets):
                Sigma_P_inv_hor[i, j] = np.exp(
                    mu_P_inv_hor_logN[i]+mu_P_inv_hor_logN[j]
                    + 0.5*(Sigma_P_inv_hor_logN[i, i]+Sigma_P_inv_hor_logN[j, j])
                ) * (np.exp(Sigma_P_inv_hor_logN[i, j])-1)

        return mu_P_inv_hor_logN, Sigma_P_inv_hor_logN, mu_P_inv_hor, Sigma_P_inv_hor
