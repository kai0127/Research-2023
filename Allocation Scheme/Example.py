"""
This is an example exactly the same as the one in the jupyter notebook 'Meucci's Framework'.
@created 05/13/2023 - 6:49 PM
@author Kaiwen Zhou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Investors_Profile import Investors_Profile
from Market_Info import Market_Info
from Invariants_Estimation import Invariants_Estimation
from Transaction_Cost import Transaction_Cost
from Model import Model
from Index_of_Satisfaction import Index_of_Satisfaction

if __name__ == '__main__':
    # Load data
    df_data = pd.read_csv('weekly_closing_prices_2012_2022.csv', index_col='Date')
    df_data.index = pd.to_datetime(df_data.index)
    print(df_data)

    ######################################
    # Step 1: Determine the investor's profile
    # Variables:
    # 1. N_assets: the number of assets
    # 2. N_observations: the number of observations
    # 3. P_T: Asset prices at current time T
    # 4. invariants: Assets' corresponding Invariants
    ######################################
    IP = Investors_Profile(initial_wealth=1000, tau_estimate=1, tau_horizon=3)

    ######################################
    # Step 2: Gather market information in the estimation range:
    # Variables:
    # 1. N_assets: the number of assets
    # 2. N_observations: the number of observations
    # 3. P_T: Asset prices at current time T
    # 4. invariants: Assets' corresponding Invariants
    ######################################
    Market_Info = Market_Info(df_data, 'equity')
    N_assets, N_observations, P_T, invariants = Market_Info.estimation_range_info(
                                                                date_start_estimation=pd.to_datetime('2012-01-09'),
                                                                date_end_estimation=pd.to_datetime('2017-01-02')
                                                            )

    ######################################
    # Step 3: Estimate the distribution of invariants
    # Variables:
    # 1. mu_X: invariants' mean vector
    # 2. Sigma_X: variance-covariance matrix for invariants
    ######################################
    Invariants_Estimation = Invariants_Estimation(invariants, 'sample')
    mu_X, Sigma_X = Invariants_Estimation.mu_X, Invariants_Estimation.Sigma_X

    ######################################
    # Step 4: - Project the invariants onto the investment horizon
    #         - Recover Prices from invariants
    #         - Convert distribution of P_inv_hor to the more general distribution for M_inv_hor
    # Variables:
    # 1. mu_P_inv_hor_logN: P_inv_hor ~ logN(mu_P_inv_hor_logN, Sigma_P_inv_hor_logN)
    # 2. Sigma_P_inv_hor_logN: P_inv_hor ~ logN(mu_P_inv_hor_logN, Sigma_P_inv_hor_logN)
    # 3. mu_P_inv_hor: mean vector for P_inv_hor, E[P_inv_hor].
    # 4. Sigma_P_inv_hor: variance-covariance matrix for P_inv_hor, V[P_inv_hor].
    # 5. mu_M_inv_hor: mean vector for M_inv_hor
    # 6. Sigma_M_inv_hor: variance-covariance matrix for M_inv_hor
    ######################################
    mu_P_inv_hor_logN, Sigma_P_inv_hor_logN, mu_P_inv_hor, Sigma_P_inv_hor = Invariants_Estimation.projection_onto_investment_horizon(tau_estimate=IP.tau_estimate,
                                                                                                                                      tau_horizon=IP.tau_horizon,
                                                                                                                                      mu_X=mu_X,
                                                                                                                                      Sigma_X=Sigma_X,
                                                                                                                                      Price_current=P_T,
                                                                                                                                      asset_class='equity')
    mu_M_inv_hor, Sigma_M_inv_hor = Investors_Profile.M_objective(P_mean=mu_P_inv_hor, P_Cov=Sigma_P_inv_hor, obj_type='absolute wealth')

    ######################################
    # Step 5: Specify the transaction cost
    # TODO: This is a very simple formulation of transaction cost, do Almgren Chriss Next
    # Variables:
    # 1. v_inf, v_sup, v_step: lower bound, higher bound, and step size for variance's (Risk) upper bound
    # 2. Sigma_P_inv_hor: variance-covariance matrix for P_inv_hor
    ######################################
    transaction_D = Transaction_Cost(N_assets=N_assets).simple_transaction_coefficient_D

    ######################################
    # Step 6: - Introduce allocation framework / model and its associated constraints
    #         - Find the optimal allocation curve using the optimization package CVXOPT
    #         - Find the optimal allocation using the index of satisfaction
    # NOTE: This is a very ad-hoc step, so it's not going to be very systematic.
    # Variables:
    # 1. v_inf, v_sup, v_step: lower bound, higher bound, and step size for variance's (Risk) upper bound
    # 2. Sigma_P_inv_hor: variance-covariance matrix for P_inv_hor
    ######################################
    reward, risk, optimal_allocation_curve, weights_optimal_curve = Model().MVO_single_period_allocation(
                                                                                        N_assets=N_assets,
                                                                                        initial_wealth=IP.initial_wealth,
                                                                                        P_T=P_T,
                                                                                        transaction_D=transaction_D,
                                                                                        v_inf=0,
                                                                                        v_sup=100,
                                                                                        v_step=1,
                                                                                        mu_M_inv_hor=mu_M_inv_hor,
                                                                                        Sigma_M_inv_hor=Sigma_M_inv_hor,
                                                                                    )

    ######################################
    # Step 6: Find the optimal allocation using Index of Satisfaction
    # Variables:
    # 1. N_MC: number of iterations for Monte-Carlo Scheme
    # 2. IoS_optimal_allocation_curve: the value of Index of Satisfaction for each optimal allocation (alpha_optimal)
    ######################################
    N_MC = 100  # number of iterations for Monte-Carlo Scheme
    IoS = Index_of_Satisfaction(optimal_allocation_curve=optimal_allocation_curve)
    IoS_optimal_allocation_curve = IoS.certainty_equivalent(utility_coef=-9,
                                                            P_mean=mu_P_inv_hor_logN,
                                                            P_Cov=Sigma_P_inv_hor_logN,
                                                            N_MC=N_MC,
                                                            CE_type='power')
    # Then, the optimal allocation is
    optimal_allocation = optimal_allocation_curve[np.argmax(IoS_optimal_allocation_curve)]
    print(f'Given the conditions provided and the restriction imposed above, the optimal allocation is {optimal_allocation}.')



    plt.figure()
    for i in range(N_assets):
        plt.scatter(risk, np.array(optimal_allocation_curve)[:, i], label=f'optimal alpha {i}')
    plt.legend()

    plt.figure()
    for i in range(N_assets):
        plt.scatter(risk, np.array(weights_optimal_curve)[:, i], label=f'optimal weight {i}')

    plt.figure()
    plt.scatter(np.array(risk), reward, s=3)

    plt.figure()
    plt.scatter(np.array(risk), IoS_optimal_allocation_curve, s=3)

    plt.show()








