"""
This is a file containing all models with different constraints
@created 05/13/2023 - 8:45 PM
@author Kaiwen Zhou
"""
import cvxopt
import numpy as np


class Model(object):
    def __init__(self):
        return

    @staticmethod
    def MVO_single_period_allocation(N_assets=None,
                                     v_inf=None,
                                     v_sup=None,
                                     v_step=None,
                                     mu_M_inv_hor=None,
                                     Sigma_M_inv_hor=None,
                                     transaction_D=None,
                                     P_T=None,
                                     initial_wealth=None
                                     ):
        reward = []  # given variance_i, the optimal value of investment objective generated by the optimal allocation
        risk = []    # given variance_i, the variance of optimal value of investment objective
        optimal_allocation_curve = []  # given variance_i, the optimal allocation in terms of units of asset
        weights_optimal_curve = []     # given variance_i, the optimal allocation in terms of weights
        variance_range = np.arange(v_inf, v_sup, v_step)

        for target_variances_i in variance_range:

            ###############################
            # Set objective coefficient c #
            ###############################
            c_objective = cvxopt.matrix(-mu_M_inv_hor.T)

            ###################
            # Set Constraints #
            ###################

            ##### Linear (Long-only) Constraint #####
            G_0 = cvxopt.matrix(-np.eye(N_assets))
            h_0 = cvxopt.matrix(np.zeros(N_assets).reshape(-1, 1))

            ##### Quadratic (Risk) Constraint #####
            L_transpose_Sigma = np.linalg.cholesky(Sigma_M_inv_hor)  # Cholesky Decomposition to go align with CVXOPT
            G_1 = cvxopt.matrix(np.r_[np.zeros(N_assets).reshape(1, -1), -L_transpose_Sigma.T])
            h_1 = cvxopt.matrix(np.r_[np.array([target_variances_i]).reshape(-1, 1), np.zeros(N_assets).reshape(-1, 1)])

            ##### Quadratic (Budget) Constraint #####
            L_transpose_D = np.linalg.cholesky(transaction_D)
            G_2 = cvxopt.matrix(np.r_[P_T.reshape(1, -1), -L_transpose_D.T])
            h_2 = cvxopt.matrix(np.r_[np.array([initial_wealth]).reshape(-1, 1), np.zeros(N_assets).reshape(-1, 1)])

            # Aggregate all the constraints that are not component-wise vector inequalities,
            # i.e. all constraints except G_0, h_0
            G = [G_1, G_2]
            h = [h_1, h_2]

            ######################################
            # Solve for optimal allocation curve #
            ######################################
            cvxopt.solvers.options['show_progress'] = False
            alpha_optimal = np.array(cvxopt.solvers.socp(c=c_objective, Gl=G_0, hl=h_0, Gq=G, hq=h)['x']).squeeze()
            optimal_allocation_curve.append(alpha_optimal)

            # Find the corresponding optimal allocation curve but in weights
            w_optimal = np.diagflat(alpha_optimal)@P_T.T/(P_T@alpha_optimal)
            weights_optimal_curve.append(w_optimal)

            # Get the corresponding reward and risk
            reward.append(alpha_optimal@mu_M_inv_hor.squeeze())
            risk.append(alpha_optimal@Sigma_M_inv_hor@alpha_optimal)

        return reward, risk, optimal_allocation_curve, weights_optimal_curve


