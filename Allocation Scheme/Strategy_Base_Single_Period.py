"""
This is a place I store strategy at.
All the strategies accounts for SINGLE PERIOD investing.
@created 05/13/2023 - 11:46 PM
@author Kaiwen Zhou
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import impyute as impy
from numpy.linalg import inv
from Estimators import sample_mean_cov


class Strategy_Base_Single_Period(object):
    def __init__(self):
        return


    def MVO(self,
            df_data=None,
            initial_wealth=1000,
            tau_estimate=1,
            tau_horizon=3,
            date_start_estimation=None,
            date_end_estimation=None,
            estimator_type='sample',
            obj_type='absolute wealth',
            asset_class='equity',
            CE_type='power',
            utility_coef=-9,
            N_MC = 100,
            v_inf=0,
            v_sup=100,
            v_step=1,):
        """
        This is a classical MVO strategy
        :return: optimal allocation 'np.array'
        """
        from Investors_Profile import Investors_Profile
        from Market_Info import Market_Info
        from Invariants_Estimation import Invariants_Estimation
        from Transaction_Cost import Transaction_Cost
        from Model import Model
        from Index_of_Satisfaction import Index_of_Satisfaction

        ######################################
        # Step 1: Determine the investor's profile
        # Variables:
        # 1. N_assets: the number of assets
        # 2. N_observations: the number of observations
        # 3. P_T: Asset prices at current time T
        # 4. invariants: Assets' corresponding Invariants
        ######################################
        IP = Investors_Profile(initial_wealth=initial_wealth, tau_estimate=tau_estimate, tau_horizon=tau_horizon)

        ######################################
        # Step 2: Gather market information in the estimation range:
        # Variables:
        # 1. N_assets: the number of assets
        # 2. N_observations: the number of observations
        # 3. P_T: Asset prices at current time T
        # 4. invariants: Assets' corresponding Invariants
        ######################################
        Market_Info = Market_Info(df_data, asset_class=asset_class)
        N_assets, N_observations, P_T, invariants = Market_Info.estimation_range_info(
            date_start_estimation=date_start_estimation,
            date_end_estimation=date_end_estimation
        )

        ######################################
        # Step 3: Estimate the distribution of invariants
        # Variables:
        # 1. mu_X: invariants' mean vector
        # 2. Sigma_X: variance-covariance matrix for invariants
        ######################################
        Invariants_Estimation = Invariants_Estimation(invariants, estimator_type=estimator_type)
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
                                                                                                                                          asset_class=asset_class)
        mu_M_inv_hor, Sigma_M_inv_hor = Investors_Profile.M_objective(P_mean=mu_P_inv_hor, P_Cov=Sigma_P_inv_hor, obj_type=obj_type)

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
            v_inf=v_inf,
            v_sup=v_sup,
            v_step=v_step,
            mu_M_inv_hor=mu_M_inv_hor,
            Sigma_M_inv_hor=Sigma_M_inv_hor,
        )

        ######################################
        # Step 6: Find the optimal allocation using Index of Satisfaction
        # Variables:
        # 1. N_MC: number of iterations for Monte-Carlo Scheme
        # 2. IoS_optimal_allocation_curve: the value of Index of Satisfaction for each optimal allocation (alpha_optimal)
        ######################################
        N_MC = N_MC  # number of iterations for Monte-Carlo Scheme
        IoS = Index_of_Satisfaction(optimal_allocation_curve=optimal_allocation_curve)
        IoS_optimal_allocation_curve = IoS.certainty_equivalent(utility_coef=utility_coef,
                                                                P_mean=mu_P_inv_hor_logN,
                                                                P_Cov=Sigma_P_inv_hor_logN,
                                                                N_MC=N_MC,
                                                                CE_type=CE_type)

        # Then, the optimal allocation is
        optimal_allocation = optimal_allocation_curve[np.argmax(IoS_optimal_allocation_curve)]
        return optimal_allocation

    @staticmethod
    def BL_APT(df_data=None,
               df_data_risk_free=None,
               df_data_market_cap=None,
               df_data_SP500=None,
               initial_wealth=1000,
               tau_estimate=1,
               tau_horizon=3,
               date_start_estimation=None,
               date_end_estimation=None,
               minimum_interval=None,
               trailing_window=None,
               estimator_type='sample',
               obj_type='absolute wealth',
               asset_class='equity',
               CE_type='power',
               utility_coef=-9,
               N_MC = 100,
               v_inf=0,
               v_sup=100,
               v_step=1, ):
        """
        This is a Black-Litterman-APT strategy
        :return: optimal allocation 'np.array'
        """
        from Investors_Profile import Investors_Profile
        from Market_Info import Market_Info
        from Invariants_Estimation import Invariants_Estimation
        from Transaction_Cost import Transaction_Cost
        from Model import Model
        from Index_of_Satisfaction import Index_of_Satisfaction

        ######################################
        # Step 1: Determine the investor's profile
        # Variables:
        # 1. N_assets: the number of assets
        # 2. N_observations: the number of observations
        # 3. P_T: Asset prices at current time T
        # 4. invariants: Assets' corresponding Invariants
        ######################################
        IP = Investors_Profile(initial_wealth=initial_wealth, tau_estimate=tau_estimate, tau_horizon=tau_horizon)

        ######################################
        # Step 2: Gather market information in the estimation range:
        # Variables:
        # 1. N_assets: the number of assets
        # 2. N_observations: the number of observations
        # 3. P_T: Asset prices at current time T
        # 4. invariants: Assets' corresponding Invariants
        ######################################
        Market_Info = Market_Info(df_data, asset_class=asset_class)
        N_assets, N_observations, P_T, invariants = Market_Info.estimation_range_info(
            date_start_estimation=date_start_estimation,
            date_end_estimation=date_end_estimation
        )

        ######################################
        # Step 3: Estimate the distribution of invariants
        # NOTE:Here is exactly where the Black-Litterman-APT model comes into play
        # Variables:
        # 1. mu_X: invariants' mean vector
        # 2. Sigma_X: variance-covariance matrix for invariants
        ######################################

        # Find the invariants, the compounded rate of return
        a = np.r_[np.ones(N_assets).reshape(1, -1)*np.nan, invariants]
        imputed_a = impy.imputation.ts.locf(a.T, axis=0)
        #print(imputed_a)
        Market_data_estimate = df_data[(df_data.index >= date_start_estimation) & (df_data.index <= date_end_estimation)]
        df_rate_of_return = pd.DataFrame(data=imputed_a, columns=Market_data_estimate.columns, index=Market_data_estimate.index)

        #################################################
        # APT Step: Estimate the mean of f and cov of f #
        #           using historical data of f-process  #
        #           over a full scale dataset, or a     #
        #           trailing window.                    #
        #################################################
        # list to store the historical data of f and residual epsilon in 'r = Xf+epsilon'
        all_f = []
        all_epsilon = []
        # The last X in the estimation trailing window
        X_date_end_estimation = None

        risk_free_last = None
        # number of estimation we need for estimating risk-factor f from date_start_estimation to date_end_estimation
        N_estimates = int((date_end_estimation-date_start_estimation)/minimum_interval)

        # From the date_start_estimation to date_end_estimation, we examine N_estimates trailing window of
        # fixed length.
        # For each window:
        # 1. Specify window_date_start_estimation and window_date_end_estimation
        # 2. Restrict all of our data to the fixed window
        # 3. Find the corresponding factor loading X_t where t=window_date_end_estimation
        # 4.
        # ignore the last one
        for i in range(1, N_estimates+1):
            # Specify our x weeks window/full scale dataset
            window_date_end_estimation = date_start_estimation + i*minimum_interval
            if trailing_window is not None:
                window_date_start_estimation = window_date_end_estimation - trailing_window*minimum_interval
            else:
                window_date_start_estimation = date_start_estimation
            # Specify our x weeks window/full scale dataset for estimating f and the associated residual epsilon
            df_esti_market_cap = df_data_market_cap[(df_data_market_cap.index >= window_date_start_estimation) & (df_data_market_cap.index <= window_date_end_estimation)]
            df_esti_rate_of_return = df_rate_of_return[(df_rate_of_return.index >= window_date_start_estimation) & (df_rate_of_return.index <= window_date_end_estimation)]
            df_esti_risk_free_rate = df_data_risk_free[(df_data_risk_free.index >= window_date_start_estimation) & (df_data_risk_free.index <= window_date_end_estimation)]
            SP500_esti_weekly_returns = df_data_SP500[(df_data_SP500.index >= window_date_start_estimation) & (df_data_SP500.index <= window_date_end_estimation)]

            # Column names (Preparation)
            tickers = df_data.columns
            column_risk_free = df_esti_risk_free_rate.columns[0]
            column_SP500_return = SP500_esti_weekly_returns.columns[0]

            ##################################################
            # Market Betas and volatility: Simple Regression #
            ##################################################
            market_betas = []
            volatilities = []
            for ticker in tickers:
                ticker_excess_return = np.array(df_esti_rate_of_return[ticker]-df_esti_risk_free_rate[column_risk_free])
                market_excess_return = np.array(SP500_esti_weekly_returns[column_SP500_return] - df_esti_risk_free_rate[column_risk_free])
                ticker_beta = (1/(market_excess_return@market_excess_return))*market_excess_return.T@ticker_excess_return
                market_betas.append(ticker_beta)
                ticker_vol = np.sqrt((ticker_excess_return-ticker_beta*market_excess_return)@(ticker_excess_return-ticker_beta*market_excess_return))
                volatilities.append(ticker_vol)

            ######################
            # Size  and Momentum #
            ######################
            size = []
            for ticker in tickers:
                ticker_size = df_esti_market_cap[ticker].iloc[-1]
                size.append(ticker_size)

            momentum = []
            for ticker in tickers:
                ticker_rate_of_returns = df_esti_rate_of_return[ticker]
                # ticker_momentum = np.exp(np.exp(np.sum(ticker_rate_of_returns)))
                ticker_momentum = np.exp(np.sum(np.log(1+ticker_rate_of_returns)))-1
                # ticker_momentum = np.sum(ticker_rate_of_returns)
                momentum.append(ticker_momentum)

            # Aggregate factor loadings
            X = [market_betas, size, volatilities, momentum]
            X = np.array(X).T

            # get the last X in the trailing window
            if window_date_end_estimation == date_end_estimation:
                X_date_end_estimation = X
                risk_free_last = df_esti_risk_free_rate.iloc[-1].values

            # Find the OLS estimator for f and store it and associated residual epsilon
            # in all_f and all_epsilon respectively
            r_current_excess = df_esti_rate_of_return.iloc[-1].values - df_esti_risk_free_rate.iloc[-1].values
            f = np.linalg.inv(X.T@X)@X.T@r_current_excess
            all_f.append(f)
            all_epsilon.append(r_current_excess-X@f)

        # convert all_f and all_epsilon matrix to a numpy array
        all_f = np.array(all_f)
        all_epsilon = np.array(all_epsilon)

        #############################################################
        # Bayesian Step: Find the posterior mean and variance for r #
        #############################################################

        ##### Step 1: Prior specification -- \pi(\theta) ~ N(\xi, V) #####
        # Priors: Here we used Data-Driven Priors, i.e. empirical mean and variance for f
        mean_f, cov_f = sample_mean_cov(all_f)
        xi, V = mean_f, cov_f

        ##### Step 2: Computing f(r|\theta) using APT model #####
        # Find the historical mean and variance for epsilon in 'r = Xf+epsilon'
        mean_epsilon, cov_epsilon = sample_mean_cov(all_epsilon)
        Sigma_APT = X_date_end_estimation@cov_f@X_date_end_estimation.T + cov_epsilon

        # Our view: mean_f in the investment horizon will be the same as it is now
        # Our Confidence: Specified in Omega
        q = all_f[-1, :]
        Omega = np.eye(len(q))*0.0009

        # posterior distribution for theta (i.e. \theta|q)
        V_f_with_view_q = inv(inv(Omega) + inv(V))
        xi_f_with_view_q = inv(inv(Omega) + inv(V))@(inv(V)@xi + inv(Omega)@q)

        # posterior distribution for r (i.e. r|q)
        cov_r_with_view_q = inv(inv(Sigma_APT) - inv(Sigma_APT)@X_date_end_estimation@inv(inv(V_f_with_view_q)+X_date_end_estimation.T@inv(Sigma_APT)@X_date_end_estimation)@X_date_end_estimation.T@inv(Sigma_APT))
        mean_r_with_view_q = cov_r_with_view_q@inv(Sigma_APT)@X_date_end_estimation@inv(inv(V_f_with_view_q)+X_date_end_estimation.T@inv(Sigma_APT)@X_date_end_estimation)@inv(V_f_with_view_q)@xi_f_with_view_q

        Sigma_X = cov_r_with_view_q
        mu_X = mean_r_with_view_q + risk_free_last

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
                                                                                                                                          asset_class=asset_class)
        mu_M_inv_hor, Sigma_M_inv_hor = Investors_Profile.M_objective(P_mean=mu_P_inv_hor, P_Cov=Sigma_P_inv_hor, obj_type=obj_type)
        # print('mu_M_inv_hor:', mu_M_inv_hor)
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
            v_inf=v_inf,
            v_sup=v_sup,
            v_step=v_step,
            mu_M_inv_hor=mu_M_inv_hor,
            Sigma_M_inv_hor=Sigma_M_inv_hor,
        )

        ######################################
        # Step 6: Find the optimal allocation using Index of Satisfaction
        # Variables:
        # 1. N_MC: number of iterations for Monte-Carlo Scheme
        # 2. IoS_optimal_allocation_curve: the value of Index of Satisfaction for each optimal allocation (alpha_optimal)
        ######################################
        N_MC = N_MC  # number of iterations for Monte-Carlo Scheme
        IoS = Index_of_Satisfaction(optimal_allocation_curve=optimal_allocation_curve)
        IoS_optimal_allocation_curve = IoS.certainty_equivalent(utility_coef=utility_coef,
                                                                P_mean=mu_P_inv_hor_logN,
                                                                P_Cov=Sigma_P_inv_hor_logN,
                                                                N_MC=N_MC,
                                                                CE_type=CE_type)

        # Then, the optimal allocation is
        optimal_allocation = optimal_allocation_curve[np.argmax(IoS_optimal_allocation_curve)]
        return optimal_allocation

