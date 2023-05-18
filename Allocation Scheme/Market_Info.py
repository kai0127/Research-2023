"""
@created 05/13/2023 - 5:31 PM
@author Kaiwen Zhou
"""
import numpy as np
import pandas as pd


class Market_Info(object):
    def __init__(self, raw_data, asset_class: 'str' = None):
        self.raw_data = raw_data
        self.asset_class = asset_class

    def estimation_range_info(self, date_start_estimation=None, date_end_estimation=None):
        """
        We specify the estimation range for the raw market data and get
        1. N_assets: the number of assets
        2. N_observations: the number of observations
        3. P_T: Asset prices at current time T
        4. invariants: Assets' corresponding Invariants
        """
        # market data for estimation period: \tilde{t} to T
        Market_data_estimate = self.raw_data[(self.raw_data.index >= date_start_estimation) & (self.raw_data.index <= date_end_estimation)].values

        # number of assets
        N_assets = Market_data_estimate.shape[1]

        # number of observations
        N_observations = Market_data_estimate.shape[0]

        # Prices of asset at current time T
        P_T = Market_data_estimate[-1:].squeeze()

        # Get invariants for the corresponding assets
        invariants = None
        if self.asset_class in ['equity', 'index', 'commodity']:
            # Get market invariants for each asset
            invariants = np.diff(np.log(Market_data_estimate), axis=0)
            print(f'In total, we have {invariants.shape[1]} different invariants for {N_assets} different assets, and '
                  f'we have {invariants.shape[0]} samples for each asset\'s invariant.')

        return N_assets, N_observations, P_T, invariants


