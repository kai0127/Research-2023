"""
@created 05/13/2023 - 9:02 PM
@author Kaiwen Zhou
"""
import numpy as np


class Transaction_Cost(object):
    def __init__(self, N_assets=None):
        self.simple_transaction_coefficient_D = np.eye(N_assets)*0.001
